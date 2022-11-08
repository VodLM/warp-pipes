"""
    Heavily based on:
    - https://github.com/stanford-futuredata/ColBERT/blob/master/colbert/indexing/faiss_index_gpu.py
    - https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py
"""
from __future__ import annotations

import math
import re
import sys
import time
from multiprocessing.pool import ThreadPool
from typing import List
from typing import Optional
from typing import Union

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import rich
import torch
from loguru import logger
from tqdm import tqdm

from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler
from warp_pipes.support.tensor_handler import TensorLike

FaissMetric = Union[str, int]

index_factory_pattern = pat = re.compile(
    "(OPQ[0-9]+(_[0-9]+)?|,PCAR[0-9]+)?,?" "(IVF[0-9]+)," "(PQ[a-zA-Z0-9]+|Flat)"
)

pq_pattern = re.compile("(PQ[0-9]+)(x[0-9]+)?(fs|fsr)?")


class FaissFactory:
    """Custom parsing of the index factory strings"""

    def __init__(self, factory: str):
        self.factory = factory

        matchobject = index_factory_pattern.match(factory)

        if not matchobject:
            raise ValueError(f"Could not parse factory string: `{factory}`")

        mog = matchobject.groups()
        self.preproc = mog[0]
        self.ivf = mog[2]
        self.pq = mog[3]
        self.n_centroids = int(self.ivf[3:])

        if self.pq.startswith("PQ"):
            pq_mog = pq_pattern.match(self.pq)
            if not pq_mog:
                raise ValueError(f"Could not parse PQ string: `{self.pq}`")

            pq_groups = pq_mog.groups()

            self.pq_ncodes = int(pq_groups[0][2:])
            if pq_groups[1] is None:
                self.pq_nbits = 8
            else:
                self.pq_nbits = int(pq_groups[1][1:])
            if pq_groups[2] is None:
                self.pq_type = "full"
            else:
                self.pq_type = pq_groups[2]
        else:
            self.pq_ncodes = None
            self.pq_nbits = None
            self.pq_type = None

    @property
    def clean(self):
        return "-".join(self.factory.split(","))

    def __repr__(self):
        return (
            f"FaissFactory("
            f"Preproc={self.preproc}, "
            f"IVF={self.ivf}, "
            f"PQ={self.pq}, "
            f"centroids={self.n_centroids})"
        )


class IdentityVectorTransform:
    """a pre-processor is either a faiss.VectorTransform or an IdentityVectorTransform"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def faiss_sanitize(x: TensorLike, force_numpy: bool = False) -> TensorLike:
    """ convert array to a c-contiguous float array """
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).contiguous()
        if force_numpy:
            x = x.cpu().numpy()
        return x
    elif isinstance(x, np.ndarray):
        return np.ascontiguousarray(x.astype("float32"))
    else:
        raise TypeError(f"{type(x)} is not supported")


def rate_limited_imap(f, seq):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in seq:
        res_next = pool.apply_async(f, (i,))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""
    handler = TensorHandler(TensorFormat.NUMPY)

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs)) for i0 in range(0, nb, bs)]

    def prepare_block(i01):
        i0, i1 = i01
        xb = handler(x, key=slice(i0, i1))
        xb = faiss_sanitize(xb, force_numpy=True)
        if preproc is not None:
            xb = preproc.apply_py(xb)
        return i0, xb

    return rate_limited_imap(prepare_block, block_ranges)


def get_gpu_resources(devices=None, tempmem: int = -1):
    """Return a list of GPU resources."""
    gpu_resources = []
    if devices is None:
        ngpu = torch.cuda.device_count()
    else:
        ngpu = len(devices)
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        # res.setLogMemoryAllocations(True)
        if tempmem >= 0:
            logger.warning(
                f"Setting GPU:{i} temporary memory to {tempmem / 1024 ** 3:.2f} GB"
            )
            res.setTempMemory(tempmem)

        gpu_resources.append(res)

    return gpu_resources


def make_vres_vdev(gpu_resources, i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = len(gpu_resources)
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def train_preprocessor(
    preproc_str, *, vectors: TensorLike, n_train: int = None
) -> faiss.VectorTransform | IdentityVectorTransform:
    """Train a faiss preprocessor VectorTransform."""
    d = vectors.shape[1]

    if preproc_str.startswith("OPQ"):
        fi = preproc_str[3:-1].split("_")
        m = int(fi[0])
        dout = int(fi[1]) if len(fi) == 2 else d
        preproc = faiss.OPQMatrix(d, m, dout)
    elif preproc_str.startswith("PCAR"):
        dout = int(preproc_str[4:-1])
        preproc = faiss.PCAMatrix(d, dout, 0, True)
    else:
        assert False
    logger.info(
        f"Train Preprocessor: "
        f"{preproc_str} with max. "
        f"{n_train or math.inf} vectors.."
    )
    t0 = time.time()
    y = faiss_sanitize(vectors[:n_train], force_numpy=True)
    preproc.train(y)
    logger.info(f"Trained Preprocessor in {(time.time() - t0):.2f}s")
    return preproc


def compute_centroids(
    vectors: TensorLike,
    *,
    n_centroids: int,
    gpu_resources: List,
    preproc: faiss.VectorTransform,
    max_points_per_centroid: int = 10_000_000,
    n_train: int = None,
    faiss_metric: FaissMetric = faiss.METRIC_INNER_PRODUCT,
) -> np.ndarray:
    """Train the centroids for the IVF index."""
    # get training vectors
    if n_train is not None:
        n_train = max(n_train, 256 * n_centroids)
    vectors = faiss_sanitize(vectors[:n_train], force_numpy=True)

    # define the Quantizer
    d = preproc.d_out
    clus = faiss.Clustering(d, n_centroids)
    clus.verbose = True
    clus.max_points_per_centroid = max_points_per_centroid

    # preprocess te vectors
    vectors = preproc.apply_py(vectors)

    # move the index to CUDA
    vres, vdev = make_vres_vdev(gpu_resources=gpu_resources)
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlat(d, faiss_metric)
    )

    # train the index
    logger.info(
        f"Train clustering index with {n_centroids} centroids "
        f"with max. {n_train} vectors.."
    )
    t0 = time.time()
    clus.train(vectors, index)
    logger.info(f"Trained clustering index in {(time.time() - t0):.2f}s")
    centroids = faiss.vector_float_to_array(clus.centroids)
    return centroids.reshape(n_centroids, d)


def build_and_train_ivf_index(
    vectors: TensorLike,
    *,
    faiss_factory: FaissFactory,
    preproc: faiss.VectorTransform,
    coarse_quantizer: faiss.IndexFlat,
    n_train: int = None,
    faiss_metric: FaissMetric = faiss.METRIC_INNER_PRODUCT,
    use_float16: bool = True,
) -> faiss.IndexIVF:
    """Build the full IFV"""
    pqflat_str = faiss_factory.pq
    n_centroids = faiss_factory.n_centroids
    dimension = preproc.d_out
    if pqflat_str == "Flat":
        logger.info("Making an IVFFlat index")
        ivf_index = faiss.IndexIVFFlat(
            coarse_quantizer, dimension, n_centroids, faiss_metric
        )
    else:
        logger.info(
            f"Making an IVFPQ index with {faiss_factory.pq} ("
            f"{faiss_factory.pq_ncodes} codes, "
            f"{faiss_factory.pq_nbits} bits, "
            f"type {faiss_factory.pq_type})"
        )
        if faiss_factory.pq_type == "full":
            ivf_index = faiss.IndexIVFPQ(
                coarse_quantizer,
                dimension,
                n_centroids,
                faiss_factory.pq_ncodes,
                faiss_factory.pq_nbits,
                faiss_metric,
            )
        elif faiss_factory.pq_type == "fs":
            if faiss_factory.pq_nbits != 4:
                raise ValueError("Only 4 bits are supported for FastScan quantizer")
            ivf_index = faiss.IndexIVFPQFastScan(
                coarse_quantizer,
                dimension,
                n_centroids,
                faiss_factory.pq_ncodes,
                faiss_factory.pq_nbits,
                faiss_metric,
            )
        else:
            raise ValueError(f"Unknown PQ type {faiss_factory.pq_type}")

    coarse_quantizer.this.disown()
    ivf_index.own_fields = True

    # finish training on CPU
    # select vectors
    vectors = faiss_sanitize(vectors[:n_train], force_numpy=True)

    # preprocess te vectors
    vectors = preproc.apply_py(vectors)

    t0 = time.time()
    logger.info("Training IVF index...")
    ivf_index.train(vectors)
    logger.info(f"Trained IVF index in {(time.time() - t0):.2f}s")

    return ivf_index


def populate_ivf_index(
    cpu_index: faiss.IndexIVF,
    *,
    preproc: faiss.VectorTransform,
    vectors: TensorLike,
    gpu_resources: List,
    max_add_per_gpu: int = 100_000,
    use_float16: bool = True,
    use_precomputed_tables=False,
    add_batch_size=65536,  # todo: try reducing this
):
    """Add elements to a sharded index. Return the index and if available
    a sharded gpu_index that contains the same data."""
    ngpu = len(gpu_resources)
    if max_add_per_gpu is not None and max_add_per_gpu >= 0:
        max_add = max_add_per_gpu * max(1, ngpu)
    else:
        max_add = vectors.shape[0]

    # cloner options
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = max_add
    co.shard = True
    assert co.shard_type in (0, 1, 2)

    # define resources and create the GPU shards
    vres, vdev = make_vres_vdev(gpu_resources=gpu_resources)
    gpu_index: faiss.IndexShards = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, cpu_index, co
    )

    # add the vectors
    logger.info(f"Populating the index with vectors of shape {vectors.shape}")
    t0 = time.time()
    nb = vectors.shape[0]
    for i0, xs in tqdm(
        dataset_iterator(vectors, preproc, add_batch_size),
        desc=f"Adding vectors (bs={add_batch_size}, max_add={max_add})",
        total=nb // add_batch_size,
    ):
        i1 = i0 + xs.shape[0]
        if np.isnan(xs).any():
            logger.warning(f"NaN detected in vectors {i0}-{i1}")
            xs[np.isnan(xs)] = 0

        gpu_index.add_with_ids(xs, np.arange(i0, i1))
        if 0 < max_add < gpu_index.ntotal:
            logger.info(
                f"Reached max. size per GPU ({max_add}), " f"flushing indices to CPU"
            )
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                index_src.copy_subset_to(cpu_index, 0, 0, nb)
                index_src_gpu.reset()
                index_src_gpu.reserveMemory(max_add)
            try:
                gpu_index.sync_with_shard_indexes()
            except AttributeError:
                gpu_index.syncWithSubIndexes()

        sys.stdout.flush()
    logger.info("Populating time: %.3f s" % (time.time() - t0))

    logger.info("Aggregate indexes to CPU")
    t0 = time.time()

    if hasattr(gpu_index, "at"):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            logger.info("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(cpu_index, 0, 0, nb)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(cpu_index, 0, 0, nb)

    logger.info("Indexed aggregated in %.3f s" % (time.time() - t0))

    del gpu_index
    return cpu_index


def get_sharded_gpu_index(
    cpu_index: faiss.Index,
    use_float16: bool = True,
    use_precomputed_tables: bool = False,
    devices: Optional[List[int]] = None,
    replicas: int = 1,
    tempmem: int = -1,
) -> faiss.IndexShards:
    # define the gpu resources
    gpu_resources = get_gpu_resources(devices=devices, tempmem=tempmem)
    ngpu = len(gpu_resources)

    # define the cloner options
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = 0
    co.verbose = True
    co.shard = True  # the replicas will be made "manually"
    t0 = time.time()

    rich.print(
        f"[magenta]Co: {co}, "
        f"gpu_resources: {gpu_resources}, "
        f"ngpu: {ngpu}, "
        f"replicas: {replicas}, "
        f"tempmem: {tempmem}, "
        f"use_float16: {use_float16}, "
        f"use_precomputed_tables: {use_precomputed_tables}, "
    )

    if replicas == 1:
        vres, vdev = make_vres_vdev(gpu_resources)
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
    else:
        logger.info("Copy CPU index to %d sharded GPU indexes" % replicas)

        index = faiss.IndexReplicas()

        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)

            print("   dispatch to GPUs %d:%d" % (gpu0, gpu1))

            index1 = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
            index1.this.disown()
            index.addIndex(index1)
        index.own_fields = True
    logger.info("Moved index to GPU in %.2f s" % (time.time() - t0))
    return gpu_index
