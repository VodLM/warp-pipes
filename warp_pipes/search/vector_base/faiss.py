from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import torch
from loguru import logger

from .base import VectorBase
from .base import VectorBaseConfig
from .utils.faiss import build_and_train_ivf_index
from .utils.faiss import compute_centroids
from .utils.faiss import faiss_sanitize
from .utils.faiss import get_gpu_resources
from .utils.faiss import get_sharded_gpu_index
from .utils.faiss import IdentityVectorTransform
from .utils.faiss import populate_ivf_index
from .utils.faiss import train_preprocessor
from warp_pipes.support.tensor_handler import TensorFormat
from warp_pipes.support.tensor_handler import TensorHandler
from warp_pipes.support.tensor_handler import TensorLike


class FaissVectorBase(VectorBase):
    _config_type: type = VectorBaseConfig

    def __init__(self, config: VectorBaseConfig):
        super().__init__(config)
        self.config = config

        logger.info(
            f"Using {type(self).__name__}({self.config.index_factory}, "
            f"shard={self.config.shard}, "
            f"nprobe={self.config.nprobe})"
        )
        logger.info(self.config.factory)

        # index attributes
        self.index = None  # delete the index, and create a new one in `train`
        self.preprocessor = None

    def train(self, vectors: TensorLike, **kwargs):
        gpu_resources = get_gpu_resources(tempmem=self.config.tempmem)
        handler = TensorHandler(TensorFormat.NUMPY)
        vectors = handler(vectors)

        # build the preprocessor
        self.preprocessor = self._build_preprocessor(vectors)

        if self.config.train_on_cpu or len(gpu_resources) == 0:
            self.index = self._build_ivf_index_cpu(vectors)
        else:
            self.index = self._build_ivf_index(gpu_resources, vectors, **kwargs)

        # set nprobe
        self.index.nprobe = self.config.nprobe

    def add(self, vectors: TensorLike, **kwargs):
        gpu_resources = get_gpu_resources(tempmem=self.config.tempmem)

        if self.config.train_on_cpu or len(gpu_resources) == 0:
            self.index = self._populate_ivf_index_cpu(self.index, vectors=vectors)

        else:
            self.index = populate_ivf_index(
                self.index,
                preproc=self.preprocessor,
                vectors=vectors,
                gpu_resources=gpu_resources,
                max_add_per_gpu=self.config.max_add_per_gpu,
                use_float16=self.config.use_float16,
                use_precomputed_tables=self.config.use_precomputed_tables,
                add_batch_size=self.config.add_batch_size,
            )

        # set nprobe
        self.index.nprobe = self.config.nprobe

    def _build_ivf_index(self, gpu_resources, vectors, **kwargs):
        # find the centroids and return a FlatIndex
        coarse_quantizer = self._build_coarse_quantizer(
            vectors, gpu_resources=gpu_resources, **kwargs
        )
        # build the index
        index = build_and_train_ivf_index(
            vectors,
            faiss_factory=self.config.factory,
            preproc=self.preprocessor,
            coarse_quantizer=coarse_quantizer,
            faiss_metric=self.config.faiss_metric,
            use_float16=self.config.use_float16,
        )
        return index

    def _build_ivf_index_cpu(self, vectors):
        index = faiss.index_factory(
            self.config.dimension,
            self.config.index_factory,
            self.config.faiss_metric,
        )
        vectors = faiss_sanitize(vectors)
        index.train(vectors)
        return index

    def _populate_ivf_index_cpu(self, index, *, vectors: TensorLike):
        handler = TensorHandler(TensorFormat.NUMPY)
        for i in range(0, vectors.shape[0], self.config.add_batch_size):
            v = handler(vectors, key=slice(i, i + self.config.add_batch_size))
            v = faiss_sanitize(v, force_numpy=True)
            index.add(v)
        return index

    def _build_preprocessor(
        self, vectors: TensorLike
    ) -> faiss.VectorTransform | IdentityVectorTransform:
        if self.config.factory.preproc is not None:
            return train_preprocessor(self.config.factory.preproc, vectors=vectors)
        else:
            return IdentityVectorTransform(self.config.dimension)

    def _build_coarse_quantizer(self, vectors: TensorLike, **kwargs) -> faiss.IndexFlat:
        centroids = compute_centroids(
            vectors=vectors,
            preproc=self.preprocessor,
            faiss_metric=self.config.faiss_metric,
            n_centroids=self.config.factory.n_centroids,
            **kwargs,
        )

        # build a FlatIndex containing the centroids
        coarse_quantizer = faiss.IndexFlat(
            self.preprocessor.d_out, self.config.faiss_metric
        )
        coarse_quantizer.add(centroids)

        return coarse_quantizer

    @staticmethod
    def index_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "index.faiss"

    @staticmethod
    def preproc_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "preproc.faiss"

    def save(self, path: PathLike):
        index_path = self.index_file(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path.as_posix())
        preproc_path = self.preproc_file(path)
        if isinstance(self.preprocessor, faiss.VectorTransform):
            faiss.write_VectorTransform(self.preprocessor, preproc_path.as_posix())

    def load(self, path: PathLike):
        index_path = self.index_file(path)
        self.index = faiss.read_index(index_path.as_posix())
        self.index.nprobe = self.config.nprobe
        preproc_path = self.preproc_file(path)
        if preproc_path.exists():
            self.preprocessor = faiss.read_VectorTransform(preproc_path.as_posix())
        else:
            self.preprocessor = IdentityVectorTransform(self.config.dimension)

    def search(
        self, query: torch.Tensor, k: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = faiss_sanitize(query)
        try:
            query = self.preprocessor.apply_py(query)
        except Exception as exc:
            logger.error(
                f"Failed to preprocess query ({type(query)}) "
                f"with preprocessor: {self.preprocessor}. "
                f"Exception: {exc}"
            )

        # logger.debug(f"Faiss:Query: {query.shape}, k: {k}")
        return self.index.search(query, k)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def _move_to_cuda_shard(
        self, index: faiss.Index, devices: List[int]
    ) -> faiss.IndexShards:
        return get_sharded_gpu_index(
            index,
            devices=devices,
            use_float16=self.config.use_float16,
            use_precomputed_tables=self.config.use_precomputed_tables,
            replicas=self.config.replicas,
            tempmem=self.config.tempmem,
        )

    @staticmethod
    def _mode_to_cuda(index: faiss.Index, devices: List[int]) -> faiss.IndexShards:
        return faiss.index_cpu_to_gpus_list(index, gpus=devices)

    def cuda(self, devices: Optional[List[int]] = None):

        if self.config.keep_on_cpu:
            return

        try:
            if isinstance(self.index, (faiss.GpuIndex)):
                return
        except Exception as e:
            logger.info(f"Couldn't check whether the index was a GPU index: {e}")

        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        # move the index to the GPU
        if self.config.shard:
            logger.info(f"Moving index to GPU shards {devices}")
            self.index = self._move_to_cuda_shard(self.index, devices)
        else:
            logger.info(f"Moving index to GPU replicas {devices} (no sharding)")
            self.index = self._mode_to_cuda(self.index, devices)

        # set the nprobe parameter
        ps = faiss.GpuParameterSpace()
        ps.initialize(self.index)
        ps.set_index_parameter(self.index, "nprobe", self.config.nprobe)

    def cpu(self):
        try:
            self.index = faiss.index_gpu_to_cpu(self.index)  # type: ignore
        except AttributeError:
            pass

        # set the nprobe parameter
        try:
            self.index.nprobe = self.config.nprobe
        except Exception as e:
            logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
            pass
