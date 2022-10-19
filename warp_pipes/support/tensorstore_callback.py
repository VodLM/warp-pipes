from __future__ import annotations

from typing import List
from typing import Optional

import pytorch_lightning as pl
import tensorstore as ts
import torch
from pytorch_lightning import Callback

from warp_pipes.support.datastruct import Batch

IDX_COL = "__idx__"


def write_vectors(
    store: ts.TensorStore,
    vectors: torch.Tensor,
    idx: List[int],
    asynch: bool = False,
) -> None:
    """write a table to file."""
    if idx is None:
        raise ValueError("idx must be provided")

    if asynch:
        store[idx].write(vectors)
    else:
        store[idx] = vectors


def select_key_from_output(batch: Batch, key: Optional[str] = None) -> torch.Tensor:
    if key is None:
        if isinstance(batch, dict):
            raise TypeError(
                "Input batch is a ditionary, the argument `field`must be set."
            )
        return batch

    return batch[key]


class TensorStoreCallback(Callback):
    """Allows storing the output of each `prediction_step` into a `ts.TensorStore`"""

    def __init__(
        self,
        store: ts.TensorStore,
        output_key: Optional[str] = None,
        asynch: bool = False,
    ):
        self.store = store
        self.output_key = output_key
        self.asynch = asynch

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Batch,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """store the outputs of the prediction step to the cache"""
        vectors = select_key_from_output(outputs, self.output_key)
        write_vectors(self.store, vectors=vectors, idx=batch.get(IDX_COL, None))
