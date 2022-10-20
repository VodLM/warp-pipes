from collections import defaultdict
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch
from warp_pipes.support.functional import get_batch_eg


class GeneratePassages(Pipe):
    """
    Extract fixed-length passages from text documents.
    """

    _required_keys = [
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "text",
    ]
    _allows_update = False

    @property
    def required_keys(self) -> List[str]:
        return [
            f"{self.key_prefix}{key}" for key in self._required_keys
        ] + self.required_prepend_keys

    @property
    def required_prepend_keys(self) -> List[str]:
        if self.prepend_key_prefix is None:
            return []
        else:
            return [f"{self.prepend_key_prefix}{key}" for key in self._required_keys]

    def __init__(
        self,
        size: int,
        stride: int,
        *,
        field: Optional[str] = None,
        prepend_field: Optional[str] = None,
        start_tokens: List[int] = None,
        end_tokens: List[int] = None,
        pad_token_id: int = 0,
        verbose: bool = False,
        global_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            size (:obj:`int`): The size of the passages to extract.
            stride (:obj:`int`): The stride of the passages to extract.
            field (:obj:`str`, optional): The name of the field to extract passages from.
                If not provided, the passages will be extracted from the input_ids, else
                the passages will be extracted from `{field}.input_ids`.
            prepend_field (:obj:`str`, optional): The name of the field to prepend to
                the extracted passages. E.g., `title`.
            start_tokens (:obj:`List[int]`): The tokens to prepend to the extracted passages.
            end_tokens (:obj:`List[int]`): The tokens to append to the extracted passages.
            pad_token_id (:obj:`int`): The token id to use for padding.
            verbose (:obj:`bool`, optional): Whether to print debug information.
            global_keys (:obj:`List[str]`, optional): The keys to include with each passage.
                E.g., `document.idx`
        """
        super(GeneratePassages, self).__init__(**kwargs)
        if start_tokens is None:
            start_tokens = []
        if end_tokens is None:
            end_tokens = []

        # register the main field prefix
        if field is None:
            self.key_prefix = ""
        else:
            self.key_prefix = f"{field}."

        # register the prepend field prefix
        if prepend_field is None:
            self.prepend_key_prefix = None
        else:
            self.prepend_key_prefix = f"{prepend_field}."

        # setup the arguments for the `generate_passages_for_all_keys` function
        self.verbose = verbose
        self.global_keys = global_keys
        self.passage_args = self.get_passage_args(
            size=size,
            stride=stride,
            start_tokens=start_tokens,
            end_tokens=end_tokens,
            pad_token_id=pad_token_id,
        )

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        self._check_input_keys(batch)

        # generate the passages for each key
        indexes, output = self.generate_passages_for_all_keys(
            batch,
            keys=[
                f"{self.key_prefix}input_ids",
                f"{self.key_prefix}attention_mask",
                f"{self.key_prefix}offset_mapping",
            ],
            passage_args=self.passage_args,
            global_keys=self.global_keys,
            prepend_key_prefix=self.prepend_key_prefix,
        )

        # extract the text field
        extracted_texts = []
        for idx, ofs_ids in zip(indexes, output[f"{self.key_prefix}offset_mapping"]):
            passage = self.extract_passage_text_from_doc(
                batch[f"{self.key_prefix}text"][idx], ofs_ids
            )
            if self.prepend_key_prefix is not None:
                passage = batch[f"{self.prepend_key_prefix}text"][idx] + passage

            extracted_texts.append(passage)

        output[f"{self.key_prefix}text"] = extracted_texts

        return output

    def _check_input_keys(self, batch):
        for key in self.required_keys:
            assert key in batch.keys(), (
                f"key={key} must be provided. "
                f"Found batch.keys={list(batch.keys())}."
            )

    @staticmethod
    def generate_passages_for_all_keys(
        batch: Dict[str, List[Any]],
        keys: List[str],
        passage_args: Dict[str, Dict[str, Any]],
        global_keys: Optional[List[str]] = None,
        prepend_key_prefix: Optional[str] = None,
    ) -> Tuple[List[int], Batch]:
        """This functions generate the passages for each attribute in `keys`,
         the `arg` dictionary must contain an entry for all `keys`.
         The first pass is used to store the document/example indexes
        and compute the `passage_mask`.

        The passage mask is used for segmentation, and is optional for this project.
        In this context, all tokens are attributed to a single passage,
        although they appear in multiple passages (strides).
        The passage mask indicates if a token is attributed to this specific passage.

        Returns:
          - indexes: index of the parent example for each passage
          - output: Batch of data for all keys + `idx` (document id) and `passage_mask`

        """
        assert all(key in passage_args.keys() for key in keys)
        L = len(next(iter(batch.values())))
        assert all(L == len(x) for x in batch.values())

        first_key, *other_keys = keys
        output = defaultdict(list)
        indexes = []

        for idx, example in enumerate(batch[first_key]):
            args_egs = GeneratePassages.get_eg_args(
                get_batch_eg(batch, idx), passage_args, prepend_key_prefix
            )

            global_values = {}
            if global_keys is not None:
                global_values = {
                    k: batch[k][idx] for k in global_keys if k in batch.keys()
                }

            # do a first pass to compute the passage masks
            for pas_idx, (passage, passage_mask) in enumerate(
                gen_passages(example, **args_egs[first_key], return_mask=True)
            ):
                indexes += [idx]
                output["passage_idx"].append(pas_idx)
                output["passage_mask"].append(passage_mask)
                output[first_key].append(passage)

                # append the global values (doc idx, cui, etc...)
                for k, v in global_values.items():
                    output[k].append(v)

        # do another pass to generate the passages for each remaining attribute
        for key in other_keys:
            for idx, example in enumerate(batch[key]):
                args_egs = GeneratePassages.get_eg_args(
                    get_batch_eg(batch, idx), passage_args, prepend_key_prefix
                )
                passages = gen_passages(example, **args_egs[key], return_mask=False)
                for i, passage in enumerate(passages):
                    output[key].append(passage)

        # check output consistency and return
        L = len(list(next(iter(output.values()))))
        assert all(len(v) == L for v in output.values())
        return indexes, output

    def get_passage_args(self, size, stride, start_tokens, end_tokens, pad_token_id):
        """Define the arguments for the `gen_passages()` for each key."""
        base_args = {"size": size, "stride": stride}
        return {
            f"{self.key_prefix}input_ids": {
                "pad_token": pad_token_id,
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                **base_args,
            },
            f"{self.key_prefix}attention_mask": {
                "pad_token": 0,
                "start_tokens": [1 for _ in start_tokens],
                "end_tokens": [1 for _ in end_tokens],
                **base_args,
            },
            f"{self.key_prefix}offset_mapping": {
                "pad_token": [-1, -1],
                "start_tokens": [[-1, -1] for _ in start_tokens],
                "end_tokens": [[-1, -1] for _ in end_tokens],
                **base_args,
            },
        }

    @staticmethod
    def get_eg_args(eg: Dict[str, Any], passage_args: Dict, prepend_key_prefix: bool):
        # retrieve the prefix f"{field}."
        eg_prefix = ".".join(next(iter(passage_args.keys())).split(".")[:-1])
        if len(eg_prefix):
            eg_prefix += "."

        # update the `passage_args` with the example values
        eg_args = deepcopy(passage_args)
        if prepend_key_prefix is not None:
            # retrieve the tokens of the auxiliary field
            aux_input_key = f"{prepend_key_prefix}input_ids"
            if aux_input_key not in eg:
                raise ValueError(
                    f"Key `{aux_input_key}` not found in batch "
                    "(prepend_key_prefix=`{prepend_key_prefix}`)."
                )
            aux_input_ids = eg[aux_input_key]

            # update the example args
            eg_args[f"{eg_prefix}input_ids"]["start_tokens"] += aux_input_ids
            eg_args[f"{eg_prefix}attention_mask"]["start_tokens"] += [
                1 for _ in aux_input_ids
            ]
            eg_args[f"{eg_prefix}offset_mapping"]["start_tokens"] += [
                [-1, -1] for _ in aux_input_ids
            ]

        return eg_args

    @staticmethod
    def extract_passage_text_from_doc(
        document: str, offset_mapping: List[Tuple[int, int]]
    ) -> str:
        indexes = [x for idxes_tok in offset_mapping for x in idxes_tok if x >= 0]
        return document[min(indexes) : max(indexes)]

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(
            size=10,
            stride=10,
            start_tokens=[0],
            end_tokens=[0],
            pad_token_id=0,
            **kwargs,
        )


def gen_passages(
    sequence: List[Any],
    *,
    size: int,
    stride: int,
    start_tokens: Optional[List[Any]] = None,
    end_tokens: Optional[List[Any]] = None,
    pad_token: Optional[Any] = None,
    return_mask: bool = True,
) -> Iterable[Union[List[int], Tuple[List[int], List[Any]]]]:
    """Generate overlapping windows with the corresponding
    masking such that each token appears only in one window.

    Args:
      sequence (:obj:`List[Any]`): The sequence to be split into passages.
      *:
      size (:obj:`int`): The size of the passages.
      stride (:obj:`int`): The stride of the passages.
      title_tokens (:obj:`Optional[List[Any]]`, `optional`): The tokens to be added at
        the beginning of each passage.
      start_tokens (:obj:`Optional[List[Any]]`, `optional`): The tokens to be added at
        the beginning of each passage.
      end_tokens (:obj:`Optional[List[Any]]`, `optional`): The tokens to be added at
        the end of each passage.
      pad_token (:obj:`Optional[Any]`, `optional`): The token to be used for padding.
      return_mask (:obj:`bool`, `optional`): Whether to return the mask or not.

    Returns:

    """

    if start_tokens is not None:
        eff_size = size - len(start_tokens)
        eff_stride = stride - len(start_tokens)
    else:
        start_tokens = []
        eff_size = size
        eff_stride = stride

    if end_tokens is not None:
        eff_size -= len(end_tokens)
        eff_stride -= len(end_tokens)
    else:
        end_tokens = []

    assert eff_size > 0
    assert eff_stride > 0
    assert eff_stride <= eff_size
    margin = eff_size - eff_stride
    for i in range(0, len(sequence), eff_stride):
        left_pad = margin // 2 + margin % 2 if i else 0
        right_pad = margin // 2
        center = eff_size - left_pad - right_pad
        seq = sequence[i : i + eff_size]
        padding = max(0, eff_size - len(seq)) if pad_token is not None else 0

        # only return if there are unmasked tokens
        if len(seq) > left_pad:

            # define the passage
            seq = start_tokens + seq + end_tokens + padding * [pad_token]
            # define the passage mask
            mask = (
                (len(start_tokens) + left_pad) * [0]
                + center * [1]
                + [0] * (len(end_tokens) + right_pad)
            )
            if padding > 0:
                mask[-padding:] = padding * [0]

            if return_mask:
                yield (
                    seq,
                    mask[: len(seq)],
                )
            else:
                yield seq
