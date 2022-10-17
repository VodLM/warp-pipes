from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch


class ExtractPassages(Pipe):
    """Extract fixed-length passages from text documents."""

    required_keys = [
        "document.input_ids",
        "document.attention_mask",
        "document.offset_mapping",
    ]
    _allows_update = False

    def __init__(
        self,
        *,
        size: int,
        stride: int,
        append_document_titles: bool = False,
        start_tokens: List[int],
        end_tokens: List[int],
        pad_token_id: int,
        verbose: bool = False,
        global_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        size
            The size of the passage to extract.
        stride
            The stride of the passage to extract.
        append_document_titles
            Whether or not append document titles to each passage
        start_tokens
            The tokens to append to the start of each passage
        end_tokens
            The tokens to append to the end of each passage
        pad_token_id
            The token to use for padding
        verbose
            Verbosity level
        global_keys
            The document keys to pass to each passage.
        kwargs
            Other arguments to pass to the base Pipe class.
        """
        super(ExtractPassages, self).__init__(**kwargs)
        self.append_document_titles = append_document_titles
        self.verbose = verbose
        self.global_keys = global_keys
        base_args = {"size": size, "stride": stride}
        self.args = {
            "document.input_ids": {
                "pad_token": pad_token_id,
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                **base_args,
            },
            "document.attention_mask": {
                "pad_token": 0,
                "start_tokens": [1 for _ in start_tokens],
                "end_tokens": [1 for _ in end_tokens],
                **base_args,
            },
            "document.offset_mapping": {
                "pad_token": [-1, -1],
                "start_tokens": [[-1, -1] for _ in start_tokens],
                "end_tokens": [[-1, -1] for _ in end_tokens],
                **base_args,
            },
        }

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """

        Args:
          batch: Batch:
          **kwargs:

        Returns:


        """
        self._check_input_keys(batch)
        indexes, output = self.generate_passages_for_all_keys(
            batch,
            keys=[
                "document.input_ids",
                "document.attention_mask",
                "document.offset_mapping",
            ],
            args=self.args,
            global_keys=self.global_keys,
            append_document_titles=self.append_document_titles,
        )

        # extract document.text
        output["document.text"] = [
            self.extract_passage_text_from_doc(batch["document.text"][idx], ofs_ids)
            for idx, ofs_ids in zip(indexes, output["document.offset_mapping"])
        ]

        return output

    def _check_input_keys(self, batch):
        """

        Args:
          batch:

        Returns:


        """
        for key in self.required_keys:
            assert key in batch.keys(), (
                f"key={key} must be provided. "
                f"Found batch.keys={list(batch.keys())}."
            )

    def generate_passages_for_all_keys(
        self,
        examples: Dict[str, List[Any]],
        keys: List[str],
        args: Dict[str, Dict[str, Any]],
        global_keys: Optional[List[str]] = None,
        append_document_titles: bool = False,
    ) -> Tuple[List[int], Batch]:
        """This functions generate the passages for each attribute in `keys`,
         the `arg` dictionary must contain an entry for all `keys`.
         The first pass is used to store the document/example indexes
        and compute the `passage_mask`.

        The passage mask is used for segmentation, and is optional for this project.
        In this context, all tokens are attributed to a single passage,
        although they appear in multiple passages (strides).
        The passage mask indicates if a token is attributed to this specific passage.

        Args:
          examples: Dict[str:
          List[Any]]:
          keys: List[str]:
          args: Dict[str:
          Dict[str:
          Any]]:
          global_keys: Optional[List[str]]:  (Default value = None)
          append_document_titles: bool:  (Default value = False)

        Returns:
          - indexes: index of the parent example for each passage
          - output: Batch of data for all keys + `idx` (document id) and `passage_mask`

        """
        assert "document.idx" in examples.keys()
        assert all(key in args.keys() for key in keys)
        L = len(next(iter(examples.values())))
        assert all(L == len(x) for x in examples.values())

        first_key, *other_keys = keys
        output = defaultdict(list)
        indexes = []

        for idx, example in enumerate(examples[first_key]):
            if append_document_titles:
                input_key = first_key.split(".")[-1]
                args = get_title_tokens(
                    args=args,
                    arg_key=first_key,
                    temp_input=examples[f"title.{input_key}"][idx],
                )

            global_values = {}
            if global_keys is not None:
                global_values = {
                    k: examples[k][idx] for k in global_keys if k in examples.keys()
                }

            # do a first pass to compute the passage masks
            for pas_idx, (passage, passage_mask) in enumerate(
                gen_passages(example, **args[first_key], return_mask=True)
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
            for idx, example in enumerate(examples[key]):
                if append_document_titles:
                    input_key = key.split(".")[-1]
                    args = get_title_tokens(
                        args=args,
                        arg_key=key,
                        temp_input=examples[f"title.{input_key}"][idx],
                    )
                passages = gen_passages(example, **args[key], return_mask=False)
                for i, passage in enumerate(passages):
                    output[key].append(passage)

        # check output consistency and return
        L = len(list(next(iter(output.values()))))
        assert all(len(v) == L for v in output.values())
        return indexes, output

    @staticmethod
    def extract_passage_text_from_doc(
        document: str, offset_mapping: List[Tuple[int, int]]
    ) -> str:
        """Extract the text passage from the original document
        given the offset mapping of the passage

        Args:
          document: str:
          offset_mapping: List[Tuple[int:
          int]]:

        Returns:

        """
        indexes = [x for idxes_tok in offset_mapping for x in idxes_tok if x >= 0]
        return document[min(indexes) : max(indexes)]


def gen_passages(
    sequence: List[Any],
    *,
    size: int,
    stride: int,
    title_tokens: Optional[List[Any]] = None,
    start_tokens: Optional[List[Any]] = None,
    end_tokens: Optional[List[Any]] = None,
    pad_token: Optional[Any] = None,
    return_mask: bool = True,
) -> Iterable[Union[List[int], Tuple[List[int], List[Any]]]]:
    """Generate overlapping windows with the corresponding
    masking such that each token appears only in one window.

    Args:
      sequence: List[Any]:
      *:
      size: int:
      stride: int:
      title_tokens: Optional[List[Any]]:  (Default value = None)
      start_tokens: Optional[List[Any]]:  (Default value = None)
      end_tokens: Optional[List[Any]]:  (Default value = None)
      pad_token: Optional[Any]:  (Default value = None)
      return_mask: bool:  (Default value = True)

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

    if title_tokens is not None:
        eff_size -= len(title_tokens)
        eff_stride -= len(title_tokens)
    else:
        title_tokens = []

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
            seq = start_tokens + title_tokens + seq + end_tokens + padding * [pad_token]
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


def get_title_tokens(
    args: dict,
    arg_key: str,
    temp_input: List[int],
) -> dict:
    """

    Args:
      args: dict:
      arg_key: str:
      temp_input: List[int]:

    Returns:


    """
    if "input_ids" in arg_key:
        args[arg_key]["title_tokens"] = temp_input
    elif "attention_mask" in arg_key:
        args[arg_key]["title_tokens"] = [1 for _ in temp_input]
    elif "offset_mapping" in arg_key:
        args[arg_key]["title_tokens"] = [[-1, -1] for _ in temp_input]
    else:
        NotImplementedError
    return args
