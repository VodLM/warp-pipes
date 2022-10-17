from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from transformers import PreTrainedTokenizerFast

from warp_pipes.core.condition import In
from warp_pipes.core.pipe import Pipe
from warp_pipes.support.datastruct import Batch


class TokenizerPipe(Pipe):
    """tokenize a batch of data"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        drop_columns: bool = True,
        fields: Union[str, List[str]] = "text",
        max_length: Optional[int] = None,
        return_token_type_ids: bool = False,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        self.fields = [fields] if isinstance(fields, str) else fields
        assert kwargs.get("input_filter", None) is None, "input_filter is not allowed"
        super(TokenizerPipe, self).__init__(**kwargs, input_filter=In(self.fields))
        self.tokenizer = tokenizer
        self.drop_columns = drop_columns
        self.args = {
            "max_length": max_length,
            "truncation": max_length is not None,
            "return_token_type_ids": return_token_type_ids,
            "return_offsets_mapping": return_offsets_mapping,
            "add_special_tokens": add_special_tokens,
        }

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> Batch:
        tokenizer_input = {field: batch[field] for field in self.fields}

        batch_encoding = self.tokenizer(
            *tokenizer_input.values(), **self.args, **kwargs
        )

        if self.drop_columns:
            batch = {k: v for k, v in batch_encoding.items()}
        else:
            batch.update({k: v for k, v in batch_encoding.items()})

        return batch

    @classmethod
    def instantiate_test(cls, **kwargs) -> "Pipe":
        return cls(PreTrainedTokenizerFast.from_pretrained("bert-base-cased"), **kwargs)
