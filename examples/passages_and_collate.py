import transformers
from warp_pipes.pipes import (
    TokenizerPipe,
    Sequential,
    Parallel,
    HasPrefix, GeneratePassages, CollateField,
)
from warp_pipes.support.functional import get_batch_eg
from warp_pipes.support.pretty import pprint_batch

bert_id = "bert-base-cased"
batch = {
    "document.text": [
        "Fipple flutes are found in many cultures around the world. Often with six holes, the shepherd's pipe is a common pastoral image. Shepherds often piped both to soothe the sheep and to amuse themselves. Modern manufactured six-hole folk pipes are referred to as pennywhistle or tin whistle. The recorder is a form of pipe, often used as a rudimentary instructional musical instrument at schools, but versatile enough that it is also used in orchestral music."
    ],
    "title.text": ["Title: Pipe. "],
    "document.idx": [0],
}


def run():
    # build a pipe to tokenize the text and the title
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_id)
    tokenizer_pipe = Parallel(
        Sequential(
            TokenizerPipe(
                tokenizer,
                key="text",
                field="document",
                return_offsets_mapping=True,
                add_special_tokens=False,
                update=True,
            ),
            input_filter=HasPrefix("document"),
        ),
        Sequential(
            TokenizerPipe(
                tokenizer,
                key="text",
                field="title",
                return_offsets_mapping=True,
                add_special_tokens=False,
                update=True,
            ),
            input_filter=HasPrefix("title"),
        ),
        update=True,
    )

    # tokenize the batch
    pprint_batch(batch, header="Input batch")
    tokenized_batch = tokenizer_pipe(batch)
    pprint_batch(tokenized_batch, header="Tokenized batch")

    # passages
    passages_pipe = GeneratePassages(
        size=30,
        stride=20,
        field="document",
        global_keys=["idx"],
        start_tokens=[tokenizer.cls_token_id],
        end_tokens=[tokenizer.sep_token_id],
        prepend_field="title",
    )
    passages = passages_pipe(tokenized_batch, header="Passages")
    pprint_batch(passages)

    # collate examples
    collate_docs = CollateField(field="document", to_tensor=["idx", "input_ids", "attention_mask"])
    egs = [get_batch_eg(passages, idx=i) for i  in [0, 1, 2, 3]]
    c_batch = collate_docs(egs,
                           to_tensor=[])
    pprint_batch(c_batch, "output batch")


if __name__ == "__main__":
    run()
