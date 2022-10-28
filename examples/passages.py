import transformers

from warp_pipes.pipes import (
    GeneratePassages,
    TokenizerPipe,
    AddPrefix,
    Sequential,
    Parallel,
    HasPrefix,
    RenameKeys,
)
from warp_pipes.support.pretty import pprint_batch

bert_id = "bert-base-cased"
batch = {
    "document": [
        "Fipple flutes are found in many cultures around the world. Often with six holes, the shepherd's pipe is a common pastoral image. Shepherds often piped both to soothe the sheep and to amuse themselves. Modern manufactured six-hole folk pipes are referred to as pennywhistle or tin whistle. The recorder is a form of pipe, often used as a rudimentary instructional musical instrument at schools, but versatile enough that it is also used in orchestral music."
    ],
    "title": ["Title: Pipe. "],
    "idx": [0],
}


def run():
    # build a pipe to tokenize the text and the title
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_id)
    tokenizer_pipe = Parallel(
        Sequential(
            TokenizerPipe(
                tokenizer,
                field="document",
                return_offsets_mapping=True,
                add_special_tokens=False,
                update=True,
            ),
            AddPrefix("document."),
            RenameKeys({"document.document": "document.text"}),
            input_filter=HasPrefix("document"),
        ),
        Sequential(
            TokenizerPipe(
                tokenizer,
                field="title",
                return_offsets_mapping=True,
                add_special_tokens=False,
                update=True,
            ),
            AddPrefix("title."),
            RenameKeys({"title.title": "title.text"}),
            input_filter=HasPrefix("title"),
        ),
        update=True,
    )

    # tokenize the batch
    pprint_batch(batch, header="Input batch")
    tokenized_batch = tokenizer_pipe(batch)
    pprint_batch(tokenized_batch, header="Tokenized batch")

    # build a pipe to generate passages
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

    # display the output batch
    for i, p in enumerate(passages["document.input_ids"]):
        print(f"{i}", tokenizer.decode(p, skip_special_tokens=False), end="\n\n")


if __name__ == "__main__":
    run()
