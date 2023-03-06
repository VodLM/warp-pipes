from warp_pipes.pipes.passages import GeneratePassages

tokenizer =  ...
tokenized_batch = ...

# generate final passages by setting the size and stride for a preprocessed batch
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
print(passages)