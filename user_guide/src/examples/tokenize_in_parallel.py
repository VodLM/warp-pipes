import transformers
from warp_pipes.pipes import Parallel, Sequential, TokenizerPipe
from warp_pipes.pipes import HasPrefix

# example batch w. one object
batch = {
    "document.text": [
        """
        Fipple flutes are found in many cultures around the world.
        Often with six holes, the shepherd's pipe is a common pastoral image.
        Shepherds often piped both to soothe the sheep and to amuse themselves.
        Modern manufactured six-hole folk pipes are referred to as pennywhistle or 
        tin whistle. The recorder is a form of pipe, often used as a rudimentary 
        instructional musical instrument at schools, but versatile enough that 
        it is also used in orchestral music.
        """
    ],
    "title.text": ["Title: Pipe. "],
    "document.idx": [0],
}

# construct pipe which tokenizes the text and title attributes
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
tokenizer_pipe = Parallel(
    Sequential(
        TokenizerPipe(
            tokenizer, key="text", field="document", 
            return_offsets_mapping=True, add_special_tokens=False, update=True),
        input_filter=HasPrefix("document")
    ),
    Sequential(
        TokenizerPipe(
            tokenizer, key="text", field="title", 
            return_offsets_mapping=True, add_special_tokens=False, update=True),
        input_filter=HasPrefix("document")
    ),
    update=True
)

# execute pipe by sending through the document batch
tokenized_batch = tokenizer_pipe(batch)
print(tokenized_batch)