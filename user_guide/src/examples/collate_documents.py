from warp_pipes.pipes import CollateField
from warp_pipes.support.functional import get_batch_eg

# some result of GeneratePassages pipe
passages = ...

# extract the document field, run pipe transformation into tensors
collate_docs = CollateField(
    field="document", 
    to_tensor=["idx", "input_ids", "attention_mask"]
)

# fetch only documents from passages, perform the collate process
egs = [get_batch_eg(passages, idx=i) for i  in [0, 1, 2, 3]]
c_batch = collate_docs(egs, to_tensor=[])
print(c_batch, "output batch")