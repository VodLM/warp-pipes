from datasets import load_dataset
from warp_pipes import Pipe, Sequential, AddPrefix, GetKey, Lambda, Identity, FilterKeys, DropKeys, ApplyToAll, Apply

# load datasets from huggingface
dataset = load_dataset("glue", "mrpc", split="train")

print(len(dataset))
print(dataset[0:3])


# Let's create the following preprocessing pipeline:
# add prefix to sentence1
# remove sentence2


# done with huggingface
updated_dataset = dataset.map(lambda x: {"sentence":  x["sentence2"] + x["sentence1"]}, remove_columns=["sentence2"])
#print(updated_dataset.column_names)
print(updated_dataset[0:3])



#
#dataset = load_dataset("glue", "mrpc", split="train")
pipeline = Sequential(Apply(ops = {"sentence1": lambda x: "pre_fix"+x}, element_wise=True),
                      DropKeys(["sentence2", "idx"]))


dataset = pipeline(dataset)

print()
print(dataset[0:3])
#dataset = pipeline(dataset,)

