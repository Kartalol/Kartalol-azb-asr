from datasets import load_dataset
# Load back
from datasets import load_from_disk
dataset_dict = load_from_disk("ASR-AZB-Gold-TestSet")
print(dataset_dict)
