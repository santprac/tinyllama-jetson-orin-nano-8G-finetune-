from datasets import load_dataset
from transformers import AutoTokenizer

def load_tokenizer_model(model_name):
    """Load the tokenizer from the model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer

def define_dataset(dataset_name, batch_size=1, shuffle=True):
    """Create a dataloader from a dataset"""
    dataset = (
        load_dataset(dataset_name, split="test_sft")
        .shuffle(seed=42)
        .select(range(3_000))
    )
    return dataset

def format_prompt(example, template_tokenizer):
    """Format the prompt of the dataset"""
    chat = example["messages"]
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

def data_mapping(dataset, template_tokenizer):
    """Map the dataset to the format required for the model"""
    dataset = dataset.map(
        lambda example: format_prompt(example, template_tokenizer), 
        batched=False,
        remove_columns=dataset.column_names
        )
    print(dataset["text"][0])
    return dataset

class DataPrep:
    """Class to prepare the data for the model"""
    def __init__(self):
        print("Starting Data Preparation")
        self.tokenizer = load_tokenizer_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.dataset = define_dataset("HuggingFaceH4/ultrachat_200k", batch_size=1, shuffle=True)
        self.data_mapping = data_mapping(self.dataset, self.tokenizer)
        self.data_mapping.save_to_disk("data_mapping")

if __name__ == "__main__":
    data_prep = DataPrep()

