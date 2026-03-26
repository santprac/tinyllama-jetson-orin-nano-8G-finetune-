import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_select(model_name):
    print(50 * "==")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"":0},
        low_cpu_mem_usage=True
    )


    print(model.hf_device_map)

    model.config.use_cache = False
    return model

def tokenizer_select(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

class SFTModelTokenization:
    def __init__(self, model_name):
        print("Setting up Model for SFT")
        self.model = model_select(model_name)
        print("Setting up Tokenizer for SFT")
        self.tokenizer = tokenizer_select(model_name)

if __name__ == "__main__":
    sft_model_tokenization = SFTModelTokenization()
    print("Model loaded")
    print(sft_model_tokenization.model.hf_device_map)
    print("Tokenizer loaded")
