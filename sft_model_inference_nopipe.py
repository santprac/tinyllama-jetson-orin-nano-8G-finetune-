from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer,pipeline,AutoModelForCausalLM
import torch

import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass



#define the Imports for Model Merges
from sft_merge_model_weights import(
        get_merged_model
        )

TOKENIZER_MODEL_PATH = "./TinyLlama-1.1B-qlora"
FT_MODEL_PATH = "./TinyLlama-1.1B-merged"

#model = get_merged_model()

tokenizer_model = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
ft_model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
        )

#ft_model = ft_model.to("cuda:0")
ft_model.eval()

#Ready to ask questions based on predefined template
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

#Tokenize
inputs = tokenizer_model(prompt, return_tensors="pt")
#inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

#Generate
with torch.inference_mode():
    outputs = ft_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer_model.decode(outputs[0], skip_speacial_tokens=False))
