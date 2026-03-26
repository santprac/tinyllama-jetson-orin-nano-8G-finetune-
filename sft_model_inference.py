from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer,pipeline,AutoModelForCausalLM
import torch


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
        torch_dtype=torch.float16,
        #device_map="cpu"
        low_cpu_mem_usage=True,
        )

ft_model = ft_model.to("cuda:0")

#Run the Instruction Tuned Model
pipe = pipeline(
        task="text-generation",
        model=ft_model,
        device=0,
        tokenizer=tokenizer_model,
        torch_dtype=torch.float16
        )

#Ready to ask questions based on predefined template
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

output=pipe(prompt, max_new_tokens=250)

print(output[0]["generated_text"])

