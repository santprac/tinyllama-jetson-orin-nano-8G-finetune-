from  peft import AutoPeftModelForCausalLM

def get_merged_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
            "TinyLlama-1.1B-qlora",
             low_cpu_mem_usage=True,
             device_map="cpu"
            )

    merged_model  = model.merge_and_unload()
    return merged_model

