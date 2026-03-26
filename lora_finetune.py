import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Standard Imports
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass



#Config Imports
import config

#Defined Imports for data preparation
from sft_dataprep import (
        load_tokenizer_model,
        define_dataset,
        format_prompt,
        data_mapping,
        DataPrep
        )

#Defined Imports for Model Selection and Tokenization for sft 
from sft_model_tokenization import(
        model_select,
        tokenizer_select,
        SFTModelTokenization
        )

#define Imports for model definition with lora configuration
from sft_lora_config import(
        apply_lora
        )

#define the Imports for training arguments and SFTTrainer
from sft_trainer_config import (
        define_training_args,
        sft_trainer
        )

#define the Imports for Model Merges
from sft_merge_model_weights import(
        get_merged_model
        )

if __name__ == "__main__":
    #Step1: Prep the data file for sft. Use the "HuggingFace/ultrachat_200k" dataset for sft
    #Use the DataPrep() class to prep the data
    #Automatically prints the dataset["text"][0]
    data_prep = DataPrep()

    #dataset
    dataset_transformed = data_prep.data_mapping

    
    #Step2: Create the Model and Tokenizer
    sft_model_tokenization = SFTModelTokenization(model_name=config.MODEL_NAME)

    #Model:
    print("Model loading Started ...", flush=True)
    model_sft = sft_model_tokenization.model
    print("Model loading Finished...", flush=True)

    #Tokenizer:
    print("Tokenizer loading Started ...", flush=True)
    tokenizer_sft = sft_model_tokenization.tokenizer
    print("Tokenizer loading Finished ...", flush=True)

    #Check for distribution of Model layers across CPU/GPU
    print("Distribution of Model layers", flush=True)
    print(sft_model_tokenization.model.hf_device_map)

    #Step3: define the lora config 
    #model, peft_config = apply_lora(model_sft,alpha,dropout,r,bias,task_type,target_modules)
    print("Applying LoRA", flush=True)
    model, peft_config = apply_lora(model_sft,config.ALPHA,config.DROPOUT,config.R,config.BIAS,config.TASK_TYPE,config.TARGET_MODULES)
    model.enable_input_require_grads()

    #Print peft_config
    print(peft_config)

    #Check if LoRA is applied to trainable parameters
    print("Checking if LoRA is applied to trainable params",flush=True)
    model.print_trainable_parameters()

    #Check loRA layers injected
    for name, module in model.named_modules():
        if "lora" in name.lower():
            print(name)

    #Check Model class, Expected output: peft.peft_model.PeftModelForCausalLM
    print(type(model))

    print("Setting up Training Arguments",flush=True)
    #Setup the training arguments
    training_arguments = define_training_args(
            config.OUTPUT_DIR,
            config.PER_DEV_TRAIN_BATCH_SIZE,
            config.GRAD_ACC_STEP,
            config.OPTIM_METHOD,
            config.LR,
            config.LR_SCHED_TYPE,
            config.NUM_OF_EPOCHS,
            config.LOG_STEPS,
            config.FP16_BOOL_VALUE,
            config.GRAD_CHECKPOINT_BOOL_VALUE) 

    #Setup the SFTTrainer
    print("Starting Training",flush=True)
    sft_trainer_obj = sft_trainer(model,dataset_transformed,tokenizer_sft,training_arguments,config.MAX_SEQ_LEN,peft_config)

    #Train Model
    train_result = sft_trainer_obj.train()
    print(train_result, flush=True)

    #Save qLoRA weights
    #sft_trainer_obj.model.save_pretrained("TinyLlama-1.1B-qlora")
    #tokenizer_sft.save_pretrained("TinyLlama_1.1B_qlora")

    #Saving the adapter and tokeniser weights: Files stored are:
    #1. adapter_config.json
    #2. adapter_model.safetensors
    #3 tokenizer.json
    #4. tokenizer_config.json
    #5. special_token_map.json
    print("Saving Model and Tokenizer",flush=True)
    sft_trainer_obj.model.save_pretrained("TinyLlama-1.1B-qlora")
    tokenizer_sft.save_pretrained("TinyLlama-1.1B-qlora")

    print("Preparing the LoRA Model with merged weights", flush=True)
    #Merge the weights of adapters and base model
    merged_model_final = get_merged_model()

    print("Saving the merged model",flush=True)
    merged_model_final.save_pretrained("TinyLlama-1.1B-merged")

