from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def apply_lora (model,l_alpha, l_dropout, l_rank, l_bias, l_task_type, l_target_modules):
    print("Setting up lora config")
    peft_config=LoraConfig(
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            r=l_rank,
            bias=l_bias,
            task_type=l_task_type,
            target_modules=l_target_modules)

    #Prepare the model for training
    #model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model, peft_config
