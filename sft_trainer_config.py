from trl import SFTTrainer
from transformers import TrainingArguments

#Set up the training argeuments:
def define_training_args(output_dir,per_dev_train_batch_size,grad_acc_step,optim_method,lr,lr_sched_type,num_of_epochs,log_steps,fp16_bool_value,grad_checkpoint_bool_value):
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_dev_train_batch_size,
        gradient_accumulation_steps=grad_acc_step,
        optim=optim_method,
        learning_rate=lr,
        lr_scheduler_type=lr_sched_type,
        num_train_epochs=num_of_epochs,
        logging_steps=log_steps,
        fp16=fp16_bool_value,
        gradient_checkpointing=grad_checkpoint_bool_value
    )

    return training_arguments


#Set supervised fine tuning parameters
def sft_trainer(model,dataset,tokenizer,training_args,max_seq_len,peft_config):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_len,
        )
    return trainer
