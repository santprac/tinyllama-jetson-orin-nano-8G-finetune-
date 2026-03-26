MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_MEMORY = {
    0: "2.25GiB",
    "cpu": "16GiB"
}

ALPHA = 8
DROPOUT = 0.0
R = 4
BIAS = "none"
TASK_TYPE = "CAUSAL_LM"
#TARGET_MODULES = ["k_proj","gate_proj","v_proj","up_proj","q_proj","o_proj","down_proj"] 
TARGET_MODULES = ["v_proj","q_proj"] 

#defne the params for training arguments
OUTPUT_DIR = "./results"

PER_DEV_TRAIN_BATCH_SIZE = 1
GRAD_ACC_STEP = 4
OPTIM_METHOD = "adamw_torch"
LR = 2e-4
LR_SCHED_TYPE = "cosine"
NUM_OF_EPOCHS = 1
LOG_STEPS = 10
FP16_BOOL_VALUE = True  #For GPU ONLY
GRAD_CHECKPOINT_BOOL_VALUE = True
NO_CUDA = False


#define the params for SFT Trainer
MAX_SEQ_LEN = 64

