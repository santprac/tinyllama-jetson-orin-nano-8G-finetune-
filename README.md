# TinyLlama Fine-Tuning on Jetson Orin Nano 8GB

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/pytorch-2.8-red.svg)](https://pytorch.org/)
[![Tested on Jetson](https://img.shields.io/badge/Tested%20on-Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

Production-ready implementation for fine-tuning **TinyLlama-1.1B-Chat-v1.0** on NVIDIA Jetson Orin Nano 8GB using **LoRA** (Low-Rank Adaptation) and **FP16 precision**.

## 🎯 Key Achievements

- ✅ **22 minutes** training time for 3,000 samples (1 epoch)
- ✅ **563,200 trainable parameters** out of 1.1B (0.05%)
- ✅ **4.5GB GPU memory** usage during training
- ✅ **FP16 precision** (BitsAndBytes 4-bit quantization not supported on Jetson ARM64/CUDA 12.6)
- ✅ **Complete automation** with shell scripts
- ✅ **Modular architecture** for easy customization

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Training Time | 1331 seconds (~22.2 minutes) |
| Speed | 1.77 seconds/iteration |
| Throughput | 2.25 samples/second |
| Initial Loss | 1.8216 |
| Final Loss | 1.5203 |
| GPU Memory | ~4.5GB |
| Trainable % | 0.0512% |

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/santprac/tinyllama-jetson-orin-nano-8G-finetune-.git
cd tinyllama-jetson-orin-nano-8G-finetune-

# 2. Install dependencies
pip install -r requirements_working.txt

# 3. Set up Jetson performance
chmod +x setup_max_clocks.sh
./setup_max_clocks.sh

# 4. Close Cursor/VSCode to free GPU memory, then run training
python3 lora_finetune.py 2>&1 | tee training.log

# OR use fully automated training (recommended)
# Switch to TTY (Ctrl+Alt+F3), then:
chmod +x run_training_optimized.sh
./run_training_optimized.sh
```

## 📁 Project Structure

```
.
├── config.py                      # Centralized hyperparameters
├── lora_finetune.py               # Main training orchestration
├── sft_dataprep.py                # Data loading & formatting
├── sft_model_tokenization.py     # Model & tokenizer (FP16)
├── sft_lora_config.py             # LoRA configuration
├── sft_trainer_config.py          # Training arguments
├── sft_merge_model_weights.py    # Adapter merging
├── setup_max_clocks.sh            # Quick performance setup
├── run_training_optimized.sh     # Automated training (TTY mode)
├── requirements_working.txt       # Dependencies
├── fine_tuning_guide_v2.html     # Complete visual guide
└── README.md                      # This file
```

## 🔧 System Requirements

- **Hardware**: NVIDIA Jetson Orin Nano 8GB (or any GPU with 8GB+ VRAM)
- **OS**: Ubuntu 22.04 LTS (JetPack 6.2.1 for Jetson)
- **CUDA**: 12.6.68
- **Python**: 3.10.12
- **PyTorch**: 2.8.0 (pre-installed on Jetson)

## 📦 Key Dependencies

```
transformers==4.46.3        # NOT 4.47.x (TRL incompatibility)
peft==0.14.0
trl==0.12.2
accelerate==1.2.1
datasets==3.2.0
numpy<2.0                   # Critical: must be 1.x
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# LoRA Configuration
ALPHA = 8                          # LoRA scaling factor
R = 4                              # LoRA rank (lower = fewer params)
TARGET_MODULES = ["v_proj", "q_proj"]  # Attention layers only

# Training Configuration
PER_DEV_TRAIN_BATCH_SIZE = 1       # Physical batch size
GRAD_ACC_STEP = 4                  # Effective batch = 4
LR = 2e-4                          # Learning rate
MAX_SEQ_LEN = 64                   # Sequence length
```

## 🎓 Training Process

### Step-by-Step Pipeline

1. **Data Preparation** (`sft_dataprep.py`)
   - Loads UltraChat 200k dataset (3,000 samples)
   - Applies chat template with special tokens
   - Formats as text for SFTTrainer

2. **Model Loading** (`sft_model_tokenization.py`)
   - Loads TinyLlama in FP16 precision
   - **NO BitsAndBytes** (4-bit quantization not supported on Jetson CUDA 12.6 ARM64)
   - Auto GPU placement

3. **LoRA Application** (`sft_lora_config.py`)
   - Injects LoRA adapters into q_proj and v_proj
   - Reduces trainable params from 1.1B → 563k

4. **Training** (`sft_trainer_config.py`)
   - FP16 mixed precision
   - Gradient checkpointing (saves 40% memory)
   - Cosine learning rate schedule

5. **Save & Merge** (`sft_merge_model_weights.py`)
   - Saves LoRA adapters (2.2MB)
   - Merges with base model (2.2GB standalone)

## 🔥 Critical Jetson Optimizations

### 1. Close Cursor IDE Before Training
```bash
# Cursor uses 2.5-3.5GB GPU memory
# Close it before loading the model!
# Workflow: Develop → Save → Close → Train
```

### 2. Fix GPU Frequency (Critical!)
```bash
# GPU gets stuck at 612 MHz on JetPack 6.x
# Manually set to 1020 MHz for 15-20% speed boost
echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq
echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq
```

### 3. Use TTY Mode for Maximum Performance
```bash
# Stop GUI to free ~2GB GPU memory
# Press Ctrl+Alt+F3 to switch to TTY
./run_training_optimized.sh
# 10-step automated optimization
```

## 📈 Performance Comparison

| Scenario | GPU Memory | Speed | Training Time |
|----------|-----------|-------|---------------|
| Cursor + GUI + 612 MHz | ~4GB (FAIL) | - | Cannot run |
| GUI only + 612 MHz | ~5.5GB | 1.77 s/it | ~40-60 min |
| **setup_max_clocks.sh** | ~5.5GB | 1.77 s/it | **~22 min** |
| **run_training_optimized.sh** ⭐ | ~7GB | 1.5-1.6 s/it | **~18-20 min** |

## 🐛 Common Issues & Solutions

### Issue: Out of Memory
```bash
# Solution: Close Cursor IDE and/or switch to TTY mode
# Cursor uses 2.5-3.5GB, GNOME uses 1.5-2GB
```

### Issue: BitsAndBytes 4-bit Quantization Not Supported
```python
# DON'T use BitsAndBytes on Jetson (not supported on ARM64/CUDA 12.6)
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # ❌

# USE FP16 instead
torch_dtype=torch.float16  # ✅
```

### Issue: NumPy Version Mismatch
```bash
pip uninstall numpy
pip install "numpy<2.0"  # Force NumPy 1.x
```

### Issue: GPU Stuck at 612 MHz
```bash
# Check current frequency
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq

# Force to 1020 MHz
echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq
echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq
```

## 📚 Output Files

After training:

```
TinyLlama-1.1B-qlora/         # LoRA adapters (2.2MB)
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files...

TinyLlama-1.1B-merged/        # Merged model (2.2GB)
├── config.json
├── generation_config.json
└── model.safetensors

results/                       # Training checkpoints
├── checkpoint-500/
└── checkpoint-750/
```

## 🎨 Visual Guide

Open `fine_tuning_guide_v2.html` in your browser for a complete interactive guide with:
- Flow diagrams
- Code explanations
- Performance charts
- Troubleshooting tips

## 🔬 Extending This Work

### Increase Model Quality
```python
# In config.py:
R = 8                              # Higher LoRA rank
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN = 128                  # Longer sequences
NUM_OF_EPOCHS = 3                  # More epochs
```

### Use Your Own Dataset
```python
# In sft_dataprep.py:
dataset = load_dataset("json", data_files="your_data.json")
# Format: [{"messages": [{"role": "user", "content": "..."}, ...]}]
```

## 📖 Documentation

- [Complete Visual Guide](fine_tuning_guide_v2.html) - Interactive HTML guide
- [NVIDIA Forum Post](#) - Community discussion (coming soon)
- [LinkedIn Article](#) - High-level overview (coming soon)

## 🙏 Acknowledgments

- **Hugging Face** - Transformers, PEFT, TRL libraries
- **NVIDIA** - Jetson platform and CUDA support
- **TinyLlama** - Excellent small language model
- **UltraChat** - High-quality conversational dataset

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{tinyllama_jetson_finetune,
  author = {Santosh},
  title = {TinyLlama Fine-Tuning on Jetson Orin Nano 8GB},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/santprac/tinyllama-jetson-orin-nano-8G-finetune-}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## 💬 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/santprac/tinyllama-jetson-orin-nano-8G-finetune-/issues)
- LinkedIn: [Your Profile](#)

---

**⭐ If this project helped you, please star it on GitHub!**

Built with ❤️ for the Edge AI community
# tinyllama-jetson-orin-nano-8G-finetune-
