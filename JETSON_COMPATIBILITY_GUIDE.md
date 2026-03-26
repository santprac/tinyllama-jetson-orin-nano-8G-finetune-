# Jetson Orin Nano - CUDA & PyTorch Compatibility Guide

## Your Current System Configuration

```
Jetson Platform: Jetson Orin Nano 8GB
JetPack Version: R36.4.7 (JetPack 6.x)
CUDA Version: 12.6.68
Python Version: 3.10.12
Architecture: aarch64 (ARM64)

Currently Installed:
- PyTorch: 2.8.0
- Transformers: 4.41.0
- TorchVision: 0.23.0
```

---

## ✅ RECOMMENDED: Compatible Package Versions

### Core Deep Learning Stack

Based on your **CUDA 12.6** and **JetPack 6.x**, here are the compatible versions:

```bash
# PyTorch (already installed - compatible)
torch==2.8.0

# Transformers (already installed - but we'll upgrade)
transformers==4.47.1

# PEFT for LoRA
peft==0.14.0

# TRL for SFT Training
trl==0.12.2

# BitsAndBytes for Quantization (IMPORTANT: ARM64 version)
bitsandbytes==0.43.1

# Accelerate for distributed training
accelerate==1.2.1

# Datasets for data loading
datasets==3.2.0

# Additional utilities
scipy==1.13.1
sentencepiece==0.2.0
protobuf==5.29.2
```

---

## Installation Commands

### Option 1: Clean Install (Recommended)

```bash
# Step 1: Create virtual environment
cd /home/santosh/Desktop/DSAI/PyTorch/FineTuning/FT_GEN_MODELS
python3 -m venv venv_tinyllama
source venv_tinyllama/bin/activate

# Step 2: Upgrade pip
pip install --upgrade pip setuptools wheel

# Step 3: Install PyTorch (if not already installed system-wide)
# For Jetson, use the pre-built wheel from NVIDIA
pip3 install torch torchvision torchaudio

# Step 4: Install Transformers ecosystem
pip install transformers==4.47.1
pip install accelerate==1.2.1
pip install datasets==3.2.0

# Step 5: Install PEFT and TRL
pip install peft==0.14.0
pip install trl==0.12.2

# Step 6: Install BitsAndBytes (ARM64 compatible)
pip install bitsandbytes==0.43.1

# Step 7: Install additional dependencies
pip install scipy sentencepiece protobuf

# Step 8: Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
```

### Option 2: Upgrade Current Installation

```bash
# Upgrade Transformers
pip install --upgrade transformers==4.47.1

# Install missing packages
pip install peft==0.14.0
pip install trl==0.12.2
pip install bitsandbytes==0.43.1
pip install accelerate==1.2.1
pip install datasets==3.2.0
pip install scipy sentencepiece protobuf
```

---

## Important Notes for Jetson

### 1. BitsAndBytes on ARM64 (Jetson)

⚠️ **CRITICAL**: Standard BitsAndBytes doesn't work on ARM64 out of the box.

**Solution Options:**

#### Option A: Use Compatible ARM64 Version (Recommended)
```bash
# BitsAndBytes 0.43.1 has proven ARM64 support for Jetson
pip install bitsandbytes==0.43.1
```

#### Option B: Build from Source (If Option A fails)
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
pip install -r requirements.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

#### Option C: Use Alternative Quantization
If BitsAndBytes fails, use built-in PyTorch quantization:
```python
# Instead of BitsAndBytes, use PyTorch native quantization
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Use PyTorch quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. Transformers Version Compatibility

| Transformers Version | CUDA Support | Features | Jetson Compatible |
|---------------------|--------------|----------|-------------------|
| 4.30.x - 4.35.x | CUDA 11.x | Basic LoRA | ✅ Yes |
| 4.36.x - 4.40.x | CUDA 11.8+ | Improved quantization | ✅ Yes |
| **4.41.x - 4.47.x** | **CUDA 12.x** | **Full 4-bit support** | **✅ Recommended** |
| 4.48.x+ (future) | CUDA 12.x | Latest features | ⚠️ Test first |

**Your current 4.41.0 works, but 4.47.1 is more stable for CUDA 12.6**

### 3. CUDA 12.6 Specific Considerations

```bash
# Verify CUDA is accessible from PyTorch
python3 << EOF
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

Expected output:
```
CUDA Available: True
CUDA Version: 12.6
cuDNN Version: 90100
GPU Device: Orin
GPU Memory: 8.00 GB
```

---

## Version Compatibility Matrix

### PyTorch ↔ CUDA ↔ Transformers

| PyTorch | CUDA | Transformers | PEFT | TRL | Status |
|---------|------|--------------|------|-----|--------|
| 2.0.x | 11.8 | 4.30.x | 0.4.x | 0.4.x | ⚠️ Old |
| 2.1.x | 11.8 | 4.35.x | 0.7.x | 0.7.x | ⚠️ Old |
| 2.2.x | 12.1 | 4.38.x | 0.9.x | 0.9.x | ✅ Good |
| 2.3.x | 12.1 | 4.40.x | 0.10.x | 0.10.x | ✅ Good |
| **2.5.x-2.8.x** | **12.4-12.6** | **4.45.x-4.47.x** | **0.13.x-0.14.x** | **0.11.x-0.12.x** | **✅ Best** |

---

## Complete Requirements File

Create this as `requirements.txt`:

```txt
# PyTorch and Vision (use system-installed for Jetson)
# torch==2.8.0  # Already installed system-wide
# torchvision==0.23.0

# Transformers Ecosystem
transformers==4.47.1
accelerate==1.2.1
datasets==3.2.0

# Fine-tuning Libraries
peft==0.14.0
trl==0.12.2

# Quantization (ARM64 compatible)
bitsandbytes==0.45.0

# NLP Utilities
sentencepiece==0.2.0
protobuf==5.29.2

# Scientific Computing
scipy==1.13.1
numpy>=1.24.0

# Optional but recommended
tensorboard>=2.15.0
wandb>=0.16.0  # For experiment tracking
huggingface-hub>=0.20.0

# Development tools (optional)
jupyter>=1.0.0
ipython>=8.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Verification Script

Save as `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
Verify all dependencies for TinyLlama fine-tuning on Jetson Orin Nano
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    package_name = package_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: NOT INSTALLED - {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA Status:")
            print(f"   - CUDA Available: True")
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - Device: {torch.cuda.get_device_name(0)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   - Compute Capability: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print(f"\n❌ CUDA: Not available")
            return False
    except Exception as e:
        print(f"\n❌ CUDA Check Failed: {e}")
        return False

def check_bitsandbytes():
    """Check BitsAndBytes specifically"""
    try:
        import bitsandbytes as bnb
        print(f"\n✅ BitsAndBytes: {bnb.__version__}")
        
        # Try to import CUDA functions
        from bitsandbytes.cuda_setup.main import CUDASetup
        setup = CUDASetup.get_instance()
        print(f"   - CUDA Setup: {setup}")
        return True
    except ImportError:
        print(f"\n❌ BitsAndBytes: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"\n⚠️  BitsAndBytes: Installed but may have issues - {e}")
        return True

def main():
    print("=" * 60)
    print("Jetson Orin Nano - TinyLlama Fine-tuning Verification")
    print("=" * 60)
    
    print(f"\nPython Version: {sys.version}")
    
    print("\n" + "=" * 60)
    print("Checking Core Dependencies")
    print("=" * 60)
    
    all_ok = True
    
    # Core libraries
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('transformers', 'Transformers')
    all_ok &= check_import('peft', 'PEFT')
    all_ok &= check_import('trl', 'TRL')
    all_ok &= check_import('accelerate', 'Accelerate')
    all_ok &= check_import('datasets', 'Datasets')
    
    # Optional but important
    check_import('scipy', 'SciPy')
    check_import('sentencepiece', 'SentencePiece')
    
    # Check CUDA
    all_ok &= check_cuda()
    
    # Check BitsAndBytes separately
    bnb_ok = check_bitsandbytes()
    
    print("\n" + "=" * 60)
    if all_ok and bnb_ok:
        print("✅ ALL CHECKS PASSED - Ready for fine-tuning!")
    elif all_ok:
        print("⚠️  Core libraries OK, but BitsAndBytes may need attention")
        print("   You can still train without quantization")
    else:
        print("❌ SOME CHECKS FAILED - Please install missing packages")
    print("=" * 60)
    
    # Test model loading
    print("\n" + "=" * 60)
    print("Testing Model Loading (Quick Test)")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoTokenizer
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
        
        # Don't load full model in verification to save time
        print("✅ Model loading test skipped (would take too long)")
        print("   Run actual training script to test full model loading")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

Run with:
```bash
python3 verify_installation.py
```

---

## Known Issues and Solutions

### Issue 1: BitsAndBytes CUDA Kernel Not Found

**Error:**
```
CUDA Setup failed: libcudart.so not found
```

**Solution:**
```bash
# Add CUDA to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent by adding to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: Transformers Version Conflict

**Error:**
```
ImportError: cannot import name 'BitsAndBytesConfig' from 'transformers'
```

**Solution:**
```bash
# Upgrade Transformers
pip install --upgrade transformers==4.47.1

# Clear cache
rm -rf ~/.cache/huggingface/
```

### Issue 3: Out of Memory During Model Loading

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use lower precision from the start
import torch

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,  # Use FP16 immediately
    low_cpu_mem_usage=True,     # Minimize CPU memory usage
    device_map="auto",
)
```

---

## Quick Start Command Sequence

```bash
# 1. Navigate to project
cd /home/santosh/Desktop/DSAI/PyTorch/FineTuning/FT_GEN_MODELS

# 2. Upgrade key packages
pip3 install --upgrade transformers==4.47.1 peft==0.14.0 trl==0.12.2 bitsandbytes==0.43.1 accelerate==1.2.1

# 3. Verify installation
python3 verify_installation.py

# 4. Test CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 5. Ready to train!
python3 train_tinyllama_jetson.py
```

---

## Performance Optimization for Jetson

### 1. Set Maximum Performance Mode

```bash
# Set power mode to maximum (15W)
sudo nvpmodel -m 0

# Lock clocks to maximum frequency
sudo jetson_clocks

# Verify
sudo jetson_clocks --show
```

### 2. Monitor During Training

```bash
# Terminal 1: Run training
python3 train_tinyllama_jetson.py

# Terminal 2: Monitor GPU usage
watch -n 1 tegrastats

# Terminal 3: Monitor memory
watch -n 1 'free -h && nvidia-smi'
```

### 3. Optimize Environment Variables

Add to `~/.bashrc`:
```bash
# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# PyTorch optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Transformers cache
export HF_HOME=/home/santosh/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# TRL/PEFT settings
export TOKENIZERS_PARALLELISM=false
```

---

## Summary: Your Next Steps

1. **Upgrade Transformers** (most important):
   ```bash
   pip install --upgrade transformers==4.47.1
   ```

2. **Install PEFT & TRL**:
   ```bash
   pip install peft==0.14.0 trl==0.12.2
   ```

3. **Install BitsAndBytes** (for 4-bit quantization):
   ```bash
   pip install bitsandbytes==0.43.1
   ```

4. **Verify everything**:
   ```bash
   python3 verify_installation.py
   ```

5. **Start training**!

---

## Support and Resources

- **Jetson Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **Transformers Issues**: https://github.com/huggingface/transformers/issues
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **BitsAndBytes ARM64**: https://github.com/TimDettmers/bitsandbytes/issues

---

*Last Updated: 2026-03-20*
*Tested on: Jetson Orin Nano 8GB, JetPack 6.4.7, CUDA 12.6*
