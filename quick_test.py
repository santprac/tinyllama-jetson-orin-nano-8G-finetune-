#!/usr/bin/env python3
"""
Quick Setup Test for Jetson Orin Nano (Fast Version)
Tests: Transformers, CUDA, Accelerate, PEFT, TRL - NO MODEL DOWNLOAD
"""

import sys

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

print("=" * 70)
print("  Jetson Orin Nano Quick Setup Verification")
print("=" * 70)

# Test 1: Package Imports
print_section("1. Testing Package Imports")

packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('peft', 'PEFT'),
    ('trl', 'TRL'),
    ('accelerate', 'Accelerate'),
    ('datasets', 'Datasets'),
    ('numpy', 'NumPy'),
]

import_results = []
for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:20s} v{version}")
        import_results.append(True)
    except ImportError as e:
        print(f"❌ {name:20s} FAILED - {e}")
        import_results.append(False)

# Test 2: CUDA
print_section("2. Testing CUDA")

try:
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        cuda_ok = False
    else:
        print(f"✅ CUDA Available: True")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   cuDNN: {torch.backends.cudnn.version()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU operation
        print("\n   Testing GPU computation...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"✅ GPU matrix multiplication: Success")
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU Memory Allocated: {allocated:.3f} GB")
        
        del x, y, z
        torch.cuda.empty_cache()
        cuda_ok = True
        
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
    cuda_ok = False

# Test 3: Transformers Config (No Model Download)
print_section("3. Testing Transformers (No Download)")

try:
    from transformers import AutoConfig, BitsAndBytesConfig
    
    print("Testing Transformers configuration...")
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"✅ AutoConfig works")
    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    
    # Note: BitsAndBytesConfig won't work, but we can import it
    print(f"✅ Transformers imports successful")
    transformers_ok = True
    
except Exception as e:
    print(f"❌ Transformers test failed: {e}")
    transformers_ok = False

# Test 4: PEFT
print_section("4. Testing PEFT/LoRA")

try:
    from peft import LoraConfig, get_peft_model
    import torch.nn as nn
    
    print("Creating test model with LoRA...")
    
    # Simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.layer(x)
    
    model = TestModel()
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["layer"],
        lora_dropout=0.05,
    )
    
    model = get_peft_model(model, lora_config)
    print(f"✅ LoRA configuration successful")
    model.print_trainable_parameters()
    
    peft_ok = True
    
except Exception as e:
    print(f"❌ PEFT test failed: {e}")
    peft_ok = False

# Test 5: Accelerate
print_section("5. Testing Accelerate")

try:
    from accelerate import Accelerator
    
    accelerator = Accelerator()
    print(f"✅ Accelerate initialized")
    print(f"   Device: {accelerator.device}")
    print(f"   Mixed precision: {accelerator.mixed_precision}")
    
    accelerate_ok = True
    
except Exception as e:
    print(f"❌ Accelerate test failed: {e}")
    accelerate_ok = False

# Test 6: TRL
print_section("6. Testing TRL")

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print(f"✅ SFTTrainer imported")
    
    # Test TrainingArguments
    training_args = TrainingArguments(
        output_dir="./test",
        per_device_train_batch_size=1,
        num_train_epochs=1,
    )
    print(f"✅ TrainingArguments created")
    
    trl_ok = True
    
except Exception as e:
    print(f"❌ TRL test failed: {e}")
    trl_ok = False

# Summary
print_section("SUMMARY")

results = [
    ("Package Imports", all(import_results)),
    ("CUDA", cuda_ok),
    ("Transformers", transformers_ok),
    ("PEFT/LoRA", peft_ok),
    ("Accelerate", accelerate_ok),
    ("TRL", trl_ok),
]

passed = sum(1 for _, ok in results if ok)
total = len(results)

print(f"\nTest Results: {passed}/{total} passed\n")

for name, ok in results:
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"{status:10s} {name}")

print("\n" + "=" * 70)

if passed == total:
    print("🎉 ALL TESTS PASSED!")
    print("\n✅ Your Jetson Orin Nano is ready for TinyLlama fine-tuning!")
    print("\nKey Information:")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - Transformers: {__import__('transformers').__version__}")
    print(f"   - CUDA: {torch.version.cuda}")
    print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\nNext Steps:")
    print("1. Review FINAL_SETUP_SUMMARY.md for training guide")
    print("2. Create training_data.json with your data")
    print("3. Set max performance: sudo nvpmodel -m 0 && sudo jetson_clocks")
    print("4. Start training!")
    print("\nNote: Using FP16 precision (no BitsAndBytes needed)")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nPlease check SOLUTION_NO_BITSANDBYTES.md for troubleshooting")

print("=" * 70)

sys.exit(0 if passed == total else 1)
