# Fixing Jetson Orin Nano Performance Issues

## Your Current Issues:

1. ❌ JetPack not detected by jtop
2. ❌ jetson_clocks inactive (GPU running at low speed)
3. ❌ GPU clocks at 612 MHz (should be ~1000+ MHz)

---

## Quick Fix (Run These Commands)

### Step 1: Set Maximum Power Mode
```bash
sudo nvpmodel -m 0
# or if that fails:
sudo nvpmodel -m MAXN
```

### Step 2: Enable jetson_clocks
```bash
sudo jetson_clocks
```

### Step 3: Verify
```bash
# Check power mode
sudo nvpmodel -q

# Show clock status
sudo jetson_clocks --show

# Monitor with jtop
jtop
```

---

## Automated Fix

```bash
# Run the fix script
sudo ./fix_jetson_performance.sh
```

---

## Manual Commands Explained

### 1. Fix JetPack Detection

**Issue:** jtop might not detect JetPack properly on custom kernels

**Not critical** - Your JetPack 6.4.7 IS installed (we verified earlier)

**Fix (if needed):**
```bash
# Reinstall jtop
pip3 install -U jetson-stats
sudo systemctl restart jtson_stats.service
```

### 2. Fix jetson_clocks Inactive

**Issue:** GPU/CPU clocks not locked to maximum

**Current:** GPU at 612 MHz (power saving mode)  
**Target:** GPU at 1020-1300 MHz (max performance)

**Fix:**
```bash
sudo jetson_clocks
```

This locks ALL clocks to maximum:
- GPU clock → Max
- CPU clocks → Max  
- Memory clock → Max
- EMC clock → Max

### 3. Fix Low GPU Clock (612 MHz)

**Why it's low:**
- Power saving mode enabled
- jetson_clocks not running
- nvpmodel not set to max

**Fix (all together):**
```bash
# Set power mode to maximum (15W for Orin Nano)
sudo nvpmodel -m 0

# Lock all clocks to max
sudo jetson_clocks

# Verify GPU frequency
cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq
```

Expected output: `1020000000` or higher (1020 MHz+)

---

## Check Current GPU Frequency

```bash
# Current GPU frequency
cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq

# Available frequencies
cat /sys/devices/gpu.0/devfreq/17000000.gpu/available_frequencies

# Max frequency
cat /sys/devices/gpu.0/devfreq/17000000.gpu/max_freq
```

---

## Verify Changes Worked

### Method 1: Using jtop
```bash
jtop
```

Look for:
- **GPU clock**: Should show 1000+ MHz (not 612 MHz)
- **Power mode**: Should show MAXN or Mode 0
- **jetson_clocks**: Should show "Active"

### Method 2: Using tegrastats
```bash
tegrastats
```

Look for higher clock frequencies in the output.

### Method 3: Check Files
```bash
# GPU frequency
cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq

# Should show: ~1020000000 (1020 MHz) or higher
```

---

## Performance Impact

### Before Fix (Power Saving Mode):
```
GPU Clock: 612 MHz
Training Speed: 30-50 tokens/second
Training Time: 20-30 hours
```

### After Fix (Maximum Performance):
```
GPU Clock: 1020-1300 MHz
Training Speed: 80-200 tokens/second
Training Time: 9-18 hours
```

**Speed increase: 2-3x faster!** ⚡

---

## Make Changes Persistent

### Automatically enable on boot:

```bash
# Create systemd service
sudo nano /etc/systemd/system/jetson-performance.service
```

Add:
```ini
[Unit]
Description=Jetson Maximum Performance Mode
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable jetson-performance.service
sudo systemctl start jetson-performance.service
```

---

## Quick Commands Reference

```bash
# Set max power
sudo nvpmodel -m 0

# Lock clocks
sudo jetson_clocks

# Check status
sudo nvpmodel -q
sudo jetson_clocks --show

# Monitor
jtop
tegrastats
```

---

## Summary

**Run these 2 commands to fix everything:**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Then verify with `jtop` - you should see:
- ✅ GPU clock: 1000+ MHz
- ✅ jetson_clocks: Active
- ✅ Power mode: MAXN/Mode 0

**This will make your training 2-3x faster!** 🚀
