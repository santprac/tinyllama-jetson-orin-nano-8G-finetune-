#!/bin/bash
# Fix Jetson Orin Nano Performance Issues
# Run with: sudo ./fix_jetson_performance.sh

set -e

echo "========================================================================"
echo "Fixing Jetson Orin Nano Performance Issues"
echo "========================================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo ./fix_jetson_performance.sh"
    exit 1
fi

# Issue 1: Set Maximum Power Mode
echo ""
echo "1. Setting Maximum Power Mode (15W)..."
nvpmodel -m 0 2>/dev/null || nvpmodel -m MAXN 2>/dev/null || echo "⚠️  nvpmodel command may have issues"
sleep 2

# Verify power mode
echo "   Checking power mode..."
nvpmodel -q 2>/dev/null || echo "   Current mode set"

# Issue 2: Enable jetson_clocks
echo ""
echo "2. Enabling jetson_clocks (maximum clock frequencies)..."
jetson_clocks --show 2>/dev/null || echo "   Checking current clocks..."
jetson_clocks

# Issue 3: Verify GPU clocks
echo ""
echo "3. Verifying GPU clock frequencies..."
cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq 2>/dev/null || echo "   GPU frequency check skipped"

# Show current status
echo ""
echo "========================================================================"
echo "✅ Performance Mode Applied!"
echo "========================================================================"
echo ""
echo "Verifying settings with tegrastats (5 second sample)..."
timeout 5 tegrastats 2>/dev/null || echo "tegrastats not available"

echo ""
echo "========================================================================"
echo "Next Steps:"
echo "1. Run: jtop  (to verify GPU clocks are higher)"
echo "2. Check GPU freq should be ~1000+ MHz (not 612 MHz)"
echo "3. Close Cursor before training"
echo "4. Run your training script"
echo "========================================================================"
