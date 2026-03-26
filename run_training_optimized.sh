#!/usr/bin/env bash
set -euo pipefail

TRAIN_CMD="python3 -u lora_finetune.py"
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Jetson Optimized Training Launcher ==="

# Must be run from TTY, not GUI
if [[ -n "${DISPLAY:-}" ]]; then
  echo "Run this from TTY only."
  echo "Use: Ctrl+Alt+F3"
  exit 1
fi

echo "[1/10] Max performance mode"
sudo nvpmodel -m 0

echo "[2/10] Locking clocks"
sudo jetson_clocks

echo "[3/10] Memory tuning"
sudo sysctl -w vm.overcommit_memory=1 >/dev/null

echo "[4/10] Stop GUI"
sudo systemctl stop gdm || true

echo "[5/10] Kill desktop/background noise"
pkill -f gnome-shell || true
pkill -f Xorg || true
pkill -f xdg-desktop-portal || true
pkill -f nvpmodel_indicator || true
pkill -f "/usr/local/bin/jtop" || true
pkill -f gnome-software || true
pkill -f goa-daemon || true
pkill -f evolution || true
pkill chromium || true

echo "[6/10] Stop docker if not needed"
sudo systemctl stop docker || true

echo "[7/10] Drop filesystem caches"
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true

echo "[8/10] Final memory snapshot"
free -h
echo
sudo nvpmodel -q || true
echo

echo "[9/10] Python env"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64

echo "[10/10] Starting training"
echo "Logging to: $LOG_FILE"
echo "========================================"

exec bash -c "$TRAIN_CMD 2>&1 | tee $LOG_FILE"
