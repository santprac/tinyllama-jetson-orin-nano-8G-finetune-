#!/usr/bin/env bash
set -euo pipefail

echo "=== Restoring Jetson GUI ==="
sudo systemctl start gdm
echo "Done."
