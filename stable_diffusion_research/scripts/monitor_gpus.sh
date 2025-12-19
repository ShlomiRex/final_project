#!/bin/bash
# Monitor GPU usage in real-time during training

echo "==================================================================="
echo "GPU Monitoring - Press Ctrl+C to stop"
echo "==================================================================="
echo ""

watch -n 2 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F", " '\''{printf "GPU %s: %s | Util: %3s%% | VRAM: %5sMB / %5sMB (%3.0f%%) | Temp: %s°C\n", $1, $2, $3, $4, $5, ($4/$5)*100, $6}'\'
