# 🚀 Quick Start: Multi-GPU Training

## Problem with Previous Job 3032859
The script crashed due to `set -u` (exit on undefined variable) when `SLURM_GPUS` wasn't set.
**✅ FIXED** in new scripts!

---

## ✅ How to Submit Training Job

```bash
cd /home/doshlom4/work/final_project

# Use the SIMPLIFIED script (recommended)
sbatch slurm/train1_flickr8_simple.sh
```

You'll see output like:
```
Submitted batch job 3032860
```

---

## 📊 How to Monitor Progress

### Option 1: Automated Monitoring Script (EASIEST)

```bash
# Auto-detect most recent training job
bash slurm/monitor_training.sh

# Or specify job ID
bash slurm/monitor_training.sh 3032860
```

**Shows:**
- Job status (running/pending/completed)
- Recent output
- Training progress (epochs, losses)
- GPU utilization
- All log file locations

### Option 2: Watch Live Logs

```bash
# After submitting, find your job ID
squeue -u $USER

# Watch standard output (job initialization)
tail -f slurm/logs/train1_flickr8_2gpu_<JOB_ID>.out

# Watch training log (epochs, losses) - THIS IS THE MAIN ONE!
tail -f slurm/logs/train1_flickr8_training_<JOB_ID>.log
```

### Option 3: Check GPU Usage

```bash
# Find compute node
squeue -u $USER
# Example output: gpu1, gpu2, etc.

# SSH to that node
ssh gpu1

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Exit with Ctrl+C
```

---

## 📁 Where to Find Outputs

### Training Logs
```
slurm/logs/train1_flickr8_2gpu_<JOB_ID>.out       ← Job initialization
slurm/logs/train1_flickr8_2gpu_<JOB_ID>.err       ← Errors (if any)
slurm/logs/train1_flickr8_training_<JOB_ID>.log   ← Training progress (MAIN)
```

### Model Checkpoints
```
outputs/train12_flickr8k_text2img/unet_step_*.pt   ← Saved every N steps
```

### TensorBoard Logs
```
outputs/train12_flickr8k_text2img/tensorboard/run_*
```

View with:
```bash
tensorboard --logdir outputs/train12_flickr8k_text2img/tensorboard/ --port 6006
```

---

## 🔍 Troubleshooting

### "Job not found in queue"
**Cause:** Job already completed or crashed  
**Solution:** Check logs in `slurm/logs/train1_flickr8_2gpu_*.out`

### "CUDA out of memory"
**Cause:** Your Jupyter sessions are using GPU memory  
**Solution:**
```bash
# Find Jupyter jobs
squeue -u $USER | grep jupyter

# Kill them
scancel 3032765
scancel 3032684

# Resubmit training
sbatch slurm/train1_flickr8_simple.sh
```

### "No output in logs"
**Cause:** Job is still initializing (loading models takes 1-2 minutes)  
**Solution:** Wait 2-3 minutes, then check again

---

## 🎯 Expected Timeline

| Stage | Duration | What's Happening |
|-------|----------|------------------|
| Queue | 0-5 min | Waiting for GPU node |
| Init | 1-3 min | Loading models, dataset |
| Training | Hours | Actual training (check logs!) |

**First epoch typically takes 2-5 minutes**

---

## 💡 Pro Tips

1. **Kill Jupyter before long training** - Frees up GPU memory
2. **Use monitoring script** - `bash slurm/monitor_training.sh`
3. **Watch training log** - `tail -f slurm/logs/train1_flickr8_training_*.log`
4. **Check TensorBoard** - Real-time metrics visualization
5. **Save checkpoints** - Configured to save every 500 steps

---

## 📞 Quick Reference

```bash
# Submit job
sbatch slurm/train1_flickr8_simple.sh

# Monitor
bash slurm/monitor_training.sh

# Check status
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# View logs
tail -f slurm/logs/train1_flickr8_training_*.log
```

---

**Next step:** Submit the job and run the monitoring script!
