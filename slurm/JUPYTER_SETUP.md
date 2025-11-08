# Interactive Jupyter Notebook on HPC Cluster

This guide explains how to run Jupyter notebooks interactively on the HPC cluster using Slurm.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐         ┌──────────────────┐
│  Your Local PC  │         │  Proxy Server    │         │  Login Node     │         │  Compute Node    │
│  (Windows/Mac)  │◄───────►│  (sheshet)       │◄───────►│  (login8)       │◄───────►│  (gpu8)          │
│                 │   SSH   │                  │   SSH   │                 │  Slurm  │                  │
│  localhost:     │         │                  │         │                 │         │  Jupyter Lab     │
│  <LOCAL_PORT>   │         │                  │         │                 │         │  :<JUPYTER_PORT> │
└─────────────────┘         └──────────────────┘         └─────────────────┘         └──────────────────┘
       │                                                                                       │
       │                                                                                       │
       └───────────────────────────── SSH Tunnel ─────────────────────────────────────────────┘
                                  (Port Forwarding)
```

**How it works:**
1. **Compute Node (gpu8)**: Runs Jupyter Lab on port `<JUPYTER_PORT>` (e.g., 8889)
2. **Login Node (login8)**: Gateway to the HPC cluster
3. **Proxy Server (sheshet)**: Jump host for external access
4. **Your Local PC**: Access Jupyter via `http://localhost:<LOCAL_PORT>` in your browser

**Port Terminology:**
- `<LOCAL_PORT>`: Port on **your local machine** (can be any available port: 8888, 9999, 8000, etc.)
- `<JUPYTER_PORT>`: Port where Jupyter runs on the **compute node** (shown in job output)
- `<COMPUTE_NODE>`: The GPU/compute node allocated by Slurm (e.g., gpu8.hpc.pub.lan)

## Quick Start

### 1. Submit the Jupyter Job

```bash
sbatch slurm/jupyter_notebook_interactive.sh
```

This will output something like:
```
Submitted batch job 123456
```

### 2. Check Job Status

```bash
squeue -u $USER
```

Wait until the job status shows `RUNNING`.

### 3. Get Connection Information

Use the helper script:
```bash
bash slurm/connect_jupyter.sh 123456
```

Or manually check the output file:
```bash
cat slurm/logs/jupyter_interactive_123456.out
```

### 4. Connect from Your Local Machine

On your **local machine** (laptop/desktop), create an SSH tunnel with port forwarding.

#### For Open University HPC (with proxy):

```bash
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> DOSHLOM4@login8.openu.ac.il
```

**Replace the placeholders:**
- `<LOCAL_PORT>`: Choose any available port on your PC (e.g., `9999`, `8000`, `8888`)
- `<COMPUTE_NODE>`: From job output (e.g., `gpu8.hpc.pub.lan`)
- `<JUPYTER_PORT>`: From job output under "Currently running servers" (e.g., `8889`)

**Example with actual values:**
```bash
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9999:gpu8.hpc.pub.lan:8889 DOSHLOM4@login8.openu.ac.il
```

**Connection Flow Diagram:**
```
Your Browser                SSH Tunnel                    HPC Cluster
━━━━━━━━━━━                ━━━━━━━━━━━                   ━━━━━━━━━━━━━

localhost:9999  ──────►  [Proxy: sheshet]  ──────►  [Login: login8]  ──────►  gpu8:8889
                           (jump host)                  (gateway)              (Jupyter)
    ▲                                                                               │
    │                                                                               │
    └───────────────────────── Encrypted SSH Tunnel ─────────────────────────────┘
```

#### For standard HPC (without proxy):

```bash
ssh -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> <USERNAME>@<HPC_LOGIN_NODE>
```

**Example:**
```bash
ssh -N -L 9999:gpu8.hpc.pub.lan:8889 doshlom4@login8.openu.ac.il
```

**Understanding the SSH command:**
- `-J`: Jump host (proxy server to go through)
- `-N`: Don't execute commands, just forward ports
- `-L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT>`: Port forwarding specification
  - `<LOCAL_PORT>`: Port on your local machine
  - `<COMPUTE_NODE>`: The remote compute node running Jupyter
  - `<JUPYTER_PORT>`: Port where Jupyter is running on the compute node

**Note:** The SSH command will appear to "hang" with no output - this is normal! Keep it running.

### 5. Open Jupyter in Browser

Open your web browser and go to:
```
http://localhost:<LOCAL_PORT>
```

**Example (if you used LOCAL_PORT=9999):**
```
http://localhost:9999
```

You should now see the Jupyter Lab interface with access to your project files!

### Port Conflict Solutions

**If you get "bind: Permission denied" or "Address already in use":**

The `<LOCAL_PORT>` on your machine is already in use. Simply change it to a different number:

```bash
# Try different local ports
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 8000:gpu8.hpc.pub.lan:8889 ...
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9999:gpu8.hpc.pub.lan:8889 ...
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 8890:gpu8.hpc.pub.lan:8889 ...
```

Then access: `http://localhost:8000` (or whatever `<LOCAL_PORT>` you chose)

### 6. Stop the Job When Done

When you're finished, cancel the job to free up resources:

```bash
scancel 123456
```

---

## Configuration

### Customize Resources

Edit `slurm/jupyter_notebook_interactive.sh` to change:

- **GPUs**: `#SBATCH --gres=gpu:1` (change `1` to desired number)
- **CPUs**: `#SBATCH --cpus-per-task=8` (change `8` to desired number)
- **Memory**: `#SBATCH --mem=32G` (change `32G` to desired amount)
- **Time**: `#SBATCH --time=48:00:00` (format: HH:MM:SS)

### Choose Python Environment

Edit the configuration section in `slurm/jupyter_notebook_interactive.sh`:

**For virtualenv** (default):
```bash
ENVIRONMENT_TYPE="virtualenv"
VIRTUALENV_PATH="/home/doshlom4/torch114"
```

**For conda**:
```bash
ENVIRONMENT_TYPE="conda"
CONDA_ENV_NAME="mycondaenv"
```

### Change Default Port

If port 8888 is busy or you want to use a different port:

```bash
JUPYTER_PORT=8889  # Or any port number you prefer
```

The script will automatically find an available port if the default is taken.

---

## Troubleshooting

### Job Won't Start

**Check job queue**:
```bash
squeue -u $USER
```

**View detailed job info**:
```bash
scontrol show job 123456
```

**Check why job is pending**:
```bash
squeue -j 123456 --start
```

### Can't Find Output File

The job might still be initializing. Wait 30-60 seconds and check again:
```bash
ls -lh slurm/logs/jupyter_interactive_*.out
```

### Connection Refused

1. **Verify job is running**:
   ```bash
   squeue -j 123456
   ```

2. **Check if Jupyter started successfully**:
   ```bash
   cat slurm/logs/jupyter_interactive_123456.out
   ```

3. **Check error log**:
   ```bash
   cat slurm/logs/jupyter_interactive_123456.err
   ```

4. **Verify SSH tunnel is running**: On your local machine, the SSH command should be running (it won't show any output, that's normal)

### Port Conflicts

**Error: "bind [127.0.0.1]:<LOCAL_PORT>: Permission denied"**

This means the `<LOCAL_PORT>` is already in use on your **local machine**.

**Solution:** Use a different local port number:
```bash
# Try different ports
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 8000:gpu8.hpc.pub.lan:8889 ...
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9999:gpu8.hpc.pub.lan:8889 ...
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 7777:gpu8.hpc.pub.lan:8889 ...
```

Then access: `http://localhost:<YOUR_CHOSEN_PORT>`

**Common ports already in use on Windows/Mac:**
- 8080: Often used by local web servers
- 8888: Common Jupyter default
- 3000: Node.js development servers
- 5000: Flask development servers

**Good alternative ports to try:** 9999, 8000, 8890, 7777, 9000

### Wrong Port or Node

**Error: Browser shows "Unable to connect" or "Connection refused"**

**Cause:** The `<COMPUTE_NODE>` or `<JUPYTER_PORT>` in your SSH command doesn't match the actual Jupyter server.

**Solution:** 
1. Check the job output for the correct values:
   ```bash
   bash slurm/connect_jupyter.sh 123456
   ```

2. Look for:
   - **Compute Node**: e.g., `gpu8.hpc.pub.lan`
   - **Port**: From "Currently running servers" line, e.g., `8889`

3. Update your SSH tunnel command with the **exact** values from the output

**Example:**
```bash
# Job output shows: http://gpu8.hpc.pub.lan:8889/
# Your SSH command should use:
ssh -J ... -L 9999:gpu8.hpc.pub.lan:8889 ...
                    └────┬────────┘ └─┬─┘
                      Must match   Must match
```

### Port Already in Use

If you see "Address already in use":

1. Kill the existing tunnel:
   ```bash
   # On local machine (Windows PowerShell)
   Get-Process | Where-Object {$_.ProcessName -eq "ssh"} | Stop-Process

   # On local machine (Mac/Linux)
   pkill -f "ssh -N -L"
   ```

2. Or use a different port in the SSH tunnel:
   ```bash
   ssh -J ... -L 8890:gpu8.hpc.pub.lan:8889 ...
   ```
   Then access: `http://localhost:8890`

### Jupyter Not Installed

If you get "Jupyter is not installed":

```bash
# For virtualenv
source /home/doshlom4/torch114/bin/activate
pip install jupyterlab notebook

# For conda
conda activate mycondaenv
conda install -c conda-forge jupyterlab notebook
```

### CUDA Not Available

Check the job output for GPU allocation:
```bash
grep -A 5 "CUDA" slurm/logs/jupyter_interactive_123456.out
```

If no GPU is allocated, check:
1. GPU requested in Slurm script: `#SBATCH --gres=gpu:1`
2. Available GPUs in cluster: `sinfo -o "%20N %10G"`

---

## Advanced Usage

### Running Multiple Jupyter Sessions

You can run multiple sessions simultaneously. Each job will use a different port:

```bash
sbatch slurm/jupyter_notebook_interactive.sh  # Job 1
sbatch slurm/jupyter_notebook_interactive.sh  # Job 2
```

Use the helper script to see all running sessions:
```bash
bash slurm/connect_jupyter.sh
```

### Custom Notebook Directory

Edit the `NOTEBOOK_DIR` variable in the script:
```bash
NOTEBOOK_DIR="/path/to/your/notebooks"
```

### Enable Jupyter Authentication

For additional security, enable token-based authentication by removing these lines from the script:
```bash
--ServerApp.token="" \
--ServerApp.password=""
```

Jupyter will then generate a token shown in the output file.

### Keep Session Running After Disconnect

The Jupyter server runs in the Slurm job, so it continues running even if you:
- Close your SSH tunnel
- Shut down your local machine
- Disconnect from VPN

To reconnect later, just create a new SSH tunnel to the same compute node and port.

---

## Best Practices

1. **Always cancel jobs when done** to free up cluster resources
2. **Use appropriate time limits** - don't request 48 hours if you only need 2
3. **Monitor resource usage**: Check if you're actually using the GPU/memory you requested
4. **Save your work frequently** - jobs can be preempted or hit time limits
5. **Use version control** - commit your notebooks to git regularly

---

## Getting Help

- **Check job logs**: All Slurm logs are in `slurm/logs/` directory
  - `slurm/logs/jupyter_interactive_<JOBID>.out` - Standard output
  - `slurm/logs/jupyter_interactive_<JOBID>.err` - Error output
- **Cluster documentation**: Check your HPC's specific documentation
- **System administrators**: Contact your HPC support team for cluster-specific issues

---

## Complete Step-by-Step Example Workflow

### On HPC (via SSH to login node):

```bash
# 1. Submit the Jupyter job
$ sbatch slurm/jupyter_notebook_interactive.sh
Submitted batch job 3030179

# 2. Wait for job to start (~30 seconds)
$ squeue -u $USER
   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  3030179      work jupyter_ doshlom4  R       0:45      1 gpu8

# 3. Get connection information
$ bash slurm/connect_jupyter.sh 3030179
======================================================================
Jupyter Lab Connection Information
======================================================================
Job ID: 3030179
Status: RUNNING
Compute Node: gpu8
Port: 8889

Active Jupyter Servers:
Currently running servers:
http://gpu8.hpc.pub.lan:8889/ :: /home/doshlom4/work/final_project

Authentication is DISABLED - no token/password required
======================================================================
```

### On Your Local Machine (Windows/Mac/Linux):

```bash
# 4. Create SSH tunnel with port forwarding
# Using the information from step 3:
#   - COMPUTE_NODE: gpu8.hpc.pub.lan
#   - JUPYTER_PORT: 8889
#   - LOCAL_PORT: 9999 (you choose this)

C:\Users\YourName> ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9999:gpu8.hpc.pub.lan:8889 DOSHLOM4@login8.openu.ac.il

# Enter passwords when prompted
# The command will appear to hang - this is NORMAL, keep it running!
```

**Visual representation:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ Your Terminal (SSH Tunnel Running)                                      │
│                                                                          │
│ C:\Users\YourName> ssh -J ... -L 9999:gpu8.hpc.pub.lan:8889 ...        │
│ Password: ****                                                          │
│ Password: ****                                                          │
│ [Cursor blinking - tunnel is active]                                   │
│                                                                          │
│ ⚠️  DO NOT CLOSE THIS WINDOW - Keep the tunnel running!                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5. Open Jupyter in Browser

Open your web browser and navigate to:
```
http://localhost:9999
```

**You should see the Jupyter Lab interface!**

```
┌─────────────────────────────────────────────────────────────────┐
│ 🌐 Browser Address Bar: http://localhost:9999                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ██ JupyterLab                                    [Settings] ⚙  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  📁 File Browser                                                │
│  ├─ notebooks/                                                  │
│  ├─ slurm/                                                      │
│  └─ README.md                                                   │
│                                                                 │
│  [+] Notebook  [+] Console  [+] Other                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. Work on Your Notebooks

- Create new notebooks
- Run code cells
- Install packages
- Access GPU resources
- Save your work

### 7. Clean Up When Finished

**On your local machine:**
```bash
# Press Ctrl+C in the terminal running the SSH tunnel
^C
```

**On HPC:**
```bash
# Cancel the Slurm job to free resources
$ scancel 3030179
```

## Port Forwarding Cheat Sheet

**Quick reference for SSH tunnel command:**

```
ssh -J <PROXY_USER>@<PROXY_HOST> -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> <HPC_USER>@<HPC_LOGIN>
     │                            │  │                                              │
     │                            │  └─ Port forwarding specification               │
     │                            └─ No remote command execution                    │
     └─ Jump through proxy server                                                   └─ HPC login credentials
```

**For Open University HPC:**
```
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> DOSHLOM4@login8.openu.ac.il
        └─────────────────────────────┘         └──────────┘ └─────────────┘ └──────────────┘ └─────────────────────────────┘
              Proxy (Jump Host)               Your choice   From job output  From job output        Login Node

Example values:
  <LOCAL_PORT> = 9999 (choose any available port)
  <COMPUTE_NODE> = gpu8.hpc.pub.lan (from job output)
  <JUPYTER_PORT> = 8889 (from "jupyter server list" in job output)
```

**Resulting connection:**
```
Browser → localhost:9999 → [SSH Tunnel] → gpu8.hpc.pub.lan:8889 (Jupyter)
```

---

## Understanding the Connection Architecture

### Multi-Hop SSH Tunnel Explained

When you work from outside the university network, you need to go through multiple servers:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         CONNECTION PATH DIAGRAM                               │
└───────────────────────────────────────────────────────────────────────────────┘

Step 1: Submit Job on HPC
─────────────────────────────────────────────────────────────────
    $ sbatch slurm/jupyter_notebook_interactive.sh
    
    ┌─────────────────┐
    │  Login Node     │  ──Slurm──►  ┌─────────────────┐
    │  (login8)       │              │  Compute Node   │
    │                 │              │  (gpu8)         │
    │  Submit job     │              │  Jupyter :8889  │
    └─────────────────┘              └─────────────────┘


Step 2: Create SSH Tunnel from Your Local Machine
─────────────────────────────────────────────────────────────────
    ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il \
         -N -L 9999:gpu8.hpc.pub.lan:8889 \
         DOSHLOM4@login8.openu.ac.il

    ┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Your PC     │──SSH─│ Proxy        │──SSH─│ Login Node  │──────│ Compute     │
    │             │      │ (sheshet)    │      │ (login8)    │ Fwd  │ (gpu8)      │
    │ localhost:  │      │              │      │             │ Port │             │
    │ 9999        │      │ Jump Host    │      │ Gateway     │──────│ Jupyter:    │
    │             │      │              │      │             │      │ 8889        │
    └─────────────┘      └──────────────┘      └─────────────┘      └─────────────┘
         │                                                                  │
         └──────────────────── Encrypted SSH Tunnel ──────────────────────┘


Step 3: Access in Browser
─────────────────────────────────────────────────────────────────
    http://localhost:9999
    
    ┌────────────────────────────────────────┐
    │  Browser                               │
    │  ┌──────────────────────────────────┐  │
    │  │ 🌐 localhost:9999                │  │
    │  ├──────────────────────────────────┤  │
    │  │                                  │  │
    │  │  JupyterLab Interface            │  │
    │  │  Running on gpu8:8889            │  │
    │  │                                  │  │
    │  │  Files, Notebooks, Terminal      │  │
    │  │  GPU Access ✓                    │  │
    │  │                                  │  │
    │  └──────────────────────────────────┘  │
    └────────────────────────────────────────┘
```

### Port Mapping Visualization

Understanding which port is which:

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PORT MAPPING DIAGRAM                           │
└──────────────────────────────────────────────────────────────────────┘

Your Local PC          SSH Tunnel              HPC Compute Node
─────────────          ──────────              ────────────────

┌─────────────┐                              ┌──────────────────┐
│  Browser    │                              │  Jupyter Lab     │
│             │                              │                  │
│  Connects   │                              │  Listening on    │
│  to:        │        Port Forwarding       │  port:           │
│             │      ◄─────────────────►     │                  │
│  localhost: │                              │  gpu8.hpc:       │
│  <LOCAL>    │══════════════════════════════│  <JUPYTER>       │
│   ↓         │                              │   ↓              │
│  9999 ────────────────────────────────────────► 8889          │
│  8000 ────────────────────────────────────────► 8889          │
│  7777 ────────────────────────────────────────► 8888          │
└─────────────┘                              └──────────────────┘
   You can                                      Fixed by
   choose any                                   Jupyter
   available                                    (from job
   port here                                    output)


Command Structure:
─────────────────
ssh -J <PROXY> -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> <LOGIN>
                     └─────┬─────┘ └─────┬─────┘ └──────┬──────┘
                      Your choice   From output    From output
                      (9999)        (gpu8.hpc...)  (8889)
```

### Common Scenarios

#### Scenario 1: Standard Setup (No Port Conflicts)
```
Local Port: 8888  →  SSH Tunnel  →  Jupyter Port: 8888
Browser: http://localhost:8888
Command: ssh -J ... -L 8888:gpu8.hpc.pub.lan:8888 ...
```

#### Scenario 2: Local Port Conflict (8888 already in use)
```
Local Port: 9999  →  SSH Tunnel  →  Jupyter Port: 8888
Browser: http://localhost:9999
Command: ssh -J ... -L 9999:gpu8.hpc.pub.lan:8888 ...
```

#### Scenario 3: Remote Port Conflict (Jupyter using 8889)
```
Local Port: 9999  →  SSH Tunnel  →  Jupyter Port: 8889
Browser: http://localhost:9999
Command: ssh -J ... -L 9999:gpu8.hpc.pub.lan:8889 ...
```

#### Scenario 4: Different Compute Node
```
Local Port: 9999  →  SSH Tunnel  →  Jupyter Port: 8889 on gpu5
Browser: http://localhost:9999
Command: ssh -J ... -L 9999:gpu5.hpc.pub.lan:8889 ...
                                └──┬──┘
                            Node changes per job!
```

---

## Quick Reference Card

### Essential Commands

```bash
# ═══════════════════════════════════════════════════════════════
# ON HPC (Login Node)
# ═══════════════════════════════════════════════════════════════

# Start Jupyter Job
sbatch slurm/jupyter_notebook_interactive.sh

# Check Job Status
squeue -u $USER

# Get Connection Info
bash slurm/connect_jupyter.sh <JOB_ID>

# Cancel Job
scancel <JOB_ID>

# View Logs
cat slurm/logs/jupyter_interactive_<JOB_ID>.out
tail -f slurm/logs/jupyter_interactive_<JOB_ID>.err

# ═══════════════════════════════════════════════════════════════
# ON YOUR LOCAL MACHINE
# ═══════════════════════════════════════════════════════════════

# Create SSH Tunnel (Open University HPC with Proxy)
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il \
     -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> \
     DOSHLOM4@login8.openu.ac.il

# Example with actual values:
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il \
     -N -L 9999:gpu8.hpc.pub.lan:8889 \
     DOSHLOM4@login8.openu.ac.il

# Access in Browser
# Open: http://localhost:<LOCAL_PORT>
# Example: http://localhost:9999

# Stop SSH Tunnel
# Press Ctrl+C in the terminal
```

### Port Quick Reference

| Port Type | Description | Example | Where to Find |
|-----------|-------------|---------|---------------|
| `<LOCAL_PORT>` | Port on your local machine | `9999`, `8000` | **You choose** (any available port) |
| `<JUPYTER_PORT>` | Jupyter's port on compute node | `8889`, `8888` | Job output: "Currently running servers" line |
| `<COMPUTE_NODE>` | GPU node running Jupyter | `gpu8.hpc.pub.lan` | Job output: "Node:" line |

### Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **Port conflict on local machine** | Change `<LOCAL_PORT>` to different number (e.g., 9999, 8000, 7777) |
| **Connection refused** | Verify `<COMPUTE_NODE>` and `<JUPYTER_PORT>` match job output exactly |
| **Can't find job output** | Check `slurm/logs/jupyter_interactive_<JOB_ID>.out` |
| **SSH tunnel not working** | Make sure the SSH command is still running (don't close the terminal) |
| **Wrong password prompt** | You'll be prompted twice: once for proxy (sheshet), once for login node (login8) |

### Workflow Checklist

- [ ] Submit job: `sbatch slurm/jupyter_notebook_interactive.sh`
- [ ] Wait for job to start: `squeue -u $USER` shows `R` (running)
- [ ] Get connection details: `bash slurm/connect_jupyter.sh <JOB_ID>`
- [ ] Note the `<COMPUTE_NODE>` and `<JUPYTER_PORT>` from output
- [ ] On local machine: Run SSH tunnel command
- [ ] Enter proxy password (sheshet)
- [ ] Enter login node password (login8)
- [ ] SSH command hangs (normal - keep it running!)
- [ ] Open browser to `http://localhost:<LOCAL_PORT>`
- [ ] Work in Jupyter Lab
- [ ] When done: Ctrl+C to close tunnel
- [ ] Cancel job: `scancel <JOB_ID>`

---
