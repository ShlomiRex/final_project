ssh gpu8.hpc.pub.lan
cd work/final_project
module load anaconda3
conda create --prefix /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025 python=3.10
conda activate /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
