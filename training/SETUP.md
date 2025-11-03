# Local Training Setup Guide for RTX 3070

This guide will help you set up UV-Net training on your local machine with RTX 3070 GPU.

## Prerequisites

- **GPU**: NVIDIA RTX 3070 (8GB VRAM)
- **OS**: Windows 10/11, Linux, or macOS (with NVIDIA eGPU)
- **Storage**: At least 50GB free space
- **Internet**: For downloading dependencies and datasets

## Step 1: Install Anaconda

### Windows
1. Download Anaconda from: https://www.anaconda.com/download
2. Run the installer (choose "Just Me" option)
3. Check "Add Anaconda to PATH" during installation
4. Open "Anaconda Prompt" from Start Menu

### Linux
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
source ~/.bashrc
```

## Step 2: Install NVIDIA CUDA Toolkit

Your RTX 3070 needs CUDA 11.8 for best compatibility.

### Windows
1. Download CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Run the installer (choose Express installation)
3. Restart your computer

### Linux
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

## Step 3: Verify NVIDIA Driver

Open terminal/command prompt:
```bash
nvidia-smi
```

You should see:
- Driver Version: 520.xx or higher
- CUDA Version: 11.8 or higher
- GPU: NVIDIA GeForce RTX 3070

## Step 4: Create Python Environment

Navigate to the `training/` folder and run:

```bash
conda env create -f environment_local.yml
```

This will install:
- Python 3.9
- PyTorch 2.0 with CUDA 11.8
- PyTorch Lightning
- DGL (Deep Graph Library)
- OCC-WL (CAD geometry library)
- All other dependencies

**Installation time**: 15-30 minutes

## Step 5: Activate Environment

```bash
conda activate uv_net_local
```

## Step 6: Verify GPU Setup

Run the verification script:

```bash
python check_gpu.py
```

Expected output:
```
✓ CUDA is available
✓ GPU detected: NVIDIA GeForce RTX 3070
✓ CUDA version: 11.8
✓ GPU memory: 8192 MB
✓ PyTorch can use GPU
✓ DGL is installed
✓ All dependencies verified
```

## Step 7: Configure Supabase Credentials

Create a `.env` file in the `training/` directory:

```env
SUPABASE_URL=https://inqabwlmvrvqsdrgskju.supabase.co
SUPABASE_KEY=your_service_role_key_here
```

**To get your service role key:**
1. Go to your app
2. Click "View Backend"
3. Go to Settings → API
4. Copy the "service_role" key (keep it secret!)

## Step 8: Run Your First Training

```bash
python train_local.py --dataset_id <your_dataset_id>
```

To find your dataset ID:
1. Go to your app's Admin Dashboard
2. Click on "ML Training"
3. Copy the Dataset ID from the table

### Training Options

```bash
# Basic training (recommended)
python train_local.py --dataset_id abc-123-def

# Custom batch size (if you get memory errors, reduce this)
python train_local.py --dataset_id abc-123-def --batch_size 8

# More epochs for better accuracy
python train_local.py --dataset_id abc-123-def --epochs 150

# Resume from checkpoint
python train_local.py --dataset_id abc-123-def --resume
```

## Step 9: Monitor Training (Optional)

Open a **second terminal** and run:

```bash
conda activate uv_net_local
python monitor_training.py
```

This shows:
- Real-time training progress
- GPU temperature and usage
- Estimated time remaining
- Current accuracy/loss

## Step 10: View TensorBoard Logs

```bash
tensorboard --logdir results/
```

Then open: http://localhost:6006

## Expected Timeline

- **Dataset download**: 5-15 minutes (15,000 files)
- **First epoch**: 8-12 minutes
- **Total training (100 epochs)**: 12-20 hours
- **Model upload**: 2-5 minutes

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python train_local.py --dataset_id abc-123 --batch_size 4
```

### "ModuleNotFoundError: No module named 'dgl'"
Reinstall DGL:
```bash
conda install -c dglteam dgl-cuda11.8
```

### "occwl not found"
```bash
conda install -c lambouj occwl
```

### Training is very slow
- Check GPU is being used: `nvidia-smi` should show Python process
- Check GPU isn't overheating (should be < 85°C)
- Close other GPU applications (games, video editing, etc.)

### Can't connect to Supabase
- Check your `.env` file has correct credentials
- Make sure you're using the **service_role** key, not anon key

## After Training Completes

The script will automatically:
1. ✓ Upload trained model (`best.ckpt`) to Supabase storage
2. ✓ Update `training_jobs` table with results
3. ✓ Save training metrics and logs
4. ✓ Make model available in your web app

You can then go to your web app's Admin Dashboard to see the trained model and use it for predictions!

## Next Steps

Once training is complete:
1. Go to Admin Dashboard → ML Training
2. You should see your job status as "completed"
3. The model is now ready to use for feature detection
4. Test it by uploading a CAD file in the quote system

## Need Help?

Common issues and solutions:
- GPU not detected → Update NVIDIA drivers
- Installation fails → Use Python 3.9 (not 3.10+)
- Training crashes → Reduce batch_size to 4
- Out of disk space → Need 50GB free minimum
