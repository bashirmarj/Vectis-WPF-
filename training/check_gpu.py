"""
GPU and dependency verification script for UV-Net training.
Run this after creating the conda environment to verify everything is set up correctly.
"""

import sys

def print_status(message, success=True):
    """Print colored status message"""
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        if success:
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {message}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} {message}")
    except ImportError:
        prefix = "✓" if success else "✗"
        print(f"{prefix} {message}")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_status(f"CUDA is available", cuda_available)
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print_status(f"GPU detected: {device_name}", True)
            
            cuda_version = torch.version.cuda
            print_status(f"CUDA version: {cuda_version}", True)
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_status(f"GPU memory: {gpu_memory:.1f} GB", True)
            
            # Test GPU computation
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print_status("PyTorch can use GPU", True)
            
            return True
        else:
            print_status("CUDA is not available", False)
            print("\nTroubleshooting:")
            print("1. Make sure you have an NVIDIA GPU")
            print("2. Install CUDA Toolkit 11.8 from https://developer.nvidia.com/cuda-downloads")
            print("3. Update your NVIDIA drivers")
            print("4. Reinstall PyTorch with CUDA support:")
            print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
            return False
    except ImportError as e:
        print_status(f"PyTorch not installed: {e}", False)
        return False

def check_dgl():
    """Check Deep Graph Library"""
    try:
        import dgl
        print_status(f"DGL is installed (version {dgl.__version__})", True)
        
        # Test DGL with CUDA
        try:
            import torch
            if torch.cuda.is_available():
                g = dgl.graph(([0, 1, 2], [1, 2, 3])).to('cuda')
                print_status("DGL can use GPU", True)
        except Exception as e:
            print_status(f"DGL GPU support issue: {e}", False)
        
        return True
    except ImportError:
        print_status("DGL is not installed", False)
        print("Install with: conda install -c dglteam dgl-cuda11.8")
        return False

def check_occwl():
    """Check OCC-WL (CAD geometry library)"""
    try:
        import occwl
        print_status("OCC-WL is installed", True)
        return True
    except ImportError:
        print_status("OCC-WL is not installed", False)
        print("Install with: conda install -c lambouj occwl")
        return False

def check_pytorch_lightning():
    """Check PyTorch Lightning"""
    try:
        import pytorch_lightning as pl
        print_status(f"PyTorch Lightning is installed (version {pl.__version__})", True)
        return True
    except ImportError:
        print_status("PyTorch Lightning is not installed", False)
        print("Install with: conda install -c conda-forge pytorch-lightning")
        return False

def check_other_dependencies():
    """Check other required packages"""
    packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'trimesh': 'Trimesh',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
        'dotenv': 'python-dotenv',
        'supabase': 'supabase',
        'psutil': 'psutil',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print_status(f"{name} is installed", True)
        except ImportError:
            print_status(f"{name} is not installed", False)
            all_ok = False
    
    return all_ok

def check_nvidia_driver():
    """Check NVIDIA driver version"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse driver version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print_status(f"NVIDIA driver detected: {line.split('Driver Version:')[1].split()[0]}", True)
                    return True
        else:
            print_status("nvidia-smi not found", False)
            return False
    except FileNotFoundError:
        print_status("NVIDIA driver not found", False)
        print("Install NVIDIA drivers from: https://www.nvidia.com/download/index.aspx")
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_free = gpu_mem_total - gpu_mem_allocated
            
            print_status(f"GPU memory available: {gpu_mem_free:.1f} GB / {gpu_mem_total:.1f} GB", True)
            
            if gpu_mem_total < 6:
                print_status(f"Warning: Low GPU memory. Reduce batch_size to 4-8", False)
            
            return True
    except Exception as e:
        print_status(f"Could not check GPU memory: {e}", False)
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("UV-Net Training Environment Verification")
    print("=" * 60)
    print()
    
    print("Checking NVIDIA driver...")
    check_nvidia_driver()
    print()
    
    print("Checking CUDA and PyTorch...")
    cuda_ok = check_cuda()
    print()
    
    if cuda_ok:
        print("Checking GPU memory...")
        check_gpu_memory()
        print()
    
    print("Checking Deep Graph Library...")
    dgl_ok = check_dgl()
    print()
    
    print("Checking OCC-WL...")
    occwl_ok = check_occwl()
    print()
    
    print("Checking PyTorch Lightning...")
    pl_ok = check_pytorch_lightning()
    print()
    
    print("Checking other dependencies...")
    deps_ok = check_other_dependencies()
    print()
    
    print("=" * 60)
    if cuda_ok and dgl_ok and occwl_ok and pl_ok and deps_ok:
        print_status("All checks passed! You're ready to train.", True)
        print("\nNext steps:")
        print("1. Create a .env file with your Supabase credentials")
        print("2. Run: python train_local.py --dataset_id <your_dataset_id>")
    else:
        print_status("Some checks failed. Please resolve issues above.", False)
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
