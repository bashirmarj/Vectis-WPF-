"""
Real-time training monitor for UV-Net.
Displays GPU usage, training progress, and estimated time remaining.
Run this in a separate terminal while training.
"""

import os
import sys
import time
from pathlib import Path

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def get_gpu_info():
    """Get GPU utilization and temperature"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature
            }
    except ImportError:
        pass
    
    # Fallback to nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return {
                'name': parts[0].strip(),
                'load': float(parts[1].strip()),
                'memory_used': int(parts[2].strip()),
                'memory_total': int(parts[3].strip()),
                'memory_percent': (int(parts[2].strip()) / int(parts[3].strip())) * 100,
                'temperature': int(parts[4].strip())
            }
    except Exception:
        pass
    
    return None

def get_training_progress():
    """Get training progress from TensorBoard logs"""
    results_path = Path("./results")
    
    if not results_path.exists():
        return None
    
    # Find latest experiment
    latest_log = None
    latest_time = 0
    
    for log_dir in results_path.rglob("events.out.tfevents.*"):
        mtime = log_dir.stat().st_mtime
        if mtime > latest_time:
            latest_time = mtime
            latest_log = log_dir
    
    if not latest_log:
        return None
    
    # Parse TensorBoard events
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(str(latest_log.parent))
        ea.Reload()
        
        # Get latest metrics
        progress = {}
        
        if 'train_loss' in ea.scalars.Keys():
            train_loss = ea.scalars.Items('train_loss')
            if train_loss:
                progress['train_loss'] = train_loss[-1].value
                progress['step'] = train_loss[-1].step
        
        if 'val_loss' in ea.scalars.Keys():
            val_loss = ea.scalars.Items('val_loss')
            if val_loss:
                progress['val_loss'] = val_loss[-1].value
        
        if 'val_accuracy' in ea.scalars.Keys():
            val_acc = ea.scalars.Items('val_accuracy')
            if val_acc:
                progress['val_accuracy'] = val_acc[-1].value
        
        if 'val_iou' in ea.scalars.Keys():
            val_iou = ea.scalars.Items('val_iou')
            if val_iou:
                progress['val_iou'] = val_iou[-1].value
        
        return progress
    except Exception as e:
        return None

def print_progress_bar(percent, width=40):
    """Print a progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    
    if COLORS:
        if percent < 30:
            color = Fore.RED
        elif percent < 70:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
        return f"{color}{bar}{Style.RESET_ALL} {percent:.1f}%"
    else:
        return f"{bar} {percent:.1f}%"

def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("UV-Net Training Monitor")
    print("=" * 80)
    print()
    print("This will monitor your training progress in real-time.")
    print("Press Ctrl+C to exit.")
    print()
    
    start_time = time.time()
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("=" * 80)
            if COLORS:
                print(f"{Fore.CYAN}UV-Net Training Monitor{Style.RESET_ALL}")
            else:
                print("UV-Net Training Monitor")
            print("=" * 80)
            print()
            
            # Runtime
            runtime = time.time() - start_time
            print(f"Runtime: {format_time(runtime)}")
            print()
            
            # GPU info
            gpu_info = get_gpu_info()
            if gpu_info:
                print("GPU STATUS:")
                print(f"  Device: {gpu_info['name']}")
                print(f"  Utilization: {print_progress_bar(gpu_info['load'])}")
                print(f"  Memory: {print_progress_bar(gpu_info['memory_percent'])} ({gpu_info['memory_used']:.0f} / {gpu_info['memory_total']:.0f} MB)")
                
                temp = gpu_info['temperature']
                if COLORS:
                    if temp > 85:
                        temp_color = Fore.RED
                    elif temp > 75:
                        temp_color = Fore.YELLOW
                    else:
                        temp_color = Fore.GREEN
                    print(f"  Temperature: {temp_color}{temp}°C{Style.RESET_ALL}")
                else:
                    print(f"  Temperature: {temp}°C")
            else:
                print("GPU STATUS: Not available")
                print("  (Install GPUtil: pip install gputil)")
            
            print()
            
            # Training progress
            progress = get_training_progress()
            if progress:
                print("TRAINING PROGRESS:")
                
                if 'step' in progress:
                    print(f"  Step: {progress['step']}")
                
                if 'train_loss' in progress:
                    print(f"  Train Loss: {progress['train_loss']:.4f}")
                
                if 'val_loss' in progress:
                    print(f"  Val Loss: {progress['val_loss']:.4f}")
                
                if 'val_accuracy' in progress:
                    acc = progress['val_accuracy'] * 100
                    print(f"  Val Accuracy: {acc:.2f}%")
                
                if 'val_iou' in progress:
                    iou = progress['val_iou'] * 100
                    print(f"  Val IoU: {iou:.2f}%")
            else:
                print("TRAINING PROGRESS: Waiting for logs...")
                print("  (Training may not have started yet)")
            
            print()
            print("=" * 80)
            print("Refreshing in 5 seconds... (Ctrl+C to exit)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
