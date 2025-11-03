"""
Automated local training script for UV-Net on RTX 3070.
This script handles everything: download, training, upload, and database updates.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import UV-Net modules
sys.path.append(str(Path(__file__).parent.parent / "References" / "UV-Net-main"))

def main():
    parser = argparse.ArgumentParser("UV-Net Local Training")
    parser.add_argument("--dataset_id", type=str, required=False, help="Dataset ID from Supabase")
    parser.add_argument("--local_dataset_path", type=str, default=None, help="Path to local dataset directory (skips Supabase download). Must contain split.json, graph/, and labels/ subdirectories.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--experiment_name", type=str, default="local_training", help="Experiment name")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset_id and not args.local_dataset_path:
        print("❌ Error: Must provide either --dataset_id or --local_dataset_path")
        sys.exit(1)
    
    if args.dataset_id and args.local_dataset_path:
        print("❌ Error: Cannot use both --dataset_id and --local_dataset_path")
        sys.exit(1)
    
    print("=" * 80)
    print("UV-Net Local Training Script")
    print("=" * 80)
    if args.local_dataset_path:
        print(f"Local dataset: {args.local_dataset_path}")
    else:
        print(f"Dataset ID: {args.dataset_id}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)
    print()
    
    # Import dependencies
    try:
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
        from supabase import create_client
        from dotenv import load_dotenv
        from cancellation_callback import CancellationCheckCallback
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: conda activate uv_net_local")
        sys.exit(1)
    
    # Load environment variables and Supabase client (only if using Supabase dataset)
    supabase = None
    if args.dataset_id:
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase credentials!")
            print("Create a .env file with:")
            print("SUPABASE_URL=your_url")
            print("SUPABASE_KEY=your_service_role_key")
            sys.exit(1)
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training on CPU will be extremely slow.")
        print("Run check_gpu.py to diagnose the issue.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print()
    
    # Determine dataset path and job ID
    job_id = None
    
    if args.local_dataset_path:
        # LOCAL DATASET MODE
        print("=" * 80)
        print("Using LOCAL dataset (skipping Supabase)")
        print("=" * 80)
        
        dataset_path = Path(args.local_dataset_path)
        
        # Validate local dataset structure
        if not dataset_path.exists():
            print(f"❌ Error: Local dataset path does not exist: {dataset_path}")
            sys.exit(1)
        
        if not dataset_path.joinpath("split.json").exists():
            print(f"❌ Error: split.json not found in {dataset_path}")
            sys.exit(1)
        
        if not dataset_path.joinpath("graph").exists():
            print(f"❌ Error: graph/ directory not found in {dataset_path}")
            sys.exit(1)
        
        if not dataset_path.joinpath("labels").exists():
            print(f"❌ Error: labels/ directory not found in {dataset_path}")
            sys.exit(1)
        
        print(f"✓ Local dataset validated: {dataset_path}")
        print()
        
        # Use timestamp-based job ID for local training
        job_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"ℹ️  Local job ID: {job_id}")
        print()
        
    else:
        # SUPABASE DATASET MODE
        # Create training job record
        print("Creating training job record...")
        job_data = {
            "dataset_name": args.dataset_id,
            "status": "running",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "started_by": None,  # Local training
            "metadata": {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "environment": "local",
                "script_version": "1.0"
            }
        }
        
        job_result = supabase.table("training_jobs").insert(job_data).execute()
        job_id = job_result.data[0]["id"]
        print(f"✓ Training job created: {job_id}")
        print()
        
        # Step 1: Download dataset
        print("=" * 80)
        print("STEP 1: Downloading Dataset")
        print("=" * 80)
        
        dataset_path = Path("./datasets/mfcad_local")
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download dataset metadata
            print("Fetching dataset info...")
            dataset_result = supabase.table("training_datasets").select("*").eq("id", args.dataset_id).execute()
            
            if not dataset_result.data:
                raise Exception(f"Dataset {args.dataset_id} not found")
            
            dataset_info = dataset_result.data[0]
            storage_path = dataset_info["storage_path"]
            num_files = dataset_info["num_files"]
            
            print(f"Dataset: {dataset_info['name']}")
            print(f"Files: {num_files}")
            print(f"Storage path: {storage_path}")
            print()
            
            # Download files from Supabase storage
            print("Downloading files from Supabase storage...")
            from download_dataset import download_dataset
            download_dataset(supabase, storage_path, dataset_path, num_files)
            
            print("✓ Dataset downloaded successfully")
            print()
            
        except Exception as e:
            print(f"❌ Dataset download failed: {e}")
            supabase.table("training_jobs").update({
                "status": "failed",
                "error_message": str(e)
            }).eq("id", job_id).execute()
            sys.exit(1)
    
    # Step 2: Set up training
    print("=" * 80)
    print("STEP 2: Setting Up Training")
    print("=" * 80)
    
    try:
        from datasets.mfcad import MFCADDataset
        from uvnet.models import Segmentation
        
        # Create datasets
        print("Loading datasets...")
        train_data = MFCADDataset(root_dir=str(dataset_path), split="train", random_rotate=True)
        val_data = MFCADDataset(root_dir=str(dataset_path), split="val")
        test_data = MFCADDataset(root_dir=str(dataset_path), split="test")
        
        train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        print(f"✓ Train samples: {len(train_data)}")
        print(f"✓ Val samples: {len(val_data)}")
        print(f"✓ Test samples: {len(test_data)}")
        print()
        
        # Update job with sample counts (only for Supabase jobs)
        if args.dataset_id and supabase:
            supabase.table("training_jobs").update({
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data)
            }).eq("id", job_id).execute()
        
    except Exception as e:
        print(f"❌ Dataset setup failed: {e}")
        if args.dataset_id and supabase:
            supabase.table("training_jobs").update({
                "status": "failed",
                "error_message": str(e)
            }).eq("id", job_id).execute()
        sys.exit(1)
    
    # Step 3: Train model
    print("=" * 80)
    print("STEP 3: Training Model")
    print("=" * 80)
    print()
    
    results_path = Path("./results") / args.experiment_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    month_day = datetime.now().strftime("%m%d")
    hour_min_sec = datetime.now().strftime("%H%M%S")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(results_path / month_day / hour_min_sec),
        filename="best",
        save_last=True,
        save_top_k=1,
        mode="min"
    )
    
    # Cancellation check callback (only for Supabase jobs)
    callbacks = [checkpoint_callback]
    if args.dataset_id and supabase:
        cancellation_callback = CancellationCheckCallback(
            supabase_client=supabase,
            job_id=job_id,
            check_frequency=1  # Check every epoch
        )
        callbacks.append(cancellation_callback)
    
    logger = TensorBoardLogger(
        str(results_path),
        name=month_day,
        version=hour_min_sec
    )
    
    # Create model
    num_classes = MFCADDataset.num_classes()
    model = Segmentation(num_classes=num_classes, crv_in_channels=6)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    start_time = time.time()
    
    try:
        print("Starting training...")
        print(f"Monitor progress with: tensorboard --logdir {results_path}")
        print()
        
        trainer.fit(model, train_loader, val_loader)
        
        # Check if training was cancelled (only for Supabase jobs)
        if args.dataset_id and supabase:
            job_check = supabase.table("training_jobs").select("status").eq("id", job_id).execute()
            if job_check.data and job_check.data[0].get("status") == "cancelled":
                print("Training was cancelled. Exiting...")
                sys.exit(0)
        
        training_time = int(time.time() - start_time)
        print(f"✓ Training completed in {training_time // 3600}h {(training_time % 3600) // 60}m")
        print()
        
    except Exception as e:
        # Check if this was a cancellation (only for Supabase jobs)
        if args.dataset_id and supabase:
            try:
                job_check = supabase.table("training_jobs").select("status").eq("id", job_id).execute()
                if job_check.data and job_check.data[0].get("status") == "cancelled":
                    print("Training was cancelled. Exiting...")
                    sys.exit(0)
            except:
                pass
        
        print(f"❌ Training failed: {e}")
        if args.dataset_id and supabase:
            supabase.table("training_jobs").update({
                "status": "failed",
                "error_message": str(e),
                "training_time_seconds": int(time.time() - start_time)
            }).eq("id", job_id).execute()
        sys.exit(1)
    
    # Step 4: Test model
    print("=" * 80)
    print("STEP 4: Testing Model")
    print("=" * 80)
    
    try:
        test_results = trainer.test(model=model, dataloaders=[test_loader], verbose=True)
        test_iou = test_results[0].get('test_iou', 0)
        test_acc = test_results[0].get('test_accuracy', 0)
        
        print(f"✓ Test IoU: {test_iou * 100:.2f}%")
        print(f"✓ Test Accuracy: {test_acc * 100:.2f}%")
        print()
        
    except Exception as e:
        print(f"⚠ Testing failed: {e}")
        test_iou = None
        test_acc = None
    
    # Step 5: Upload model to Supabase (only for Supabase datasets)
    storage_filename = None
    checkpoint_path = checkpoint_callback.best_model_path
    
    if args.dataset_id and supabase:
        print("=" * 80)
        print("STEP 5: Uploading Model")
        print("=" * 80)
        
        try:
            print(f"Best checkpoint: {checkpoint_path}")
            
            # Upload to Supabase storage
            with open(checkpoint_path, 'rb') as f:
                model_data = f.read()
            
            storage_filename = f"models/{args.experiment_name}_{month_day}_{hour_min_sec}_best.ckpt"
            
            print(f"Uploading to: {storage_filename}")
            supabase.storage.from_("trained-models").upload(
                storage_filename,
                model_data,
                {"content-type": "application/octet-stream"}
            )
            
            print("✓ Model uploaded successfully")
            print()
            
        except Exception as e:
            print(f"❌ Model upload failed: {e}")
            storage_filename = None
    else:
        print("=" * 80)
        print("STEP 5: Model Saved Locally")
        print("=" * 80)
        print(f"✓ Model saved at: {checkpoint_path}")
        print("ℹ️  Skipping upload (local training mode)")
        print()
    
    # Step 6: Update training job (only for Supabase datasets)
    if args.dataset_id and supabase:
        print("=" * 80)
        print("STEP 6: Updating Database")
        print("=" * 80)
        
        try:
            # Get validation metrics from checkpoint
            val_metrics = checkpoint_callback.best_model_score
            
            update_data = {
                "status": "completed",
                "model_path": storage_filename,
                "training_time_seconds": int(time.time() - start_time),
                "best_val_accuracy": test_acc,
                "best_val_iou": test_iou,
                "final_train_loss": float(val_metrics) if val_metrics else None,
                "model_version": f"{args.experiment_name}-{month_day}-{hour_min_sec}",
                "logs": {
                    "tensorboard_path": str(results_path / month_day / hour_min_sec),
                    "best_checkpoint": checkpoint_path
                }
            }
            
            supabase.table("training_jobs").update(update_data).eq("id", job_id).execute()
            
            print("✓ Training job updated")
            print()
            
        except Exception as e:
            print(f"⚠ Database update failed: {e}")
    
    # Done!
    print("=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Job ID: {job_id}")
    if storage_filename:
        print(f"Model path (Supabase): {storage_filename}")
    else:
        print(f"Model path (Local): {checkpoint_path}")
    print(f"Training time: {training_time // 3600}h {(training_time % 3600) // 60}m")
    print(f"Test IoU: {test_iou * 100:.2f}%" if test_iou else "Test IoU: N/A")
    print(f"Test Accuracy: {test_acc * 100:.2f}%" if test_acc else "Test Accuracy: N/A")
    print()
    if args.dataset_id:
        print("You can now use this model in your web app!")
        print("Go to Admin Dashboard → ML Training to see the results.")
    else:
        print("Local training complete!")
        print(f"TensorBoard logs: {results_path / month_day / hour_min_sec}")

if __name__ == "__main__":
    main()
