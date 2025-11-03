"""
PyTorch Lightning callback to check for training cancellation.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CancellationCheckCallback(Callback):
    """Callback that checks if training should be cancelled by querying the database."""
    
    def __init__(self, supabase_client, job_id: str, check_frequency: int = 1):
        """
        Args:
            supabase_client: Supabase client instance
            job_id: Training job ID to monitor
            check_frequency: Check every N epochs (default: 1)
        """
        self.supabase = supabase_client
        self.job_id = job_id
        self.check_frequency = check_frequency
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for cancellation at the end of each epoch."""
        # Only check at specified frequency
        if trainer.current_epoch % self.check_frequency != 0:
            return
        
        try:
            # Query job status from database
            result = self.supabase.table("training_jobs").select("status").eq("id", self.job_id).execute()
            
            if result.data and len(result.data) > 0:
                status = result.data[0].get("status")
                
                if status == "cancelled":
                    print("\n" + "=" * 80)
                    print("⚠️  TRAINING CANCELLED BY USER")
                    print("=" * 80)
                    print(f"Stopping training at epoch {trainer.current_epoch + 1}")
                    print("Cleaning up and exiting gracefully...")
                    print()
                    
                    # Stop training
                    trainer.should_stop = True
                    
        except Exception as e:
            # Don't stop training if we can't check status (network issue, etc.)
            print(f"⚠️  Warning: Could not check cancellation status: {e}")
