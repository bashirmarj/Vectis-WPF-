"""
Download dataset from Supabase storage to local disk.
Handles the MFCAD dataset structure with train/val/test splits.
"""

import os
from pathlib import Path
from tqdm import tqdm


def download_dataset(supabase, storage_path, local_path, num_files):
    """
    Download dataset from Supabase storage to local directory.
    
    Args:
        supabase: Supabase client
        storage_path: Path in Supabase storage bucket
        local_path: Local directory to save files
        num_files: Expected number of files
    """
    
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # MFCAD dataset structure: train/val/test folders with graph/label subfolders
    splits = ["train", "val", "test"]
    subfolders = ["graph", "label"]
    
    print(f"Downloading dataset to: {local_path}")
    print()
    
    # List all files in storage
    try:
        all_files = supabase.storage.from_("training-datasets").list(storage_path)
    except Exception as e:
        print(f"Error listing files: {e}")
        raise
    
    # Download split.json if it exists
    try:
        split_json = supabase.storage.from_("training-datasets").download(f"{storage_path}/split.json")
        with open(local_path / "split.json", "wb") as f:
            f.write(split_json)
        print("✓ Downloaded split.json")
    except Exception as e:
        print(f"⚠ split.json not found (will be auto-generated): {e}")
    
    # Download all files for each split
    downloaded = 0
    
    for split in splits:
        split_path = local_path / split
        split_path.mkdir(exist_ok=True)
        
        for subfolder in subfolders:
            subfolder_path = split_path / subfolder
            subfolder_path.mkdir(exist_ok=True)
            
            # List files in this subfolder
            storage_subfolder = f"{storage_path}/{split}/{subfolder}"
            
            try:
                files = supabase.storage.from_("training-datasets").list(storage_subfolder)
            except Exception as e:
                print(f"⚠ No files in {storage_subfolder}: {e}")
                continue
            
            # Download each file
            print(f"Downloading {split}/{subfolder}...")
            
            for file in tqdm(files, desc=f"{split}/{subfolder}"):
                if file["name"].endswith(".pkl"):  # Only download pickle files
                    storage_file_path = f"{storage_subfolder}/{file['name']}"
                    local_file_path = subfolder_path / file["name"]
                    
                    # Skip if already exists
                    if local_file_path.exists():
                        downloaded += 1
                        continue
                    
                    try:
                        file_data = supabase.storage.from_("training-datasets").download(storage_file_path)
                        with open(local_file_path, "wb") as f:
                            f.write(file_data)
                        downloaded += 1
                    except Exception as e:
                        print(f"\n⚠ Failed to download {file['name']}: {e}")
            
            print(f"✓ Downloaded {len(files)} files to {split}/{subfolder}")
    
    print()
    print(f"✓ Total files downloaded: {downloaded}")
    
    if downloaded < num_files * 0.9:  # Allow 10% tolerance
        print(f"⚠ Warning: Expected ~{num_files} files, got {downloaded}")
    
    return downloaded
