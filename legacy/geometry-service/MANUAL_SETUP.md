# Manual Setup Instructions for BRepNet Model

## Required Manual Step: Download Pre-Trained Model

The BRepNet pre-trained model is **too large** (~200-500 MB) to store in git and must be downloaded manually.

### Step 1: Navigate to Models Directory

```bash
cd geometry-service/models
```

### Step 2: Clone BRepNet Repository

```bash
# Clone the official BRepNet repository
git clone https://github.com/AutodeskAILab/BRepNet.git
```

### Step 3: Copy Pre-trained Checkpoint

```bash
# Navigate to pretrained models
cd BRepNet/example_files/pretrained_models/

# The checkpoint file you need:
# pretrained_s2.0.0_step_all_features_0519_073100.ckpt

# Copy it to the models directory
cp pretrained_s2.0.0_step_all_features_0519_073100.ckpt ../../../brepnet_pretrained.ckpt

# Go back to geometry-service
cd ../../..
```

### Step 4: Cleanup (Optional)

```bash
# Remove the cloned repository to save space
rm -rf BRepNet
```

### Step 5: Verify Installation

```bash
# Check that the model exists
ls -lh brepnet_pretrained.ckpt

# Expected output:
# -rw-r--r--  1 user  staff   200-500M  date  brepnet_pretrained.ckpt
```

### Expected File Location

After completing these steps, you should have:

```
geometry-service/
└── models/
    └── brepnet_pretrained.ckpt  # ← This file (200-500 MB)
```

### Alternative: Direct Download (if GitHub is slow)

If cloning the repository is slow, you can try direct download:

```bash
cd geometry-service/models

# Direct download using wget
wget https://github.com/AutodeskAILab/BRepNet/raw/main/example_files/pretrained_models/pretrained_s2.0.0_step_all_features_0519_073100.ckpt \
  -O brepnet_pretrained.ckpt
```

### Troubleshooting

**Problem**: `git clone` fails or is too slow

**Solution**: Try the direct download method above, or download the file manually from:
- https://github.com/AutodeskAILab/BRepNet/tree/main/example_files/pretrained_models

**Problem**: Model file not loading

**Solution**: Verify the file path in `app.py` line 73:
```python
brepnet_recognizer = BRepNetRecognizer(
    model_path="models/brepnet_pretrained.ckpt",  # ← Check this path
    device="cpu",
    confidence_threshold=0.70
)
```

### Next Steps

Once you have downloaded the model:

1. **Local Testing**: Follow `docs/QUICKSTART.md` to test the service locally
2. **Deployment**: Follow `docs/DEPLOYMENT_PLAN.md` for production deployment
3. **Integration**: Update the Supabase edge function as described in the deployment plan

## Support

If you encounter issues downloading the model:
1. Check the BRepNet repository: https://github.com/AutodeskAILab/BRepNet
2. Verify you have sufficient disk space (need ~1GB for the repo, ~500MB for final model)
3. Ensure you have a stable internet connection for the download
