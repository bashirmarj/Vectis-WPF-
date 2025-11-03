# ML Model Deployment Guide

## Pre-trained UV-Net Model Integration

This geometry service now includes ML-based feature recognition using the pre-trained UV-Net model from the MFCAD dataset.

### Model Files Required

The following files need to be uploaded to Supabase Storage bucket `trained-models`:

1. **best.ckpt** - Pre-trained model checkpoint
   - Source: `References/cad-feature-detection-master/feature_detector/model/best.ckpt`
   - Destination: `trained-models/best.ckpt`

2. **hparams.yaml** - Model hyperparameters
   - Source: `References/cad-feature-detection-master/feature_detector/model/hparams.yaml`
   - Destination: `trained-models/hparams.yaml`

### Upload Instructions

Since the model files are in the References directory, they need to be uploaded to Supabase Storage:

**Option 1: Via Lovable Cloud UI**
1. Go to Backend → Storage → trained-models bucket
2. Upload `best.ckpt` (from References directory)
3. Upload `hparams.yaml` (from References directory)

**Option 2: Via Python Script**
```python
from supabase import create_client
import os

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

# Upload model checkpoint
with open("References/cad-feature-detection-master/feature_detector/model/best.ckpt", "rb") as f:
    supabase.storage.from_("trained-models").upload("best.ckpt", f)

# Upload hyperparameters
with open("References/cad-feature-detection-master/feature_detector/model/hparams.yaml", "rb") as f:
    supabase.storage.from_("trained-models").upload("hparams.yaml", f)
```

### How It Works

1. **On first request**: The geometry service downloads the model from Supabase Storage and caches it locally in `/tmp/ml_models/`
2. **Graph construction**: STEP files are converted to DGL graphs using face adjacency and UV-grid sampling
3. **Inference**: The UV-Net model predicts 16 feature classes per face
4. **Response**: Results include both heuristic features AND ML predictions

### Feature Classes (16 Classes)

The model can detect:
- **Manufacturing Features**: hole, boss, pocket, slot, chamfer, fillet, groove, step
- **Surface Types**: plane, cylinder, cone, sphere, torus, bspline, revolution, extrusion

### Response Format

```json
{
  "manufacturing_features": { /* ... heuristic results ... */ },
  "ml_features": {
    "face_predictions": [
      {
        "face_id": 0,
        "predicted_class": "hole",
        "confidence": 0.95,
        "probabilities": [0.95, 0.02, 0.01, ...]
      }
    ],
    "feature_summary": {
      "total_faces": 12,
      "hole": 1,
      "boss": 1,
      "pocket": 0,
      "plane": 8,
      "cylinder": 2
    }
  }
}
```

### Fallback Behavior

If the model is unavailable (files not uploaded or ML libraries fail):
- Service continues to work using heuristic feature detection
- Warning logged: "⚠️ ML inference not available"
- Response omits `ml_features` field

### Dependencies

Added to `requirements.txt`:
```
torch==2.0.1
dgl==1.1.2
pytorch-lightning==2.0.9
```

### Performance

- **Model loading**: ~5 seconds on first request (cached afterward)
- **Graph construction**: ~2-3 seconds for typical parts
- **Inference**: <1 second for parts with <100 faces
- **Total overhead**: ~3-5 seconds additional latency

### Future Training

To use a newly trained model:
1. Train UV-Net locally using your dataset
2. Upload new `best.ckpt` to Supabase Storage (overwrites old one)
3. Restart geometry service to pick up new model
4. No code changes needed!

### Monitoring

Check logs for:
- ✅ "ML inference module loaded" - Module initialized
- ✅ "Model downloaded to /tmp/ml_models/best.ckpt" - Model cached
- ✅ "UV-Net model loaded successfully" - Model ready
- ✅ "ML detected {N} faces" - Inference complete
- ⚠️ "ML inference failed (falling back to heuristics)" - Check model files
