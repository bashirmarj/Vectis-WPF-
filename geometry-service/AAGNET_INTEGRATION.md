# AAGNet Integration Guide

This document explains the AAGNet (Attribute Adjacency Graph Network) integration for multi-task machining feature recognition in the Vectis Machining platform.

## Overview

AAGNet replaces the previous UV-Net implementation with superior feature recognition capabilities:

- **24 feature classes** (vs 16 in UV-Net)
- **Instance segmentation** - groups faces into individual features
- **Bottom face detection** - identifies bottom faces for each feature
- **Extended attributes** - geometric properties for better analysis

## Architecture

```
Frontend (React/TypeScript)
  ↓
Edge Function (analyze-cad)
  ↓
Geometry Service (Flask/Python) → AAGNet Model
  ↓
Database (Supabase) → cad_meshes table with AAGNet fields
```

## Database Schema

The `cad_meshes` table now includes AAGNet-specific columns:

```sql
-- AAGNet output columns
instance_features JSONB,           -- Feature instances with face groups
extended_attributes JSONB,          -- Geometric attributes
semantic_labels INTEGER[],          -- Per-face semantic labels (0-24)
recognition_method VARCHAR(50),     -- 'AAGNet', 'UV-Net', or 'tessellation'
processing_time_ms INTEGER         -- Processing time in milliseconds
```

Additionally, a new table `manufacturing_feature_instances` stores individual feature instances for detailed queries.

## Feature Classes

AAGNet detects 25 feature classes (24 features + stock):

### Holes
- `through_hole` - Complete penetration
- `blind_hole` - Partial depth

### Pockets  
- `triangular_pocket`
- `rectangular_pocket`
- `6sides_pocket`
- `circular_end_pocket`

### Slots
- `triangular_through_slot`
- `rectangular_through_slot`
- `circular_through_slot`
- `rectangular_blind_slot`
- `v_circular_end_blind_slot`
- `h_circular_end_blind_slot`

### Steps
- `rectangular_through_step`
- `2sides_through_step`
- `slanted_through_step`
- `triangular_blind_step`
- `circular_blind_step`
- `rectangular_blind_step`

### Passages
- `triangular_passage`
- `rectangular_passage`
- `6sides_passage`

### Others
- `chamfer`
- `round`
- `Oring`
- `stock` - Base material

## API Response Format

The geometry service returns AAGNet results in this format:

```json
{
  "mesh_data": {
    "vertices": [...],
    "indices": [...],
    "normals": [...],
    "feature_edges": [...]
  },
  "ml_features": {
    "feature_instances": [
      {
        "feature_type": "through_hole",
        "face_ids": [1, 2, 3],
        "confidence": 0.95
      }
    ],
    "num_features_detected": 5,
    "num_faces_analyzed": 24,
    "inference_time_sec": 2.3
  }
}
```

## Frontend Integration

### FeatureTree Component

The updated `FeatureTree.tsx` component displays AAGNet output:

```typescript
interface AAGNetFeatures {
  instances: AAGNetFeatureInstance[];
  semantic_labels: number[];
  extended_attributes?: {
    face_attributes: number[][];
    edge_attributes: number[][];
  };
  num_faces: number;
  num_instances: number;
  processing_time?: number;
}
```

Features are grouped by category (Holes, Pockets, Slots, Steps, Passages, Chamfers) and displayed with confidence scores.

### State Management

The `PartConfigScreen.tsx` uses React's `useReducer` for robust state management:

```typescript
const [filesWithData, dispatch] = useReducer(filesReducer, []);

// Actions
dispatch({ type: 'SET_ANALYZING', fileName, stage: 'ml-inference' });
dispatch({ type: 'SET_ANALYSIS', fileName, analysis });
```

## Backend Setup (For Local Development)

### Prerequisites

1. **AAGNet Repository**
```bash
cd geometry-service/
git clone https://github.com/whjdark/AAGNet.git AAGNet-main
```

2. **Python Dependencies**
```bash
pip install torch==2.0.1
pip install dgl-cu118==1.1.0  # or dgl for CPU only
pip install pythonocc-core==7.5.1
pip install scikit-learn timm torch-ema tqdm h5py
```

3. **Install occwl**
```bash
git clone https://github.com/AutodeskAILab/occwl.git
cd occwl && python setup.py install && cd ..
```

4. **Download Model Weights**
Place the pretrained `weight_on_MFInstseg.pth` file in `AAGNet-main/weights/`

### Integration in app.py

The AAGNet recognizer is available but not yet integrated into the main Flask app. To enable it:

```python
from aagnet_recognizer import AAGNetRecognizer

# Initialize at startup
recognizer = AAGNetRecognizer(device='cpu')  # Use 'cuda' for GPU

# Add to your existing analyze-cad endpoint
ml_features = recognizer.recognize_features(step_file_path)
```

## Performance Considerations

- **Cold start**: First request may take 5-10 seconds for model loading
- **Inference time**: ~2-5 seconds for typical parts (<100 faces)
- **Memory**: ~200MB for model weights + graph construction
- **CPU vs GPU**: CPU inference is sufficient for production use

## Debugging

Check correlation IDs in logs:

```bash
# Frontend logs (browser console)
{
  "correlation_id": "abc123",
  "tier": "FRONTEND",
  "message": "Starting CAD analysis"
}

# Edge function logs
{
  "correlation_id": "abc123",
  "tier": "EDGE",
  "message": "Geometry service processing complete"
}

# Geometry service logs
{
  "correlation_id": "abc123",
  "tier": "GEOMETRY",
  "message": "AAGNet inference complete"
}
```

## Testing

A comprehensive test suite is available in `test_aagnet_integration.py`:

```bash
python test_aagnet_integration.py
```

Tests cover:
- Flask service health
- AAGNet model loading
- STEP file processing
- Database storage
- Edge function integration
- Frontend data flow

## Troubleshooting

### Empty FeatureTree

1. Check browser console for correlation ID logs
2. Verify Edge Function returns `mlFeatures` in network tab
3. Check database for `instance_features` column population

### Timeout Errors

1. Verify model is pre-loaded at startup
2. Check Flask logs for UV-grid computation time
3. Reduce file complexity or UV-grid resolution

### Import Errors

```bash
# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import dgl; print(dgl.__version__)"
python -c "from OCC.Core.STEPControl import STEPControl_Reader"
```

## Future Enhancements

1. **Model Training**: Train custom AAGNet on proprietary datasets
2. **GPU Acceleration**: Enable CUDA for faster inference
3. **Model Quantization**: Reduce memory footprint
4. **Batch Processing**: Process multiple parts simultaneously
5. **Real-time Updates**: WebSocket-based progress streaming

## References

- AAGNet Paper: [AAG-Net: Attributed Adjacency Graph Network](https://arxiv.org/abs/2211.12999)
- MFCAD Dataset: Manufacturing Feature CAD Dataset
- Repository: https://github.com/whjdark/AAGNet
