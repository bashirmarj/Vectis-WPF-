# QUICK START GUIDE - Vectis Geometry Service v2.0

**For Developers: Get BRepNet running in 30 minutes**

---

## Prerequisites

- Python 3.10+
- Miniconda or Anaconda
- Docker (optional, for deployment)
- Git

---

## Local Development Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/bashirmarj/vectismachining.git
cd vectismachining
git checkout refactor/v2.0-clean-architecture
```

### Step 2: Set Up Python Environment

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment with OpenCascade
conda create -n vectis python=3.10 -y
conda activate vectis
conda install -c conda-forge pythonocc-core=7.7.0 -y

# Navigate to service directory
cd geometry-service-v2

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Download BRepNet Model

```bash
# Create models directory
mkdir -p models
cd models

# Clone BRepNet repository (contains pre-trained weights)
git clone https://github.com/AutodeskAILab/BRepNet.git

# For now, we'll use the checkpoint directly
# In production, convert to ONNX (see conversion script below)
ln -s BRepNet/example_files/pretrained_models/pretrained_s2.0.0_step_all_features_0519_073100.ckpt \
      brepnet_pretrained.ckpt

cd ..
```

### Step 4: Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
PORT=8080
EOF

# Load environment variables
export $(cat .env | xargs)
```

### Step 5: Test Locally

```bash
# Start development server
python app.py

# In another terminal, test with curl
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "service": "vectis-geometry-service",
#   "version": "2.0.0",
#   "brepnet_loaded": true,
#   "turning_detector_loaded": true
# }
```

### Step 6: Test with STEP File

```bash
# Download test file
wget https://github.com/AutodeskAILab/BRepNet/raw/main/example_files/00000005.step

# Analyze it
curl -X POST http://localhost:8080/analyze \
  -F "file=@00000005.step" \
  | jq .

# Expected: JSON with mesh_data and features
```

---

## Convert Model to ONNX (Production)

For production deployment, convert PyTorch checkpoint to ONNX:

```python
# convert_to_onnx.py
import torch
import torch.onnx

# Load BRepNet model (you'll need the model architecture)
# This is a placeholder - actual conversion requires BRepNet source
checkpoint = torch.load('models/brepnet_pretrained.ckpt')
model = checkpoint['model']  # Adjust based on actual checkpoint structure
model.eval()

# Create dummy input
dummy_input = {
    'face_features': torch.randn(1, 100, 10),
    'edge_indices': torch.tensor([[0], [0]], dtype=torch.long),
    'edge_features': torch.randn(1, 1, 5)
}

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input,),
    'models/brepnet_pretrained.onnx',
    input_names=['face_features', 'edge_indices', 'edge_features'],
    output_names=['predictions'],
    dynamic_axes={
        'face_features': {1: 'num_faces'},
        'edge_indices': {1: 'num_edges'},
        'edge_features': {1: 'num_edges'}
    },
    opset_version=14
)

print("✅ Model converted to ONNX")
```

```bash
# Run conversion
python convert_to_onnx.py

# Update brepnet_wrapper.py to use .onnx file
# Change model_path in app.py from .ckpt to .onnx
```

---

## Docker Build & Test

```bash
# Build image
docker build -t vectis-geometry-service:2.0.0 .

# Run container
docker run -p 8080:8080 \
  -e SUPABASE_URL=$SUPABASE_URL \
  -e SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_SERVICE_ROLE_KEY \
  vectis-geometry-service:2.0.0

# Test
curl http://localhost:8080/health
```

---

## Integration with Frontend

### Update Supabase Edge Function

In `supabase/functions/analyze-cad/index.ts`:

```typescript
// Replace old geometry service URL
const GEOMETRY_SERVICE_URL = 
  Deno.env.get('GEOMETRY_SERVICE_URL') || 
  'https://geometry-api.vectismachining.com';

// Call new /analyze endpoint
const response = await fetch(`${GEOMETRY_SERVICE_URL}/analyze`, {
  method: 'POST',
  body: formData,
  headers: {
    'X-Correlation-ID': correlationId
  }
});

const result = await response.json();

// Store mesh_data in cad_meshes table
// Store features in part_features table
```

### Update CADViewer Component

In `src/components/CADViewer.tsx`:

```typescript
// Use new mesh format with face_mapping
const { vertices, indices, normals, face_mapping } = meshData;

// Create BufferGeometry
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', 
  new THREE.Float32BufferAttribute(vertices, 3));
geometry.setIndex(indices);
geometry.setAttribute('normal',
  new THREE.Float32BufferAttribute(normals, 3));

// Add face_id attribute for highlighting
const faceIds = new Float32Array(vertices.length / 3);
Object.entries(face_mapping).forEach(([faceId, data]) => {
  data.triangle_indices.forEach(triIdx => {
    // Set face ID for this triangle's vertices
    faceIds[indices[triIdx * 3] * 3] = parseInt(faceId);
    faceIds[indices[triIdx * 3 + 1] * 3] = parseInt(faceId);
    faceIds[indices[triIdx * 3 + 2] * 3] = parseInt(faceId);
  });
});
geometry.setAttribute('faceId',
  new THREE.Float32BufferAttribute(faceIds, 1));
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'OCC'"

**Solution:**
```bash
# Ensure pythonocc-core is installed
conda install -c conda-forge pythonocc-core=7.7.0

# Verify installation
python -c "from OCC.Core.TopoDS import TopoDS_Shape; print('✅ OCC installed')"
```

### Issue: "BRepNet model not found"

**Solution:**
```bash
# Check model path
ls -la models/
# Should show brepnet_pretrained.ckpt or .onnx

# Download if missing
cd models
git clone https://github.com/AutodeskAILab/BRepNet.git
```

### Issue: "Tessellation failed"

**Solution:**
```bash
# Check STEP file validity
python -c "
from OCC.Core.STEPControl import STEPControl_Reader
reader = STEPControl_Reader()
status = reader.ReadFile('your_file.step')
print(f'Read status: {status}')  # Should be 1 (IFSelect_RetDone)
"
```

### Issue: Docker build fails

**Solution:**
```bash
# Use base image with pythonocc-core pre-installed
# Or build in stages:

# Stage 1: Build environment
FROM conda/miniconda3:latest AS builder
RUN conda create -n vectis python=3.10 -y && \
    conda install -n vectis -c conda-forge pythonocc-core=7.7.0 -y

# Stage 2: Runtime
FROM builder
COPY --from=builder /opt/conda/envs/vectis /opt/conda/envs/vectis
# ... rest of Dockerfile
```

---

## Performance Benchmarks

Expected performance on 4 vCPU (DigitalOcean):

| File Complexity | Processing Time | Memory Usage |
|----------------|-----------------|--------------|
| Simple (100-500 faces) | 1-3 seconds | ~200 MB |
| Medium (500-2000 faces) | 3-5 seconds | ~400 MB |
| Complex (2000+ faces) | 5-10 seconds | ~800 MB |

If processing times exceed these, check:
- CPU throttling
- Memory constraints
- ONNX optimization enabled

---

## Next Steps

1. **Test with real customer files** - Run through 10-20 actual STEP files from quotes
2. **Tune confidence thresholds** - Adjust `confidence_threshold` in brepnet_wrapper.py
3. **Deploy to staging** - Test on DigitalOcean before production
4. **Monitor performance** - Set up logging and metrics
5. **Train custom model** - Begin AAGNet training on your dataset (Month 3)

---

## Resources

- **BRepNet Paper:** https://arxiv.org/abs/2104.00706
- **OpenCascade Docs:** https://dev.opencascade.org/
- **Deployment Guide:** [DEPLOYMENT_PLAN.md](./DEPLOYMENT_PLAN.md)
- **Cleanup Guide:** [CLEANUP_PLAN.md](./CLEANUP_PLAN.md)

---

## Support

**Issues:** https://github.com/bashirmarj/vectismachining/issues  
**Discussions:** GitHub Discussions  
**Email:** support@vectismachining.com

---

**Last Updated:** November 13, 2025  
**Version:** 2.0.0
