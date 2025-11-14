# CAD Geometry Analysis Microservice

Python-based microservice that provides real geometry analysis for STEP/IGES CAD files using pythonOCC.

## Features

- Real 3D geometry parsing (not filename heuristics)
- Accurate volume, surface area, and bounding box calculations
- Cylindrical part detection based on face analysis
- Flat surface and feature detection
- 95% confidence analysis (vs 30% for heuristic fallback)

## Deployment Options

### Option 1: Railway (Recommended - Easy)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Create new project: `railway init`
4. Deploy from this directory:
   ```bash
   cd geometry-service
   railway up
   ```
5. Copy the generated URL (e.g., `https://your-service.railway.app`)
6. Add as secret in your Lovable project: `GEOMETRY_SERVICE_URL=https://your-service.railway.app`

### Option 2: Render

1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your repository
4. Set **Root Directory**: `geometry-service`
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app`
7. Copy the generated URL
8. Add as secret in your Lovable project

### Option 3: DigitalOcean App Platform

1. Go to [cloud.digitalocean.com/apps](https://cloud.digitalocean.com/apps)
2. Create new app from repository
3. Select `geometry-service` directory
4. Dockerfile will be auto-detected
5. Deploy and copy URL
6. Add as secret in your Lovable project

### Option 4: Docker (Self-Hosted)

```bash
cd geometry-service
docker build -t cad-geometry-service .
docker run -p 5000:5000 cad-geometry-service
```

Then expose via nginx/caddy with HTTPS.

## Environment Setup

After deploying, add the service URL to your Lovable project:

1. In Lovable project settings, add secret:
   - Name: `GEOMETRY_SERVICE_URL`
   - Value: `https://your-deployed-service-url.com`

2. The edge function will automatically use this service for STEP/IGES analysis

## Testing Locally

```bash
cd geometry-service
pip install -r requirements.txt
python app.py
```

Test with curl:
```bash
curl -X POST http://localhost:5000/analyze-cad \
  -F "file=@/path/to/your/file.stp"
```

## API Endpoints

### `GET /health`
Health check endpoint
- Returns: `{"status": "healthy", "service": "cad-geometry-analyzer"}`

### `POST /analyze-cad`
Analyze STEP/IGES file
- Content-Type: `multipart/form-data`
- Body: `file` (STEP/IGES file)
- Returns: JSON with geometry properties

Example response:
```json
{
  "volume_cm3": 181.99,
  "surface_area_cm2": 1091.96,
  "part_width_cm": 8.5,
  "part_height_cm": 12.3,
  "part_depth_cm": 8.5,
  "complexity_score": 5,
  "is_cylindrical": true,
  "has_flat_surfaces": true,
  "cylindrical_faces": 12,
  "planar_faces": 8,
  "total_faces": 20,
  "confidence": 0.95,
  "method": "pythonOCC_geometry_parsing"
}
```

## Cost Estimates

- **Railway**: ~$5-10/month (includes 500 hours free)
- **Render**: Free tier available (spins down when idle)
- **DigitalOcean**: $5/month basic app
- **Self-hosted**: Depends on your infrastructure

## Fallback Behavior

If the service is unavailable, the edge function automatically falls back to filename-based heuristics with 30% confidence, ensuring the quotation system continues to work.
