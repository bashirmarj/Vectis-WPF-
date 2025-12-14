# Measurement Service API Documentation

## Overview

The Measurement Service is a standalone Flask/Python application that provides REST API endpoints for CAD measurement calculations. It uses OpenCascade (pythonocc-core) for precise geometric calculations.

**Base URL:** `http://localhost:8081` (default)

**Content-Type:** `application/json` (unless specified otherwise)

## Authentication

Currently, no authentication is required. For production deployments, consider adding API key authentication.

---

## Endpoints

### Health Check

#### `GET /health`

Check if the service is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "service": "vectis-measurement-service",
  "version": "1.0.0",
  "capabilities": [
    "point_to_point",
    "edge_measurement",
    "face_area",
    "face_to_face",
    "angle",
    "unit_conversion"
  ],
  "timestamp": "2025-12-14T10:30:00.000Z"
}
```

---

### Point-to-Point Distance

#### `POST /measure/point-to-point`

Calculate the distance between two 3D points with XYZ delta components.

**Request:**
```json
{
  "point1": [0, 0, 0],
  "point2": [10, 20, 30],
  "unit_system": "metric"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `point1` | `[number, number, number]` | Yes | First point coordinates (X, Y, Z) |
| `point2` | `[number, number, number]` | Yes | Second point coordinates (X, Y, Z) |
| `unit_system` | `"metric" \| "imperial"` | No | Display unit system (default: metric) |

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "point_to_point",
  "value": 37.416573867739416,
  "unit": "mm",
  "label": "Distance: 37.417 mm",
  "display_value": "37.417 mm",
  "delta_x": 10.0,
  "delta_y": 20.0,
  "delta_z": 30.0,
  "point1": [0, 0, 0],
  "point2": [10, 20, 30],
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

---

### Edge Measurement

#### `POST /measure/edge`

Measure an edge's length, diameter (for circles), or radius (for arcs).

**Request:**
```json
{
  "edge_info": {
    "edge_id": 0,
    "edge_type": "circle",
    "start_point": [10, 0, 0],
    "end_point": [10, 0, 0],
    "length": null,
    "radius": 10.0,
    "diameter": 20.0,
    "center": [0, 0, 0]
  },
  "unit_system": "metric"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `edge_info.edge_id` | `number` | No | Edge identifier |
| `edge_info.edge_type` | `"line" \| "circle" \| "arc" \| "ellipse" \| "spline"` | Yes | Edge geometry type |
| `edge_info.start_point` | `[number, number, number]` | Yes | Start point coordinates |
| `edge_info.end_point` | `[number, number, number]` | Yes | End point coordinates |
| `edge_info.length` | `number \| null` | No | Edge length (for lines) |
| `edge_info.radius` | `number \| null` | No | Radius (for arcs/circles) |
| `edge_info.diameter` | `number \| null` | No | Diameter (for circles) |
| `edge_info.center` | `[number, number, number] \| null` | No | Center point (for arcs/circles) |
| `unit_system` | `"metric" \| "imperial"` | No | Display unit system |

**Response (Circle):**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "edge_diameter",
  "value": 20.0,
  "unit": "mm",
  "label": "Diameter: Ø 20.000 mm",
  "display_value": "Ø 20.000 mm",
  "edge_type": "circle",
  "diameter": 20.0,
  "radius": 10.0,
  "center": [0, 0, 0],
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

**Response (Arc):**
```json
{
  "success": true,
  "measurement_type": "edge_radius",
  "value": 10.0,
  "label": "Radius: R 10.000 mm",
  "display_value": "R 10.000 mm",
  "edge_type": "arc",
  "radius": 10.0
}
```

**Response (Line):**
```json
{
  "success": true,
  "measurement_type": "edge_length",
  "value": 50.0,
  "label": "Length: 50.000 mm",
  "display_value": "50.000 mm",
  "edge_type": "line"
}
```

---

### Face Area

#### `POST /measure/face-area`

Calculate the surface area of a face.

**Request:**
```json
{
  "face_info": {
    "face_id": 0,
    "surface_type": "plane",
    "center": [50, 50, 0],
    "normal": [0, 0, 1],
    "area": 10000.0
  },
  "unit_system": "metric"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `face_info.face_id` | `number` | No | Face identifier |
| `face_info.surface_type` | `string` | Yes | Surface type (plane, cylinder, etc.) |
| `face_info.center` | `[number, number, number]` | Yes | Face center point |
| `face_info.normal` | `[number, number, number]` | Yes | Face normal vector |
| `face_info.area` | `number` | Yes | Face area in mm² |
| `unit_system` | `"metric" \| "imperial"` | No | Display unit system |

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "face_area",
  "value": 10000.0,
  "unit": "mm²",
  "label": "Area: 10000.000 mm²",
  "display_value": "10000.000 mm²",
  "face_id": 0,
  "surface_type": "plane",
  "area": 10000.0,
  "normal": [0, 0, 1],
  "point1": [50, 50, 0],
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

---

### Face-to-Face Distance

#### `POST /measure/face-to-face`

Calculate the distance and angle between two faces.

**Request:**
```json
{
  "face1": {
    "center": [0, 0, 0],
    "normal": [0, 0, 1]
  },
  "face2": {
    "center": [0, 0, 10],
    "normal": [0, 0, 1]
  },
  "unit_system": "metric"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `face1.center` | `[number, number, number]` | Yes | First face center point |
| `face1.normal` | `[number, number, number]` | Yes | First face normal vector |
| `face2.center` | `[number, number, number]` | Yes | Second face center point |
| `face2.normal` | `[number, number, number]` | Yes | Second face normal vector |
| `unit_system` | `"metric" \| "imperial"` | No | Display unit system |

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "face_to_face",
  "value": 10.0,
  "unit": "mm",
  "label": "Distance: 10.000 mm",
  "display_value": "10.000 mm (0.0°)",
  "angle": 0.0,
  "is_parallel": true,
  "perpendicular_distance": 10.0,
  "point1": [0, 0, 0],
  "point2": [0, 0, 10],
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

---

### Angle Measurement

#### `POST /measure/angle`

Calculate the angle between two vectors or three points.

**Request (Two Vectors):**
```json
{
  "vector1": [1, 0, 0],
  "vector2": [0, 1, 0]
}
```

**Request (Three Points):**
```json
{
  "point1": [10, 0, 0],
  "vertex": [0, 0, 0],
  "point2": [0, 10, 0]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vector1` | `[number, number, number]` | * | First vector |
| `vector2` | `[number, number, number]` | * | Second vector |
| `point1` | `[number, number, number]` | * | First point |
| `vertex` | `[number, number, number]` | * | Vertex point (angle measured here) |
| `point2` | `[number, number, number]` | * | Second point |

*Either (vector1, vector2) or (point1, vertex, point2) are required.*

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "angle",
  "value": 90.0,
  "unit": "deg",
  "label": "Angle: 90.00°",
  "display_value": "90.00°",
  "angle": 90.0,
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

---

### Coordinate Information

#### `POST /measure/coordinate`

Get formatted coordinate information for a point.

**Request:**
```json
{
  "point": [100.5, 200.25, 50.125],
  "unit_system": "metric"
}
```

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "measurement_type": "coordinate",
  "x": 100.5,
  "y": 200.25,
  "z": 50.125,
  "unit": "mm",
  "label": "Coordinate",
  "display_value": "X: 100.500 mm, Y: 200.250 mm, Z: 50.125 mm",
  "point1": [100.5, 200.25, 50.125]
}
```

---

### Geometry Analysis

#### `POST /analyze-geometry`

Analyze a STEP file and extract all faces and edges for measurements.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | STEP file (.step or .stp) |

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "file_hash": "sha256...",
  "face_count": 24,
  "edge_count": 48,
  "faces": [
    {
      "face_id": 0,
      "surface_type": "plane",
      "center": [50, 50, 0],
      "normal": [0, 0, 1],
      "area": 10000.0
    }
  ],
  "edges": [
    {
      "edge_id": 0,
      "edge_type": "line",
      "start_point": [0, 0, 0],
      "end_point": [100, 0, 0],
      "length": 100.0
    }
  ],
  "processing_time_ms": 150
}
```

---

### Unit Conversion

#### `POST /convert-units`

Convert measurement values between unit systems.

**Request:**
```json
{
  "value": 25.4,
  "from_unit": "mm",
  "to_unit": "in",
  "decimals": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `value` | `number` | Yes | Value to convert |
| `from_unit` | `"mm" \| "in" \| "mm2" \| "in2" \| "deg"` | Yes | Source unit |
| `to_unit` | `"mm" \| "in" \| "mm2" \| "in2" \| "deg"` | Yes | Target unit |
| `decimals` | `number` | No | Decimal places (default: 3) |

**Response:**
```json
{
  "success": true,
  "correlation_id": "20251214-103000-abc123",
  "original_value": 25.4,
  "original_unit": "mm",
  "converted_value": 1.0,
  "converted_unit": "in",
  "display_value": "1.000 in"
}
```

**Supported Conversions:**
- `mm` ↔ `in` (millimeters ↔ inches)
- `mm2` ↔ `in2` (square millimeters ↔ square inches)
- `deg` (no conversion needed)

---

### Unified Mesh Measurement

#### `POST /measure/from-mesh`

Unified endpoint for measurements using existing mesh data format.

**Request:**
```json
{
  "measurement_type": "point_to_point",
  "data": {
    "point1": [0, 0, 0],
    "point2": [10, 20, 30]
  },
  "unit_system": "metric"
}
```

| `measurement_type` | `data` fields |
|-------------------|---------------|
| `point_to_point` | `point1`, `point2` |
| `edge` | `edge_type`, `start`, `end`, `diameter`, `radius`, `length`, `center` |
| `face_area` | `face_id`, `area`, `center`, `normal`, `surface_type` |
| `face_to_face` | `face1: {center, normal}`, `face2: {center, normal}` |

**Response:** Same as the specific measurement endpoints.

---

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "success": false,
  "error": "Error message describing the problem",
  "correlation_id": "20251214-103000-abc123"
}
```

**HTTP Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (e.g., cached geometry not found) |
| 500 | Internal Server Error |

---

## Request Headers

| Header | Description | Required |
|--------|-------------|----------|
| `Content-Type` | `application/json` (for JSON requests) | Yes* |
| `X-Correlation-ID` | Request tracking ID | No |

*Not required for multipart/form-data requests.

---

## Rate Limiting

Currently no rate limiting is implemented. For production:
- Consider adding rate limiting (e.g., 100 requests/minute)
- Add request queuing for heavy geometric calculations

---

## Examples

### cURL Examples

**Point-to-Point:**
```bash
curl -X POST http://localhost:8081/measure/point-to-point \
  -H "Content-Type: application/json" \
  -d '{"point1": [0, 0, 0], "point2": [10, 20, 30]}'
```

**Edge Measurement:**
```bash
curl -X POST http://localhost:8081/measure/edge \
  -H "Content-Type: application/json" \
  -d '{"edge_info": {"edge_type": "circle", "start_point": [10, 0, 0], "end_point": [10, 0, 0], "diameter": 20}}'
```

**Analyze STEP File:**
```bash
curl -X POST http://localhost:8081/analyze-geometry \
  -F "file=@part.step"
```

### JavaScript/TypeScript Examples

```typescript
import { measurementApi } from '@/services/measurementApiService';

// Point-to-point
const result = await measurementApi.measurePointToPoint(
  [0, 0, 0],
  [10, 20, 30],
  'metric'
);
console.log(result.display_value); // "37.417 mm"

// Edge measurement
const edgeResult = await measurementApi.measureEdge({
  edge_type: 'circle',
  start: [10, 0, 0],
  end: [10, 0, 0],
  diameter: 20,
});
console.log(edgeResult.display_value); // "Ø 20.000 mm"
```

---

## Changelog

### v1.0.0 (2025-12-14)
- Initial release
- Point-to-point distance measurement
- Edge measurements (length, diameter, radius)
- Face area calculation
- Face-to-face distance and angle
- Unit conversion (metric/imperial)
- STEP file geometry analysis
