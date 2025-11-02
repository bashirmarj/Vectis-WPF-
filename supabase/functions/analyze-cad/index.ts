// deno-lint-ignore-file no-explicit-any
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.38.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-test-flask",
};

interface AnalysisRequest {
  file_name: string;
  file_size: number;
  quantity: number;
}

interface DetectedFeatures {
  is_cylindrical: boolean;
  has_keyway: boolean;
  has_flat_surfaces: boolean;
  has_internal_holes: boolean;
  requires_precision_boring: boolean;
  cylindricity_score: number;
  flat_surface_percentage: number;
  internal_surface_percentage: number;
}

interface DetectedHole {
  type: "hole";
  diameter_mm: number;
  depth_mm: number;
  through: boolean;
  position: [number, number, number];
  axis: [number, number, number];
  orientation: string;
}

interface DetectedGroove {
  type: "groove";
  inner_diameter_mm: number;
  outer_diameter_mm: number;
  depth_mm: number;
  location: "external" | "internal";
  orientation: string;
}

interface DetectedFlat {
  type: "flat";
  orientation: string;
  area_mm2: number;
  width_mm: number;
  length_mm: number;
}

interface PrimaryDimensions {
  major_diameter_mm?: number;
  minor_diameter_mm?: number;
  length_mm: number;
  width_mm?: number;
  height_mm?: number;
  primary_axis: "X" | "Y" | "Z";
}

interface DetailedFeatures {
  holes: DetectedHole[];
  grooves: DetectedGroove[];
  flat_surfaces: DetectedFlat[];
  primary_dimensions: PrimaryDimensions;
}

interface OrientedSection {
  orientation: string;
  features: Array<DetectedHole | DetectedGroove | DetectedFlat>;
}

interface FeatureTree {
  common_dimensions: Array<{
    label: string;
    value: number;
    unit: string;
  }>;
  oriented_sections: OrientedSection[];
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
  vertex_colors?: string[];
  triangle_count: number;
  feature_edges?: number[][][];
  edge_classifications?: any[]; // Backend ground truth edge data
  tagged_feature_edges?: any[]; // Direct feature_id lookup
  vertex_face_ids?: number[];
  face_classifications?: Array<{
    face_id: number;
    type: "internal" | "external" | "through" | "planar";
    center: [number, number, number];
    normal: [number, number, number];
    area: number;
    surface_type: "cylinder" | "plane" | "other";
    radius?: number;
  }>;
}

interface MachiningOperation {
  routing: string;
  machining_time_min: number;
  machining_cost: number;
}

interface AnalysisResult {
  volume_cm3: number;
  surface_area_cm2: number;
  complexity_score: number;
  confidence: number;
  method: string;
  part_width_cm?: number;
  part_height_cm?: number;
  part_depth_cm?: number;
  triangle_count?: number;
  feature_count?: number;
  detected_features?: DetectedFeatures;
  recommended_processes?: string[];
  detailed_features?: DetailedFeatures;
  feature_tree?: FeatureTree;
  mesh_id?: string;
  mesh_data?: MeshData; // ✅ Add mesh_data to interface
  // Industrial routing enhancements
  recommended_routings?: string[];
  routing_reasoning?: string[];
  machining_summary?: MachiningOperation[];
  estimated_total_cost_usd?: number;
}

interface Vector3 {
  x: number;
  y: number;
  z: number;
}

interface Triangle {
  normal: Vector3;
  vertices: [Vector3, Vector3, Vector3];
}

// Parse binary STL file
function parseSTLBinary(buffer: ArrayBuffer): Triangle[] {
  const view = new DataView(buffer);
  let offset = 80; // Skip 80-byte header

  const triangleCount = view.getUint32(offset, true);
  offset += 4;

  const triangles: Triangle[] = [];

  for (let i = 0; i < triangleCount; i++) {
    const normal: Vector3 = {
      x: view.getFloat32(offset, true),
      y: view.getFloat32(offset + 4, true),
      z: view.getFloat32(offset + 8, true),
    };
    offset += 12;

    const vertices: [Vector3, Vector3, Vector3] = [
      {
        x: view.getFloat32(offset, true),
        y: view.getFloat32(offset + 4, true),
        z: view.getFloat32(offset + 8, true),
      },
      {
        x: view.getFloat32(offset + 12, true),
        y: view.getFloat32(offset + 16, true),
        z: view.getFloat32(offset + 20, true),
      },
      {
        x: view.getFloat32(offset + 24, true),
        y: view.getFloat32(offset + 28, true),
        z: view.getFloat32(offset + 32, true),
      },
    ];
    offset += 36;
    offset += 2; // Skip attribute byte count

    triangles.push({ normal, vertices });
  }

  return triangles;
}

// Calculate actual volume using signed tetrahedron method
function calculateVolume(triangles: Triangle[]): number {
  let volume = 0;

  for (const triangle of triangles) {
    const [v1, v2, v3] = triangle.vertices;
    // Volume of tetrahedron from origin to triangle
    volume +=
      (v1.x * v2.y * v3.z -
        v1.x * v3.y * v2.z -
        v2.x * v1.y * v3.z +
        v2.x * v3.y * v1.z +
        v3.x * v1.y * v2.z -
        v3.x * v2.y * v1.z) /
      6;
  }

  return Math.abs(volume) / 1000; // Convert mm³ to cm³
}

// Calculate surface area from triangles
function calculateSurfaceArea(triangles: Triangle[]): number {
  let area = 0;

  for (const triangle of triangles) {
    const [v1, v2, v3] = triangle.vertices;
    const edge1 = {
      x: v2.x - v1.x,
      y: v2.y - v1.y,
      z: v2.z - v1.z,
    };
    const edge2 = {
      x: v3.x - v1.x,
      y: v3.y - v1.y,
      z: v3.z - v1.z,
    };

    // Cross product
    const cross = {
      x: edge1.y * edge2.z - edge1.z * edge2.y,
      y: edge1.z * edge2.x - edge1.x * edge2.z,
      z: edge1.x * edge2.y - edge1.y * edge2.x,
    };

    // Triangle area is half the magnitude of cross product
    area += Math.sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z) / 2;
  }

  return area / 100; // Convert mm² to cm²
}

// Calculate bounding box
function getBoundingBox(triangles: Triangle[]): { min: Vector3; max: Vector3 } {
  const min: Vector3 = { x: Infinity, y: Infinity, z: Infinity };
  const max: Vector3 = { x: -Infinity, y: -Infinity, z: -Infinity };

  for (const triangle of triangles) {
    for (const vertex of triangle.vertices) {
      min.x = Math.min(min.x, vertex.x);
      min.y = Math.min(min.y, vertex.y);
      min.z = Math.min(min.z, vertex.z);
      max.x = Math.max(max.x, vertex.x);
      max.y = Math.max(max.y, vertex.y);
      max.z = Math.max(max.z, vertex.z);
    }
  }

  return { min, max };
}

// Calculate complexity from geometry
function calculateComplexity(triangles: Triangle[], bbox: { min: Vector3; max: Vector3 }): number {
  const triangleCount = triangles.length;

  // Base complexity from triangle count
  let complexity = Math.min(Math.log10(triangleCount) * 2, 10);

  // Analyze surface normal variation (more varied = more complex)
  const normalVariations = new Set<string>();
  for (const triangle of triangles) {
    const normalKey = `${triangle.normal.x.toFixed(1)},${triangle.normal.y.toFixed(1)},${triangle.normal.z.toFixed(1)}`;
    normalVariations.add(normalKey);
  }
  const normalComplexity = Math.min(normalVariations.size / 100, 3);

  // Analyze aspect ratio (tall/thin parts are more complex)
  const width = bbox.max.x - bbox.min.x;
  const height = bbox.max.y - bbox.min.y;
  const depth = bbox.max.z - bbox.min.z;
  const aspectRatio = Math.max(width, height, depth) / Math.min(width, height, depth);
  const aspectComplexity = Math.min(aspectRatio / 5, 2);

  return Math.min(Math.round(complexity + normalComplexity + aspectComplexity), 10);
}

// Detect if part is cylindrical (suitable for lathe)
function isCylindricalPart(
  triangles: Triangle[],
  bbox: { min: Vector3; max: Vector3 },
): { is_cylindrical: boolean; score: number } {
  const width = bbox.max.x - bbox.min.x;
  const height = bbox.max.y - bbox.min.y;
  const depth = bbox.max.z - bbox.min.z;

  // Check if two dimensions are similar (cylindrical cross-section)
  const aspectRatioXY = Math.max(width, height) / Math.min(width, height);
  const aspectRatioXZ = Math.max(width, depth) / Math.min(width, depth);
  const aspectRatioYZ = Math.max(height, depth) / Math.min(height, depth);

  // Find the best cylindrical match
  const cylindricalScore = Math.min(aspectRatioXY, aspectRatioXZ, aspectRatioYZ);

  // Check normal distribution - cylindrical parts have many radial normals
  let radialNormals = 0;
  for (const tri of triangles) {
    const n = tri.normal;
    // Radial normals have strong X or Y component, weak Z (for vertical cylinder)
    const radialStrength = Math.sqrt(n.x * n.x + n.y * n.y);
    if (radialStrength > 0.7) radialNormals++;
  }
  const radialPercentage = radialNormals / triangles.length;

  // Cylindrical if aspect ratio is good AND has radial normals
  const is_cylindrical = cylindricalScore < 1.3 && radialPercentage > 0.4;

  return { is_cylindrical, score: is_cylindrical ? 1 - (cylindricalScore - 1) : 0 };
}

// Detect keyway or slot features
function hasKeyway(triangles: Triangle[]): boolean {
  let horizontalSurfaces = 0;
  let verticalWalls = 0;
  let totalArea = 0;

  for (const tri of triangles) {
    const n = tri.normal;
    const area = getTriangleArea(tri);
    totalArea += area;

    // Horizontal surface (normal pointing up/down)
    if (Math.abs(n.z) > 0.9) horizontalSurfaces += area;

    // Vertical wall (normal pointing sideways)
    if (Math.abs(n.x) > 0.9 || Math.abs(n.y) > 0.9) verticalWalls += area;
  }

  // Keyways have sharp transitions: significant vertical walls + horizontal floors
  const wallPercentage = verticalWalls / totalArea;
  const floorPercentage = horizontalSurfaces / totalArea;

  // Keyway detection: at least 15% vertical walls and some horizontal floors
  return wallPercentage > 0.15 && floorPercentage > 0.05;
}

// Detect flat surfaces requiring milling
function hasFlatSurfaces(triangles: Triangle[]): { has_flat: boolean; percentage: number } {
  let flatArea = 0;
  let totalArea = 0;

  for (const tri of triangles) {
    const n = tri.normal;
    const area = getTriangleArea(tri);
    totalArea += area;

    // Flat surfaces have normals aligned with major axes
    if (Math.abs(n.x) > 0.95 || Math.abs(n.y) > 0.95 || Math.abs(n.z) > 0.95) {
      flatArea += area;
    }
  }

  const percentage = flatArea / totalArea;
  // If >20% of surface is flat, it likely needs milling
  return { has_flat: percentage > 0.2, percentage };
}

// Detect internal holes requiring boring
function hasInternalHoles(
  triangles: Triangle[],
  bbox: { min: Vector3; max: Vector3 },
): { has_holes: boolean; percentage: number } {
  // Calculate part centroid
  const centroid = {
    x: (bbox.min.x + bbox.max.x) / 2,
    y: (bbox.min.y + bbox.max.y) / 2,
    z: (bbox.min.z + bbox.max.z) / 2,
  };

  let inwardArea = 0;
  let totalArea = 0;

  for (const tri of triangles) {
    const area = getTriangleArea(tri);
    totalArea += area;

    // Get triangle centroid
    const triCentroid = {
      x: (tri.vertices[0].x + tri.vertices[1].x + tri.vertices[2].x) / 3,
      y: (tri.vertices[0].y + tri.vertices[1].y + tri.vertices[2].y) / 3,
      z: (tri.vertices[0].z + tri.vertices[1].z + tri.vertices[2].z) / 3,
    };

    // Vector from tri to part center
    const toCenter = {
      x: centroid.x - triCentroid.x,
      y: centroid.y - triCentroid.y,
      z: centroid.z - triCentroid.z,
    };

    // Dot product: if normal points toward center, it's an internal surface
    const dotProduct = tri.normal.x * toCenter.x + tri.normal.y * toCenter.y + tri.normal.z * toCenter.z;
    if (dotProduct > 0) inwardArea += area;
  }

  const percentage = inwardArea / totalArea;
  // If >25% of surfaces point inward, likely has internal holes
  return { has_holes: percentage > 0.25, percentage };
}

// Helper: Calculate triangle area
function getTriangleArea(tri: Triangle): number {
  const [v1, v2, v3] = tri.vertices;
  const edge1 = { x: v2.x - v1.x, y: v2.y - v1.y, z: v2.z - v1.z };
  const edge2 = { x: v3.x - v1.x, y: v3.y - v1.y, z: v3.z - v1.z };

  const cross = {
    x: edge1.y * edge2.z - edge1.z * edge2.y,
    y: edge1.z * edge2.x - edge1.x * edge2.z,
    z: edge1.x * edge2.y - edge1.y * edge2.x,
  };

  return Math.sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z) / 2;
}

// Determine required processes based on detected features
function detectRequiredProcesses(features: DetectedFeatures, complexity: number): string[] {
  const processes: string[] = [];

  if (features.is_cylindrical) {
    // Always add CNC Lathe for cylindrical parts (OD machining)
    processes.push("CNC Lathe");

    // Default to adding VMC for most cylindrical parts
    // Only skip VMC for very simple turned parts (low complexity, no features)
    const isSimpleTurnedPart =
      complexity <= 3 &&
      !features.has_keyway &&
      !features.has_flat_surfaces &&
      !features.has_internal_holes &&
      !features.requires_precision_boring;

    if (!isSimpleTurnedPart) {
      // Most cylindrical parts need VMC for additional operations:
      // - End face machining, drilling, keyways, flats, threads, etc.
      processes.push("VMC Machining");
    }
  } else {
    // Prismatic parts use VMC
    processes.push("VMC Machining");
  }

  // Add keyway process if detected
  if (features.has_keyway && !processes.includes("Key way")) {
    processes.push("Key way");
  }

  // Ensure we always return at least one process
  return processes.length > 0 ? processes : ["VMC Machining"];
}

// Test Flask backend connectivity
async function testFlaskConnection(): Promise<{ success: boolean; error?: string; latency?: number }> {
  const GEOMETRY_SERVICE_URL = Deno.env.get("GEOMETRY_SERVICE_URL");

  if (!GEOMETRY_SERVICE_URL) {
    console.error("❌ GEOMETRY_SERVICE_URL not configured");
    return {
      success: false,
      error: "GEOMETRY_SERVICE_URL environment variable not set",
    };
  }

  const startTime = Date.now();

  try {
    console.log(`🔍 Testing Flask connection: ${GEOMETRY_SERVICE_URL}/health`);

    const response = await fetch(`${GEOMETRY_SERVICE_URL}/health`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(65000), // 65 seconds for Render cold starts
    });

    const latency = Date.now() - startTime;

    if (!response.ok) {
      console.error(`❌ Flask connection failed: HTTP ${response.status}`);
      return {
        success: false,
        error: `HTTP ${response.status}: ${response.statusText}`,
        latency,
      };
    }

    const data = await response.json();

    if (data.status !== "ok") {
      console.error(`❌ Flask health check failed: unexpected response`, data);
      return {
        success: false,
        error: `Unexpected health check response: ${JSON.stringify(data)}`,
        latency,
      };
    }

    console.log(`✅ Flask connection verified successfully (${latency}ms)`);
    return { success: true, latency };
  } catch (error: any) {
    const latency = Date.now() - startTime;
    console.error(`❌ Flask connection failed:`, error.message);

    let errorMessage = error.message;
    if (error.name === "AbortError") {
      errorMessage = "Connection timeout (>10s)";
    } else if (error.message.includes("fetch")) {
      errorMessage = "Network error - unable to reach Flask backend";
    }

    return {
      success: false,
      error: errorMessage,
      latency,
    };
  }
}

// STEP/IGES Analysis via External Python Microservice
async function analyzeSTEPViaService(
  fileData: ArrayBuffer,
  fileName: string,
  material?: string,
  tolerance?: number,
): Promise<AnalysisResult | null> {
  const GEOMETRY_SERVICE_URL = Deno.env.get("GEOMETRY_SERVICE_URL");

  console.log(`🔍 GEOMETRY_SERVICE_URL configured: ${GEOMETRY_SERVICE_URL ? "YES - " + GEOMETRY_SERVICE_URL : "NO"}`);

  if (!GEOMETRY_SERVICE_URL) {
    console.error("❌ GEOMETRY_SERVICE_URL not configured - cannot parse STEP/IGES geometry");
    return null;
  }

  try {
    console.log(`📞 Calling geometry service at ${GEOMETRY_SERVICE_URL}/analyze-cad for ${fileName}`);
    console.log(`📊 Connection details:`, {
      url: `${GEOMETRY_SERVICE_URL}/analyze-cad`,
      fileName: fileName,
      fileSize: `${(fileData.byteLength / 1024).toFixed(2)} KB`,
      material: material || "Cold Rolled Steel",
      tolerance: tolerance || 0.02,
    });

    const formData = new FormData();
    formData.append("file", new Blob([fileData]), fileName);
    formData.append("material", material || "Cold Rolled Steel");
    formData.append("tolerance", (tolerance || 0.02).toString());
    // Quality and sample_density removed - backend uses internal adaptive preset

    // Retry logic for Render.com cold starts (free tier spins down after 15min inactivity)
    let attempts = 0;
    const maxAttempts = 3;
    let lastError: any = null;
    let response: Response | null = null;

    while (attempts < maxAttempts) {
      try {
        attempts++;
        console.log(`🔄 Attempt ${attempts}/${maxAttempts} calling Flask backend...`);

        response = await fetch(`${GEOMETRY_SERVICE_URL}/analyze-cad`, {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
          signal: AbortSignal.timeout(120000), // 120 seconds per attempt for cold starts + processing
        });

        console.log(`📡 Flask response received: HTTP ${response.status} (${response.headers.get("content-type")})`);

        if (!response.ok) {
          const errorText = await response.text();
          console.error(`❌ Geometry service returned ${response.status}: ${errorText}`);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        // Success - break out of retry loop
        break;
      } catch (error: any) {
        lastError = error;
        console.error(`❌ Attempt ${attempts} failed:`, error.message);

        if (attempts < maxAttempts) {
          const backoffMs = Math.pow(2, attempts) * 5000; // 10s, 20s backoff
          console.log(`⏳ Retrying in ${backoffMs / 1000}s...`);
          await new Promise((resolve) => setTimeout(resolve, backoffMs));
        } else {
          console.error(`❌ All ${maxAttempts} attempts failed. Last error:`, lastError.message);
          return null;
        }
      }
    }

    if (!response || !response.ok) {
      console.error(`❌ Failed after ${maxAttempts} attempts`);
      return null;
    }

    const data = await response.json();
    console.log(`✅ Flask BREP analysis complete:`, {
      exact_volume: data.exact_volume,
      exact_area: data.exact_surface_area,
      triangles: data.mesh_data?.triangle_count,
      edges: data.mesh_data?.feature_edges?.length,
      features: {
        holes: data.manufacturing_features?.holes?.length || 0,
        bosses: data.manufacturing_features?.cylindrical_bosses?.length || 0,
        planar: data.manufacturing_features?.planar_faces?.length || 0,
      },
    });

    // Map service response to our result format with BREP data
    const detected_features: DetectedFeatures = {
      is_cylindrical:
        data.manufacturing_features?.holes?.length > 0 || data.manufacturing_features?.cylindrical_bosses?.length > 0,
      has_keyway: false,
      has_flat_surfaces: data.manufacturing_features?.planar_faces?.length > 0,
      has_internal_holes: data.manufacturing_features?.holes?.length > 0,
      requires_precision_boring: data.manufacturing_features?.holes?.length > 0,
      cylindricity_score: data.is_cylindrical ? 0.9 : 0.1,
      flat_surface_percentage: data.total_faces > 0 ? data.planar_faces / data.total_faces : 0,
      internal_surface_percentage: 0,
    };

    const recommended_processes = detectRequiredProcesses(detected_features, data.complexity_score);

    console.log(`Recommended processes: ${JSON.stringify(recommended_processes)}`);

    // Build detailed features structure from BREP manufacturing features
    const detailed_features: DetailedFeatures | undefined = data.manufacturing_features
      ? {
          holes: (data.manufacturing_features.holes || []).map((h: any) => ({
            type: "hole" as const,
            diameter_mm: h.diameter,
            depth_mm: 0,
            through: true,
            position: h.position,
            axis: h.axis,
            orientation: "Z",
          })),
          grooves: [],
          flat_surfaces: (data.manufacturing_features.planar_faces || []).map((f: any) => ({
            type: "flat" as const,
            orientation: "Z",
            area_mm2: f.area,
            width_mm: 0,
            length_mm: 0,
          })),
          primary_dimensions: {
            length_mm: (data.part_depth_cm || 0) * 10,
            primary_axis: "Z" as "X" | "Y" | "Z",
          },
        }
      : undefined;

    // Build feature tree organized by orientation
    const feature_tree: FeatureTree | undefined = detailed_features ? buildFeatureTree(detailed_features) : undefined;

    // Store mesh data in database if available
    let mesh_id: string | undefined;
    if (data.mesh_data && data.mesh_data.vertices && data.mesh_data.vertices.length > 0) {
      console.log(`💾 Storing mesh data: ${data.mesh_data.triangle_count} triangles`);
      mesh_id = await storeMeshData(data.mesh_data, fileName);
      console.log(`✅ Mesh stored with ID: ${mesh_id}`);
      console.log(`📐 Mesh includes ${(data.mesh_data.feature_edges || []).length} feature edges`);
    } else {
      console.log(`⚠️ No mesh data available to store`);
    }

    return {
      // Use exact BREP-based calculations for quotation
      volume_cm3: (data.exact_volume || data.volume_cm3) / 1000, // mm³ to cm³
      surface_area_cm2: (data.exact_surface_area || data.surface_area_cm2) / 100, // mm² to cm²
      complexity_score: data.complexity_score,
      confidence: 0.98, // Higher confidence with BREP analysis
      method: "brep_dual_representation",
      part_width_cm: data.part_width_cm,
      part_height_cm: data.part_height_cm,
      part_depth_cm: data.part_depth_cm,
      detected_features,
      recommended_processes,
      detailed_features,
      mesh_id,
      mesh_data: data.mesh_data, // Display mesh from tessellation
      feature_tree,
      triangle_count: data.mesh_data?.triangle_count,
      // Industrial routing data from geometry service
      recommended_routings: data.recommended_routings,
      routing_reasoning: data.routing_reasoning,
      machining_summary: data.machining_summary,
      estimated_total_cost_usd: data.estimated_total_cost_usd,
    };
  } catch (error) {
    console.error("Error calling geometry service:", error);
    return null;
  }
}

// Caching disabled - every upload generates fresh analysis

// Store mesh data in database - NO CACHING (always fresh)
async function storeMeshData(meshData: MeshData, fileName: string): Promise<string | undefined> {
  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Generate timestamp-based file hash (for schema compliance, not for caching)
    const fileHash = `${fileName}_${Date.now()}_${Math.random().toString(36).substring(7)}`;

    // Store new mesh - NO CACHE LOOKUP
    const { data: newMesh, error } = await supabase
      .from("cad_meshes")
      .insert({
        file_hash: fileHash,
        file_name: fileName,
        vertices: meshData.vertices,
        indices: meshData.indices,
        normals: meshData.normals,
        vertex_colors: meshData.vertex_colors || [],
        triangle_count: meshData.triangle_count,
        feature_edges: Array.isArray(meshData.feature_edges) ? meshData.feature_edges : [],
        edge_classifications: Array.isArray(meshData.edge_classifications) ? meshData.edge_classifications : [],
        tagged_feature_edges: Array.isArray(meshData.tagged_feature_edges) ? meshData.tagged_feature_edges : [],
        vertex_face_ids: meshData.vertex_face_ids || [],
        face_classifications: Array.isArray(meshData.face_classifications) ? meshData.face_classifications : [],
      })
      .select("id")
      .single();

    if (error) {
      console.error("Error storing mesh data:", error);
      return undefined;
    }

    console.log(`Stored mesh data for ${fileName}: ${meshData.triangle_count} triangles, mesh_id: ${newMesh.id}`);
    return newMesh.id;
  } catch (error) {
    console.error("Error in storeMeshData:", error);
    return undefined;
  }
}

function buildFeatureTree(features: DetailedFeatures): FeatureTree {
  const common_dimensions: Array<{ label: string; value: number; unit: string }> = [];

  // Add primary dimensions
  const dims = features.primary_dimensions;
  if (dims.major_diameter_mm) {
    common_dimensions.push({
      label: "Major Diameter",
      value: dims.major_diameter_mm,
      unit: "mm",
    });
  }
  if (dims.length_mm) {
    common_dimensions.push({
      label: "Length",
      value: dims.length_mm,
      unit: "mm",
    });
  }
  if (dims.width_mm) {
    common_dimensions.push({
      label: "Width",
      value: dims.width_mm,
      unit: "mm",
    });
  }
  if (dims.height_mm) {
    common_dimensions.push({
      label: "Height",
      value: dims.height_mm,
      unit: "mm",
    });
  }

  // Group features by orientation
  const orientationMap = new Map<string, Array<DetectedHole | DetectedGroove | DetectedFlat>>();

  features.holes.forEach((hole) => {
    const arr = orientationMap.get(hole.orientation) || [];
    arr.push(hole);
    orientationMap.set(hole.orientation, arr);
  });

  features.grooves.forEach((groove) => {
    const arr = orientationMap.get(groove.orientation) || [];
    arr.push(groove);
    orientationMap.set(groove.orientation, arr);
  });

  features.flat_surfaces.forEach((flat) => {
    const arr = orientationMap.get(flat.orientation) || [];
    arr.push(flat);
    orientationMap.set(flat.orientation, arr);
  });

  const oriented_sections: OrientedSection[] = Array.from(orientationMap.entries()).map(([orientation, features]) => ({
    orientation,
    features,
  }));

  return {
    common_dimensions,
    oriented_sections,
  };
}

// Deprecated: STEP/IGES file parsing using occt-import-js (doesn't work in Deno)
async function analyzeSTEP(fileData: ArrayBuffer, fileName: string): Promise<AnalysisResult> {
  console.log("Parsing STEP/IGES geometry with OpenCascade...");

  try {
    // Dynamically import occt-import-js
    const { default: initOpenCascade } = await import("npm:occt-import-js@0.0.12");
    const occt: any = await initOpenCascade();

    // Read the file based on extension
    const lowerName = fileName.toLowerCase();
    const isSTEP = lowerName.endsWith(".step") || lowerName.endsWith(".stp");

    const fileBuffer = new Uint8Array(fileData);
    let result;

    if (isSTEP) {
      result = occt.ReadStepFile(fileBuffer, null);
    } else {
      result = occt.ReadIgesFile(fileBuffer, null);
    }

    if (!result.success) {
      throw new Error("Failed to parse STEP/IGES file");
    }

    // Get bounding box
    const shapes = result.shapes;
    if (!shapes || shapes.length === 0) {
      throw new Error("No shapes found in file");
    }

    // Combine all shapes to get overall bounding box
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;
    let totalVolume = 0;
    let totalSurfaceArea = 0;

    for (const shape of shapes) {
      const bbox = shape.BoundingBox;
      minX = Math.min(minX, bbox.xmin);
      minY = Math.min(minY, bbox.ymin);
      minZ = Math.min(minZ, bbox.zmin);
      maxX = Math.max(maxX, bbox.xmax);
      maxY = Math.max(maxY, bbox.ymax);
      maxZ = Math.max(maxZ, bbox.zmax);

      // Get mesh for volume/surface calculations
      if (shape.mesh) {
        totalVolume += shape.mesh.attributes.volume || 0;
        totalSurfaceArea += shape.mesh.attributes.surfaceArea || 0;
      }
    }

    // Convert from mm to cm
    const part_width_cm = Number(((maxX - minX) / 10).toFixed(2));
    const part_height_cm = Number(((maxY - minY) / 10).toFixed(2));
    const part_depth_cm = Number(((maxZ - minZ) / 10).toFixed(2));
    const volume_cm3 = Number((totalVolume / 1000).toFixed(2)); // mm³ to cm³
    const surface_area_cm2 = Number((totalSurfaceArea / 100).toFixed(2)); // mm² to cm²

    // Detect features from geometry
    const dimensions = [part_width_cm, part_height_cm, part_depth_cm].sort((a, b) => a - b);
    const aspectRatios = [dimensions[1] / dimensions[0], dimensions[2] / dimensions[0], dimensions[2] / dimensions[1]];

    // Cylindrical detection: two similar dimensions, one elongated
    const crossSectionRatio = aspectRatios[0]; // Should be ~1.0 for circular cross-section
    const lengthRatio = aspectRatios[1]; // Should be >1.5 for elongated shape
    const is_cylindrical_by_geometry = crossSectionRatio < 1.3 && lengthRatio > 1.5;
    const is_cylindrical_by_filename =
      lowerName.match(/shaft|pulley|cylinder|rod|tube|bearing|bushing|sleeve|piston/i) !== null;
    const is_cylindrical = is_cylindrical_by_geometry || is_cylindrical_by_filename;

    const detected_features: DetectedFeatures = {
      is_cylindrical,
      has_keyway: lowerName.match(/keyway|key|slot|groove/i) !== null,
      has_flat_surfaces: !is_cylindrical,
      has_internal_holes: lowerName.match(/housing|bearing|bushing/i) !== null,
      requires_precision_boring: false,
      cylindricity_score: is_cylindrical ? 0.9 : 0,
      flat_surface_percentage: is_cylindrical ? 0.1 : 0.5,
      internal_surface_percentage: 0,
    };

    const complexity = Math.min(Math.round(5 + (shapes.length - 1) * 0.5), 10);
    const recommended_processes = detectRequiredProcesses(detected_features, complexity);

    console.log(
      `STEP/IGES Analysis - Volume: ${volume_cm3}cm³, Surface: ${surface_area_cm2}cm², Complexity: ${complexity}/10`,
    );
    console.log(`Dimensions: ${part_width_cm} × ${part_height_cm} × ${part_depth_cm} cm`);
    console.log(`Features:`, detected_features);
    console.log(`Recommended processes:`, recommended_processes);

    return {
      volume_cm3,
      surface_area_cm2,
      complexity_score: complexity,
      confidence: 0.95,
      method: "occt_geometry_parsing",
      part_width_cm,
      part_height_cm,
      part_depth_cm,
      detected_features,
      recommended_processes,
    };
  } catch (error) {
    console.error("STEP/IGES parsing error:", error);
    throw error;
  }
}

// Analyze STL file
async function analyzeSTL(fileData: ArrayBuffer): Promise<AnalysisResult> {
  console.log("Parsing STL geometry...");

  const triangles = parseSTLBinary(fileData);
  const volume = calculateVolume(triangles);
  const surfaceArea = calculateSurfaceArea(triangles);
  const bbox = getBoundingBox(triangles);
  const complexity = calculateComplexity(triangles, bbox);

  const part_width_cm = Number(((bbox.max.x - bbox.min.x) / 10).toFixed(2));
  const part_height_cm = Number(((bbox.max.y - bbox.min.y) / 10).toFixed(2));
  const part_depth_cm = Number(((bbox.max.z - bbox.min.z) / 10).toFixed(2));

  // Detect features
  const cylindrical = isCylindricalPart(triangles, bbox);
  const keyway = hasKeyway(triangles);
  const flatSurfaces = hasFlatSurfaces(triangles);
  const internalHoles = hasInternalHoles(triangles, bbox);

  const detected_features: DetectedFeatures = {
    is_cylindrical: cylindrical.is_cylindrical,
    has_keyway: keyway,
    has_flat_surfaces: flatSurfaces.has_flat,
    has_internal_holes: internalHoles.has_holes,
    requires_precision_boring: internalHoles.has_holes && internalHoles.percentage > 0.35,
    cylindricity_score: cylindrical.score,
    flat_surface_percentage: flatSurfaces.percentage,
    internal_surface_percentage: internalHoles.percentage,
  };

  const recommended_processes = detectRequiredProcesses(detected_features, complexity);

  console.log(
    `STL Analysis - Volume: ${volume.toFixed(2)}cm³, Surface: ${surfaceArea.toFixed(2)}cm², Complexity: ${complexity}/10, Triangles: ${triangles.length}`,
  );
  console.log(`Dimensions: ${part_width_cm} × ${part_height_cm} × ${part_depth_cm} cm`);
  console.log(`Features:`, detected_features);
  console.log(`Recommended processes:`, recommended_processes);

  return {
    volume_cm3: Number(volume.toFixed(2)),
    surface_area_cm2: Number(surfaceArea.toFixed(2)),
    complexity_score: complexity,
    confidence: 0.9,
    method: "stl_geometry",
    part_width_cm,
    part_height_cm,
    part_depth_cm,
    triangle_count: triangles.length,
    detected_features,
    recommended_processes,
  };
}

// Enhanced heuristic analysis for non-STL files
function enhancedHeuristic(fileName: string, fileSize: number): AnalysisResult {
  console.log(`Enhanced heuristic analysis: ${fileName}, size: ${fileSize} bytes`);

  const fileSizeKB = fileSize / 1024;
  const lowerName = fileName.toLowerCase();

  // Better volume estimation based on file type
  let volumeMultiplier = 0.8;
  if (lowerName.endsWith(".step") || lowerName.endsWith(".stp")) {
    volumeMultiplier = 1.2; // STEP files are more verbose
  } else if (lowerName.endsWith(".iges") || lowerName.endsWith(".igs")) {
    volumeMultiplier = 1.0;
  }

  const estimatedVolume = Math.max(fileSizeKB * volumeMultiplier, 10);
  const estimatedSurfaceArea = estimatedVolume * 6;

  // Enhanced complexity scoring
  let complexity = 5;

  // File name patterns
  if (lowerName.match(/simple|basic|bracket|plate|block|washer/i)) complexity -= 2;
  if (lowerName.match(/complex|assembly|multi|intricate|detailed/i)) complexity += 2;

  // File size indicators
  if (fileSizeKB > 5000) complexity += 2;
  else if (fileSizeKB < 100) complexity -= 1;

  // File entropy estimate (larger files relative to volume = more complex)
  const entropyFactor = fileSizeKB / estimatedVolume;
  if (entropyFactor > 5) complexity += 1;

  complexity = Math.max(1, Math.min(10, complexity));

  // Deterministic dimension estimation (no randomness)
  const cubeSide = Math.pow(estimatedVolume, 1 / 3);
  const part_width_cm = Number((cubeSide * 1.0).toFixed(2));
  const part_height_cm = Number((cubeSide * 0.97).toFixed(2));
  const part_depth_cm = Number((cubeSide * 1.12).toFixed(2));

  // Enhanced cylindrical detection with broader filename patterns
  const cylindricalKeywords = /shaft|pulley|cylinder|rod|tube|bearing|bushing|sleeve|piston|pin|bolt|axle|spindle/i;
  const is_cylindrical_by_filename = cylindricalKeywords.test(lowerName);

  // Geometry-based detection (aspect ratios)
  const dimensions = [part_width_cm, part_height_cm, part_depth_cm].sort((a, b) => a - b);
  const aspectRatios = [
    dimensions[1] / dimensions[0], // middle/smallest
    dimensions[2] / dimensions[0], // largest/smallest
    dimensions[2] / dimensions[1], // largest/middle
  ];

  // If two dimensions are similar and one is elongated, likely cylindrical
  const crossSectionRatio = aspectRatios[0]; // Should be ~1.0 for circular cross-section
  const lengthRatio = aspectRatios[1]; // Should be >1.5 for elongated shape
  const is_cylindrical_by_geometry = crossSectionRatio < 1.3 && lengthRatio > 1.5;

  // Accept as cylindrical if EITHER condition is met
  const is_cylindrical = is_cylindrical_by_geometry || is_cylindrical_by_filename;
  const has_keyway = lowerName.match(/keyway|key|slot|groove/i) !== null;
  const has_flat_surfaces = lowerName.match(/bracket|plate|block|housing|mount/i) !== null;

  const detected_features: DetectedFeatures = {
    is_cylindrical,
    has_keyway,
    has_flat_surfaces,
    has_internal_holes: lowerName.match(/housing|bearing|bushing/i) !== null,
    requires_precision_boring: false,
    cylindricity_score: is_cylindrical ? 0.8 : 0,
    flat_surface_percentage: has_flat_surfaces ? 0.4 : 0.1,
    internal_surface_percentage: 0,
  };

  const recommended_processes = detectRequiredProcesses(detected_features, complexity);

  console.log(
    `Enhanced Heuristic - Volume: ${estimatedVolume.toFixed(2)}cm³, Surface: ${estimatedSurfaceArea.toFixed(2)}cm², Complexity: ${complexity}/10`,
  );
  console.log(
    `Cylindrical detection: filename=${is_cylindrical_by_filename}, geometry=${is_cylindrical_by_geometry}, final=${is_cylindrical}`,
  );
  console.log(`Heuristic Features:`, detected_features);
  console.log(`Recommended processes:`, recommended_processes);

  return {
    volume_cm3: Number(estimatedVolume.toFixed(2)),
    surface_area_cm2: Number(estimatedSurfaceArea.toFixed(2)),
    complexity_score: complexity,
    confidence: 0.75,
    method: "enhanced_heuristic",
    part_width_cm,
    part_height_cm,
    part_depth_cm,
    detected_features,
    recommended_processes,
  };
}

const handler = async (req: Request): Promise<Response> => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Check if this is a test-flask request
  const url = new URL(req.url);
  if (url.pathname.endsWith("/test-flask") || req.headers.get("x-test-flask") === "true") {
    console.log("🧪 Flask connection test requested");
    const testResult = await testFlaskConnection();

    return new Response(JSON.stringify(testResult), {
      status: testResult.success ? 200 : 503,
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders,
      },
    });
  }

  try {
    const contentType = req.headers.get("content-type") || "";
    let file_name: string;
    let file_size: number;
    let file_data: ArrayBuffer | null = null;
    let quantity: number = 1;
    let material: string | undefined;
    let tolerance: number | undefined;

    // Handle both JSON (metadata only) and multipart/form-data (with file)
    if (contentType.includes("multipart/form-data")) {
      const formData = await req.formData();
      const file = formData.get("file") as File;
      file_name = (formData.get("file_name") as string) || file?.name || "";
      file_size = file?.size || 0;
      quantity = parseInt(formData.get("quantity") as string) || 1;
      material = (formData.get("material") as string) || undefined;
      const toleranceStr = formData.get("tolerance") as string;
      tolerance = toleranceStr ? parseFloat(toleranceStr) : undefined;

      if (file) {
        file_data = await file.arrayBuffer();
      }
    } else {
      const body = await req.json();
      file_name = body.file_name;
      file_size = body.file_size;
      quantity = body.quantity || 1;
      material = body.material;
      tolerance = body.tolerance;

      // Decode base64 file data if provided
      if (body.file_data) {
        try {
          const base64Data = body.file_data;
          const binaryString = atob(base64Data);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          file_data = bytes.buffer;
          console.log(`✅ Decoded base64 file data: ${(file_data.byteLength / 1024).toFixed(2)} KB`);
        } catch (decodeError) {
          console.error("❌ Failed to decode base64 file data:", decodeError);
        }
      }
    }

    if (!file_name || !file_size) {
      throw new Error("Missing required parameters: file_name and file_size");
    }

    console.log(`CAD Analysis request for: ${file_name} (${file_size} bytes)`);

    let analysis: AnalysisResult;
    const lowerName = file_name.toLowerCase();
    const isSTL = lowerName.endsWith(".stl");
    const isSTEP = lowerName.endsWith(".step") || lowerName.endsWith(".stp");
    const isIGES = lowerName.endsWith(".iges") || lowerName.endsWith(".igs");

    // Use appropriate parser based on file type
    if (file_data && isSTL) {
      try {
        analysis = await analyzeSTL(file_data);
      } catch (error) {
        console.error("STL parsing failed, falling back to heuristic:", error);
        analysis = enhancedHeuristic(file_name, file_size);
      }
    } else if (file_data && (isSTEP || isIGES)) {
      // STEP/IGES: Always call Python microservice for accurate geometry analysis
      console.log(`🔧 Attempting geometry service analysis for: ${file_name}`);
      const serviceResult = await analyzeSTEPViaService(file_data, file_name, material, tolerance);

      if (serviceResult && serviceResult.mesh_id) {
        analysis = serviceResult;
        console.log(`✅ Geometry service analysis successful with mesh_id: ${serviceResult.mesh_id}`);
      } else if (serviceResult) {
        analysis = serviceResult;
        console.log(`⚠️ Geometry service analysis successful but no mesh data`);
      } else {
        console.log("❌ Geometry service unavailable, falling back to heuristic (reduced confidence)");
        analysis = enhancedHeuristic(file_name, file_size);
        analysis.confidence = 0.3; // Mark low confidence for fallback
        analysis.method = "fallback_heuristic";
      }
    } else {
      // For other formats or when no file data, use enhanced heuristic
      console.log(`Using enhanced heuristic for: ${file_name}`);
      analysis = enhancedHeuristic(file_name, file_size);
    }

    return new Response(
      JSON.stringify({
        success: true,
        ...analysis,
        quantity,
        detected_features: analysis.detected_features,
        recommended_processes: analysis.recommended_processes,
        mesh_data: analysis.mesh_data,
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      },
    );
  } catch (error: any) {
    console.error("Error in analyze-cad function:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: error.message,
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      },
    );
  }
};

serve(handler);
