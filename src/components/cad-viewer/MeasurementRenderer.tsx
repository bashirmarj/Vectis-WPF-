import React from "react";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import type { Measurement } from "@/stores/measurementStore";

/**
 * MeasurementRenderer Component
 * Renders measurement annotations in 3D space with dimension lines and labels
 * Similar to SolidWorks/Fusion 360 measurement display
 */
export const MeasurementRenderer: React.FC = () => {
  const { measurements } = useMeasurementStore();

  const renderMeasurement = (measurement: Measurement) => {
    if (!measurement.visible || measurement.points.length === 0) return null;

    switch (measurement.type) {
      case "distance":
        return renderDistanceMeasurement(measurement);
      case "angle":
        return renderAngleMeasurement(measurement);
      case "radius":
      case "diameter":
        return renderRadiusDiameterMeasurement(measurement);
      case "edge-select":
        return renderEdgeSelectMeasurement(measurement);
      case "edge-to-edge":
        return renderEdgeToEdgeMeasurement(measurement);
      case "face-to-face":
        return renderFaceToFaceMeasurement(measurement);
      case "coordinate":
        return renderCoordinateMeasurement(measurement);
      default:
        return null;
    }
  };

  // ========================================
  // DISTANCE MEASUREMENT (2 points)
  // ========================================
  const renderDistanceMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;

    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    // Direction vector and perpendicular offset for label
    const direction = new THREE.Vector3().subVectors(p2, p1);
    const length = direction.length();

    // Create perpendicular offset for cleaner label placement
    const offset = new THREE.Vector3(0, 1, 0);
    if (Math.abs(direction.y) > 0.9) {
      offset.set(1, 0, 0);
    }
    const labelPos = midpoint.clone().add(offset.multiplyScalar(5));

    // Create quaternion for line rotation
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());

    return (
      <group key={m.id}>
        {/* Dimension Line */}
        <mesh position={midpoint} quaternion={quaternion}>
          <cylinderGeometry args={[0.3, 0.3, length, 8]} />
          <meshBasicMaterial color="#0066FF" transparent opacity={0.8} />
        </mesh>

        {/* Extension Lines */}
        <mesh position={p1}>
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial color="#0066FF" />
        </mesh>
        <mesh position={p2}>
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial color="#0066FF" />
        </mesh>

        {/* Label */}
        <Html position={labelPos} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-blue-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-blue-600">DISTANCE</div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // ANGLE MEASUREMENT (3 points)
  // ========================================
  const renderAngleMeasurement = (m: Measurement) => {
    if (m.points.length < 3) return null;

    const p1 = m.points[0].position;
    const vertex = m.points[1].position;
    const p2 = m.points[2].position;

    // Calculate arc for angle visualization
    const v1 = new THREE.Vector3().subVectors(p1, vertex).normalize();
    const v2 = new THREE.Vector3().subVectors(p2, vertex).normalize();
    const angle = v1.angleTo(v2);

    // Arc radius (visual only)
    const arcRadius = 10;
    const arcMidpoint = vertex.clone().add(v1.clone().add(v2).normalize().multiplyScalar(arcRadius));

    return (
      <group key={m.id}>
        {/* Vertex point */}
        <mesh position={vertex}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshBasicMaterial color="#FF6B00" />
        </mesh>

        {/* Extension lines */}
        {[p1, p2].map((p, i) => {
          const dir = new THREE.Vector3().subVectors(p, vertex);
          const len = Math.min(dir.length(), 15);
          const midpoint = vertex.clone().add(dir.normalize().multiplyScalar(len / 2));
          const quat = new THREE.Quaternion();
          quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());

          return (
            <mesh key={i} position={midpoint} quaternion={quat}>
              <cylinderGeometry args={[0.3, 0.3, len, 8]} />
              <meshBasicMaterial color="#FF6B00" transparent opacity={0.6} />
            </mesh>
          );
        })}

        {/* Label */}
        <Html position={arcMidpoint} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-orange-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-orange-600">ANGLE</div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // RADIUS / DIAMETER MEASUREMENT
  // ========================================
  const renderRadiusDiameterMeasurement = (m: Measurement) => {
    if (m.points.length < 3) return null;

    // Calculate circle center from 3 points
    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const p3 = m.points[2].position;

    // Use first point as reference for label position
    const labelPos = p1.clone();

    return (
      <group key={m.id}>
        {/* Center point */}
        {m.metadata?.arcCenter && (
          <mesh position={m.metadata.arcCenter}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshBasicMaterial color="#9C27B0" />
          </mesh>
        )}

        {/* Measurement points */}
        {[p1, p2, p3].map((p, i) => (
          <mesh key={i} position={p}>
            <sphereGeometry args={[0.8, 16, 16]} />
            <meshBasicMaterial color="#9C27B0" />
          </mesh>
        ))}

        {/* Label */}
        <Html position={labelPos} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-purple-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-purple-600">{m.type.toUpperCase()}</div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // EDGE-SELECT MEASUREMENT (Smart single-click)
  // ========================================
  const renderEdgeSelectMeasurement = (m: Measurement) => {
    if (!m.metadata?.edgeStart || !m.metadata?.edgeEnd) return null;

    const start = new THREE.Vector3(m.metadata.edgeStart[0], m.metadata.edgeStart[1], m.metadata.edgeStart[2]);
    const end = new THREE.Vector3(m.metadata.edgeEnd[0], m.metadata.edgeEnd[1], m.metadata.edgeEnd[2]);
    const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

    // Determine color based on edge type
    const colorMap = {
      line: "#00D084",
      circle: "#FFB84D",
      arc: "#FF6B6B",
    };
    const color = colorMap[m.metadata.edgeType || "line"] || "#00D084";

    // Icon for edge type
    const iconMap = {
      line: "📏",
      circle: "⊙",
      arc: "⌒",
    };
    const icon = iconMap[m.metadata.edgeType || "line"] || "📏";

    return (
      <group key={m.id}>
        {/* Highlight the edge */}
        <mesh position={midpoint}>
          <sphereGeometry args={[2, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>

        {/* Label - Enhanced for edge-select */}
        <Html position={midpoint.clone().add(new THREE.Vector3(0, 5, 0))} center distanceFactor={10}>
          <div
            className="px-3 py-1.5 bg-white border-2 rounded shadow-lg text-center pointer-events-none"
            style={{ borderColor: color }}
          >
            <div className="text-sm font-semibold" style={{ color }}>
              {icon} {m.metadata.edgeType?.toUpperCase()}
            </div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // EDGE-TO-EDGE MEASUREMENT
  // ========================================
  const renderEdgeToEdgeMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;

    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    return (
      <group key={m.id}>
        {/* Points */}
        <mesh position={p1}>
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial color="#00BCD4" />
        </mesh>
        <mesh position={p2}>
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial color="#00BCD4" />
        </mesh>

        {/* Perpendicular distance line */}
        <mesh position={midpoint}>
          <sphereGeometry args={[0.5, 16, 16]} />
          <meshBasicMaterial color="#00BCD4" />
        </mesh>

        {/* Label */}
        <Html position={midpoint.clone().add(new THREE.Vector3(0, 5, 0))} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-cyan-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-cyan-600">EDGE TO EDGE</div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // FACE-TO-FACE MEASUREMENT
  // ========================================
  const renderFaceToFaceMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;

    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    return (
      <group key={m.id}>
        {/* Face points */}
        <mesh position={p1}>
          <sphereGeometry args={[1.2, 16, 16]} />
          <meshBasicMaterial color="#4CAF50" />
        </mesh>
        <mesh position={p2}>
          <sphereGeometry args={[1.2, 16, 16]} />
          <meshBasicMaterial color="#4CAF50" />
        </mesh>

        {/* Label */}
        <Html position={midpoint.clone().add(new THREE.Vector3(0, 5, 0))} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-green-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-green-600">FACE TO FACE</div>
            <div className="text-lg font-bold text-gray-900">{m.label}</div>
            <div className="text-xs text-gray-500">{m.unit}</div>
          </div>
        </Html>
      </group>
    );
  };

  // ========================================
  // COORDINATE MEASUREMENT (Single point XYZ)
  // ========================================
  const renderCoordinateMeasurement = (m: Measurement) => {
    if (m.points.length < 1) return null;

    const p = m.points[0].position;

    return (
      <group key={m.id}>
        {/* Point marker */}
        <mesh position={p}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshBasicMaterial color="#E91E63" />
        </mesh>

        {/* Coordinate axes lines */}
        {[
          { dir: new THREE.Vector3(5, 0, 0), color: "#FF0000" },
          { dir: new THREE.Vector3(0, 5, 0), color: "#00FF00" },
          { dir: new THREE.Vector3(0, 0, 5), color: "#0000FF" },
        ].map((axis, i) => {
          const midpoint = p.clone().add(axis.dir.clone().multiplyScalar(0.5));
          const quat = new THREE.Quaternion();
          quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.dir.clone().normalize());

          return (
            <mesh key={i} position={midpoint} quaternion={quat}>
              <cylinderGeometry args={[0.2, 0.2, 5, 8]} />
              <meshBasicMaterial color={axis.color} transparent opacity={0.4} />
            </mesh>
          );
        })}

        {/* Label with XYZ coordinates */}
        <Html position={p.clone().add(new THREE.Vector3(0, 8, 0))} center distanceFactor={10}>
          <div className="px-3 py-1.5 bg-white border-2 border-pink-500 rounded shadow-lg text-center pointer-events-none">
            <div className="text-xs font-semibold text-pink-600">COORDINATE</div>
            <div className="text-sm font-mono text-gray-900">
              <div>X: {p.x.toFixed(2)}</div>
              <div>Y: {p.y.toFixed(2)}</div>
              <div>Z: {p.z.toFixed(2)}</div>
            </div>
          </div>
        </Html>
      </group>
    );
  };

  return (
    <>
      {measurements.map((measurement) => (
        <React.Fragment key={measurement.id}>{renderMeasurement(measurement)}</React.Fragment>
      ))}
    </>
  );
};
