import React from "react";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasurementStore } from "@/stores/measurementStore";
import type { Measurement } from "@/stores/measurementStore";

export const MeasurementRenderer: React.FC = () => {
  const { measurements } = useMeasurementStore();

  const renderFaceToFaceMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;
    
    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    
    // Use useMemo to compute midpoint only when positions change
    const midpoint = React.useMemo(
      () => new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5),
      [p1, p2]
    );
    
    return (
      <group key={m.id}>
        {/* Declarative line using BufferGeometry */}
        <line>
          <bufferGeometry attach="geometry">
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                p1.x, p1.y, p1.z,
                p2.x, p2.y, p2.z
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial attach="material" color="#00CC66" linewidth={2} />
        </line>
        
        {/* Declarative spheres at endpoints */}
        <mesh position={p1}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshBasicMaterial color="#00CC66" />
        </mesh>
        <mesh position={p2}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshBasicMaterial color="#00CC66" />
        </mesh>
        
        {/* Label at midpoint */}
        <Html position={midpoint} center distanceFactor={2}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            color: '#1a1a1a',
            padding: '12px 20px',
            borderRadius: '6px',
            fontSize: '20px',
            fontWeight: '600',
            border: '2px solid #00CC66',
            boxShadow: '0 3px 10px rgba(0,0,0,0.2)',
            minWidth: '180px',
            textAlign: 'center',
            transform: 'scale(1.3)'
          }}>
            <div style={{fontSize: '32px', fontWeight: 'bold', color: '#00CC66'}}>
              {m.label}
            </div>
          </div>
        </Html>
      </group>
    );
  };

  const renderMeasurement = (measurement: Measurement) => {
    if (!measurement.visible || measurement.points.length === 0) return null;

    if (measurement.type === "face-to-face") {
      return renderFaceToFaceMeasurement(measurement);
    }
    
    return renderEdgeSelectMeasurement(measurement);
  };

  const renderDistanceMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;
    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    // Cylindrical surface arc rendering
    if (m.metadata?.cylindrical && m.metadata.center && m.metadata.radius && m.metadata.axis) {
      const curve = new THREE.CatmullRomCurve3([
        p1,
        new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.33),
        new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.67),
        p2,
      ]);
      const tubeGeometry = new THREE.TubeGeometry(curve, 20, 0.8, 8, false);
      
      return (
        <group key={m.id}>
          <mesh geometry={tubeGeometry}>
            <meshBasicMaterial color="#0066CC" opacity={0.85} transparent />
          </mesh>
          <mesh position={p1}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
          <mesh position={p2}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
          <Html position={midpoint} center distanceFactor={2}>
            <div style={{background: 'rgba(255, 255, 255, 0.95)', color: '#1a1a1a', padding: '12px 20px', borderRadius: '6px', fontSize: '20px', fontWeight: '600', border: '2px solid #0066CC', boxShadow: '0 3px 10px rgba(0,0,0,0.2)', minWidth: '180px', textAlign: 'center', transform: 'scale(1.3)'}}>
              <div style={{fontSize: '32px', fontWeight: 'bold', color: '#0066CC'}}>{m.label}</div>
              <div style={{fontSize: '16px', color: '#666', marginTop: '4px'}}>Arc Surface</div>
            </div>
          </Html>
        </group>
      );
    }

    return (
      <group key={m.id}>
        <mesh position={p1}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <mesh position={p2}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <Html position={midpoint} center distanceFactor={2}>
          <div style={{background: 'rgba(255, 255, 255, 0.95)', color: '#1a1a1a', padding: '12px 20px', borderRadius: '6px', fontSize: '20px', fontWeight: '600', border: '2px solid #0066CC', boxShadow: '0 3px 10px rgba(0,0,0,0.2)', minWidth: '180px', textAlign: 'center', transform: 'scale(1.3)'}}>
            <div style={{fontSize: '32px', fontWeight: 'bold', color: '#0066CC'}}>{m.label}</div>
          </div>
        </Html>
      </group>
    );
  };

  const renderEdgeSelectMeasurement = (m: Measurement) => {
    if (m.points.length === 0) return null;
    const point = m.points[0].position;

    const faceOverlay = m.metadata?.faceVertices ? (
      <mesh>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={m.metadata.faceVertices.length / 3} array={new Float32Array(m.metadata.faceVertices)} itemSize={3} />
        </bufferGeometry>
        <meshBasicMaterial color="#0066CC" opacity={0.25} transparent side={THREE.DoubleSide} depthTest={false} />
      </mesh>
    ) : null;

    return (
      <group key={m.id}>
        {faceOverlay}
        <mesh position={point}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <Html position={point} center distanceFactor={2}>
          <div style={{background: 'rgba(255, 255, 255, 0.95)', color: '#1a1a1a', padding: '12px 20px', borderRadius: '6px', fontSize: '20px', fontWeight: '600', border: '2px solid #0066CC', boxShadow: '0 3px 10px rgba(0,0,0,0.2)', minWidth: '180px', textAlign: 'center', transform: 'scale(1.3)'}}>
            <div style={{fontSize: '32px', fontWeight: 'bold', color: '#0066CC'}}>{m.label}</div>
          </div>
        </Html>
      </group>
    );
  };

  const renderEdgeToEdgeMeasurement = (m: Measurement) => {
    if (m.points.length < 2) return null;
    const p1 = m.points[0].position;
    const p2 = m.points[1].position;
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

    return (
      <group key={m.id}>
        <mesh position={p1}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <mesh position={p2}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <Html position={midpoint} center distanceFactor={2}>
          <div style={{background: 'rgba(255, 255, 255, 0.95)', color: '#1a1a1a', padding: '12px 20px', borderRadius: '6px', fontSize: '20px', fontWeight: '600', border: '2px solid #0066CC', boxShadow: '0 3px 10px rgba(0,0,0,0.2)', minWidth: '180px', textAlign: 'center', transform: 'scale(1.3)'}}>
            <div style={{fontSize: '36px', fontWeight: 'bold', color: '#0066CC', marginBottom: '6px'}}>{m.label}</div>
            {m.metadata?.deltaX !== undefined && (
              <div style={{fontSize: '16px', color: '#666', lineHeight: '1.5'}}>
                ΔX: {m.metadata.deltaX.toFixed(2)} mm<br />
                ΔY: {m.metadata.deltaY?.toFixed(2)} mm<br />
                ΔZ: {m.metadata.deltaZ?.toFixed(2)} mm
              </div>
            )}
          </div>
        </Html>
      </group>
    );
  };

  const renderDefaultMeasurement = (m: Measurement) => {
    const point = m.points[0]?.position || new THREE.Vector3();
    return (
      <group key={m.id}>
        <mesh position={point}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#0066CC" /></mesh>
        <Html position={point} center distanceFactor={2}>
          <div style={{background: 'rgba(255, 255, 255, 0.95)', color: '#1a1a1a', padding: '12px 20px', borderRadius: '6px', fontSize: '20px', fontWeight: '600', border: '2px solid #0066CC', boxShadow: '0 3px 10px rgba(0,0,0,0.2)', minWidth: '180px', textAlign: 'center', transform: 'scale(1.3)'}}>
            <div style={{fontSize: '32px', fontWeight: 'bold', color: '#0066CC'}}>{m.label}</div>
          </div>
        </Html>
      </group>
    );
  };

  return <>{measurements.map((measurement) => <React.Fragment key={measurement.id}>{renderMeasurement(measurement)}</React.Fragment>)}</>;
};
