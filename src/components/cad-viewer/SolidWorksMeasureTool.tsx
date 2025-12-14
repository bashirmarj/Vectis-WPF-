/**
 * SolidWorksMeasureTool - Independent SolidWorks-style Measurement Tool
 * 
 * Features:
 * - Edge measurement (length, diameter, radius)
 * - Point-to-point distance
 * - Circle/arc detection
 * - Face-to-face distance
 * 
 * Design: Fresh SolidWorks-inspired UI
 * Units: All values in MILLIMETERS
 */

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useMeasureAPI, TaggedEdge, MeasureResult } from "@/hooks/useMeasureAPI";
import { toast } from "@/hooks/use-toast";

// ============= Types =============

interface MeshData {
    tagged_edges?: TaggedEdge[];
    vertex_face_ids?: number[];
    face_classifications?: any[];
}

interface StoredMeasurement {
    id: string;
    result: MeasureResult;
    position: THREE.Vector3;
    endPosition?: THREE.Vector3;
    color: string;
    visible: boolean;
    createdAt: Date;
}

interface Props {
    meshData: MeshData | null;
    meshRef: THREE.Mesh | null;
    enabled: boolean;
    boundingSphere?: { center: THREE.Vector3; radius: number };
}

// ============= Component =============

export const SolidWorksMeasureTool: React.FC<Props> = ({
    meshData,
    meshRef,
    enabled,
    boundingSphere,
}) => {
    const { camera, gl, raycaster } = useThree();
    const { measureEdge, measurePointToPoint, reset } = useMeasureAPI();

    // Hover state
    const [hoverEdge, setHoverEdge] = useState<TaggedEdge | null>(null);
    const [hoverPosition, setHoverPosition] = useState<THREE.Vector3 | null>(null);
    const [hoverLabel, setHoverLabel] = useState<string>("");

    // Point-to-point state
    const [firstPoint, setFirstPoint] = useState<THREE.Vector3 | null>(null);
    const [mode, setMode] = useState<"edge" | "point">("edge");

    // Stored measurements
    const [measurements, setMeasurements] = useState<StoredMeasurement[]>([]);

    // Drag detection
    const dragStartRef = useRef<{ x: number; y: number } | null>(null);

    // Tagged edges from backend (already in mm)
    const taggedEdges = meshData?.tagged_edges || [];

    // ============= Edge Detection =============

    /**
     * Find closest tagged edge to a point
     * Threshold in mm (coordinates are in mm)
     */
    const findClosestEdge = useCallback((
        point: THREE.Vector3,
        threshold: number = 50
    ): TaggedEdge | null => {
        if (!taggedEdges.length) return null;

        let closestEdge: TaggedEdge | null = null;
        let minDistance = threshold;

        for (const edge of taggedEdges) {
            // Check distance to snap_points if available
            if (edge.snap_points && edge.snap_points.length > 0) {
                for (const sp of edge.snap_points) {
                    const snapPoint = new THREE.Vector3(sp[0], sp[1], sp[2]);
                    const dist = point.distanceTo(snapPoint);
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestEdge = edge;
                    }
                }
            } else {
                // Fallback to start/end
                const start = new THREE.Vector3(...edge.start);
                const end = new THREE.Vector3(...edge.end);
                const line = new THREE.Line3(start, end);
                const closest = new THREE.Vector3();
                line.closestPointToPoint(point, true, closest);
                const dist = point.distanceTo(closest);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestEdge = edge;
                }
            }
        }

        return closestEdge;
    }, [taggedEdges]);

    // ============= Pointer Move Handler =============

    useEffect(() => {
        if (!enabled || !meshRef) {
            setHoverEdge(null);
            setHoverPosition(null);
            setHoverLabel("");
            return;
        }

        const handlePointerMove = (event: PointerEvent) => {
            const canvas = gl.domElement;
            const rect = canvas.getBoundingClientRect();
            const mouse = new THREE.Vector2(
                ((event.clientX - rect.left) / rect.width) * 2 - 1,
                -((event.clientY - rect.top) / rect.height) * 2 + 1
            );

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(meshRef, false);

            if (intersects.length === 0) {
                setHoverEdge(null);
                setHoverPosition(null);
                setHoverLabel("");
                return;
            }

            const point = intersects[0].point;

            if (mode === "edge") {
                // Find closest edge
                const edge = findClosestEdge(point);
                if (edge) {
                    setHoverEdge(edge);
                    setHoverPosition(point.clone());

                    // Generate preview label
                    const result = measureEdge(edge);
                    setHoverLabel(result.label);
                } else {
                    setHoverEdge(null);
                    setHoverPosition(null);
                    setHoverLabel("");
                }
            } else {
                // Point mode - show cursor position
                setHoverPosition(point.clone());
                if (firstPoint) {
                    const dist = point.distanceTo(firstPoint);
                    setHoverLabel(`${dist.toFixed(2)} mm`);
                } else {
                    setHoverLabel("Click first point");
                }
            }
        };

        gl.domElement.addEventListener("pointermove", handlePointerMove);
        return () => gl.domElement.removeEventListener("pointermove", handlePointerMove);
    }, [enabled, meshRef, mode, firstPoint, findClosestEdge, measureEdge, camera, gl, raycaster]);

    // ============= Click Handler =============

    useEffect(() => {
        if (!enabled || !meshRef) return;

        const handlePointerDown = (event: PointerEvent) => {
            dragStartRef.current = { x: event.clientX, y: event.clientY };
        };

        const handlePointerUp = (event: PointerEvent) => {
            if (!dragStartRef.current) return;

            // Check if drag (camera rotation)
            const dx = event.clientX - dragStartRef.current.x;
            const dy = event.clientY - dragStartRef.current.y;
            if (Math.sqrt(dx * dx + dy * dy) > 5) {
                dragStartRef.current = null;
                return;
            }
            dragStartRef.current = null;

            // Get intersection
            const canvas = gl.domElement;
            const rect = canvas.getBoundingClientRect();
            const mouse = new THREE.Vector2(
                ((event.clientX - rect.left) / rect.width) * 2 - 1,
                -((event.clientY - rect.top) / rect.height) * 2 + 1
            );

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(meshRef, false);

            if (intersects.length === 0) return;

            const point = intersects[0].point;

            if (mode === "edge" && hoverEdge) {
                // Edge measurement
                const result = measureEdge(hoverEdge);

                if (result.success) {
                    const centerPoint = hoverEdge.center
                        ? new THREE.Vector3(...hoverEdge.center)
                        : point.clone();

                    const measurement: StoredMeasurement = {
                        id: `sw-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        result,
                        position: centerPoint,
                        color: getColorForType(result.edge_type),
                        visible: true,
                        createdAt: new Date()
                    };

                    setMeasurements(prev => [...prev, measurement]);

                    toast({
                        title: getToastTitle(result.edge_type),
                        description: result.label,
                    });
                }
            } else if (mode === "point") {
                // Point-to-point measurement
                if (!firstPoint) {
                    setFirstPoint(point.clone());
                    toast({
                        title: "First Point",
                        description: "Click second point to measure distance",
                    });
                } else {
                    const result = measurePointToPoint(firstPoint, point);

                    const measurement: StoredMeasurement = {
                        id: `sw-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        result,
                        position: firstPoint.clone(),
                        endPosition: point.clone(),
                        color: "#00CC66",
                        visible: true,
                        createdAt: new Date()
                    };

                    setMeasurements(prev => [...prev, measurement]);
                    setFirstPoint(null);

                    toast({
                        title: "Distance Measured",
                        description: result.label,
                    });
                }
            }
        };

        gl.domElement.addEventListener("pointerdown", handlePointerDown);
        gl.domElement.addEventListener("pointerup", handlePointerUp);

        return () => {
            gl.domElement.removeEventListener("pointerdown", handlePointerDown);
            gl.domElement.removeEventListener("pointerup", handlePointerUp);
        };
    }, [enabled, meshRef, mode, hoverEdge, firstPoint, measureEdge, measurePointToPoint, camera, gl, raycaster]);

    // ============= Helpers =============

    const getColorForType = (edgeType?: string): string => {
        switch (edgeType) {
            case 'circle': return '#0066CC';
            case 'arc': return '#00AACC';
            case 'line': return '#0066CC';
            case 'ellipse': return '#9900CC';
            default: return '#666666';
        }
    };

    const getToastTitle = (edgeType?: string): string => {
        switch (edgeType) {
            case 'circle': return 'Circle Measured';
            case 'arc': return 'Arc Measured';
            case 'line': return 'Line Measured';
            case 'ellipse': return 'Ellipse Measured';
            default: return 'Edge Measured';
        }
    };

    // ============= Render =============

    if (!enabled) return null;

    const radius = boundingSphere?.radius || 50;
    const markerRadius = radius * 0.008;

    return (
        <group>
            {/* Edge highlight on hover */}
            {hoverEdge && hoverEdge.snap_points && (
                <group>
                    {hoverEdge.snap_points.slice(0, -1).map((sp, idx) => {
                        const next = hoverEdge.snap_points![idx + 1];
                        if (!next) return null;

                        const positions = new Float32Array([
                            sp[0], sp[1], sp[2],
                            next[0], next[1], next[2]
                        ]);
                        const geometry = new THREE.BufferGeometry();
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                        return (
                            <lineSegments key={idx} geometry={geometry}>
                                <lineBasicMaterial color="#FF8800" linewidth={3} depthTest={false} />
                            </lineSegments>
                        );
                    })}
                    {/* Close loop for closed edges */}
                    {hoverEdge.is_closed && hoverEdge.snap_points.length > 2 && (() => {
                        const first = hoverEdge.snap_points![0];
                        const last = hoverEdge.snap_points![hoverEdge.snap_points!.length - 1];
                        const positions = new Float32Array([
                            last[0], last[1], last[2],
                            first[0], first[1], first[2]
                        ]);
                        const geometry = new THREE.BufferGeometry();
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        return (
                            <lineSegments geometry={geometry}>
                                <lineBasicMaterial color="#FF8800" linewidth={3} depthTest={false} />
                            </lineSegments>
                        );
                    })()}
                </group>
            )}

            {/* Hover label - SolidWorks style */}
            {hoverPosition && hoverLabel && (
                <Html position={hoverPosition} center>
                    <div style={{
                        background: 'linear-gradient(180deg, #2A2A2A 0%, #1A1A1A 100%)',
                        color: '#FFFFFF',
                        padding: '6px 12px',
                        borderRadius: '4px',
                        fontSize: '13px',
                        fontFamily: 'Arial, sans-serif',
                        fontWeight: 500,
                        whiteSpace: 'nowrap',
                        pointerEvents: 'none',
                        border: '1px solid #444',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                        transform: 'translateY(20px)',
                    }}>
                        {hoverLabel}
                    </div>
                </Html>
            )}

            {/* First point marker for point-to-point */}
            {firstPoint && (
                <group position={firstPoint}>
                    <mesh>
                        <sphereGeometry args={[markerRadius, 16, 16]} />
                        <meshBasicMaterial color="#00FF00" />
                    </mesh>
                </group>
            )}

            {/* Stored measurements display */}
            {measurements.map(m => (
                <group key={m.id}>
                    {/* Position marker */}
                    <group position={m.position}>
                        <mesh>
                            <sphereGeometry args={[markerRadius * 0.5, 8, 8]} />
                            <meshBasicMaterial color={m.color} />
                        </mesh>
                    </group>

                    {/* Label */}
                    <Html position={m.position} center>
                        <div style={{
                            background: m.color,
                            color: '#FFFFFF',
                            padding: '4px 8px',
                            borderRadius: '3px',
                            fontSize: '12px',
                            fontFamily: 'Arial, sans-serif',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                            pointerEvents: 'none',
                            transform: 'translateY(-25px)',
                            boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                        }}>
                            {m.result.label}
                        </div>
                    </Html>

                    {/* Line for point-to-point */}
                    {m.endPosition && (
                        <line>
                            <bufferGeometry>
                                <bufferAttribute
                                    attach="attributes-position"
                                    count={2}
                                    array={new Float32Array([
                                        m.position.x, m.position.y, m.position.z,
                                        m.endPosition.x, m.endPosition.y, m.endPosition.z
                                    ])}
                                    itemSize={3}
                                />
                            </bufferGeometry>
                            <lineBasicMaterial color={m.color} linewidth={2} />
                        </line>
                    )}
                </group>
            ))}
        </group>
    );
};

export default SolidWorksMeasureTool;
