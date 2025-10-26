import React from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

interface ProfessionalLightingProps {
  intensity?: number;
  enableShadows?: boolean;
  shadowQuality?: "low" | "medium" | "high";
}

/**
 * Professional 5-Light PBR Lighting System
 *
 * Based on industry-standard photography/3D lighting:
 * - Key Light: Main illumination (front-top-right)
 * - Fill Light: Softens shadows (front-left, lower intensity)
 * - Rim Light 1: Edge definition (back-left-top)
 * - Rim Light 2: Edge definition (back-right-top)
 * - Hemisphere: Ambient base lighting (sky-ground gradient)
 *
 * FIXED VERSION: Improved shadow settings to prevent self-shadowing artifacts
 */
export function ProfessionalLighting({
  intensity = 1.0,
  enableShadows = true,
  shadowQuality = "high",
}: ProfessionalLightingProps) {
  // Shadow map size based on quality
  const shadowMapSize = {
    low: 512,
    medium: 1024,
    high: 2048,
  }[shadowQuality];

  return (
    <>
      {/* Hemisphere Light - Ambient base (sky + ground) - INCREASED for softer shadows */}
      <hemisphereLight
        args={[
          "#ffffff", // Sky color (cool white)
          "#444444", // Ground color (warm gray)
          0.4 * intensity, // Increased from 0.3 to reduce shadow contrast
        ]}
      />

      {/* Key Light - Main illumination (front-top-right) - SOFTER with improved shadow bias */}
      <directionalLight
        position={[5, 8, 5]}
        intensity={0.8 * intensity}
        castShadow={enableShadows}
        shadow-mapSize-width={shadowMapSize}
        shadow-mapSize-height={shadowMapSize}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
        shadow-camera-near={0.1}
        shadow-camera-far={50}
        shadow-bias={-0.0005}
        shadow-normalBias={0.02}
      />

      {/* Fill Light - Softens shadows (front-left) - BRIGHTER */}
      <directionalLight position={[-4, 4, 4]} intensity={0.7 * intensity} castShadow={false} />

      {/* Rim Light 1 - Edge definition (back-left-top) - with improved shadow settings */}
      <directionalLight
        position={[-5, 6, -5]}
        intensity={0.6 * intensity}
        castShadow={enableShadows && shadowQuality !== "low"}
        shadow-mapSize-width={shadowMapSize / 2}
        shadow-mapSize-height={shadowMapSize / 2}
        shadow-bias={-0.0005}
        shadow-normalBias={0.02}
      />

      {/* Rim Light 2 - Edge definition (back-right-top) */}
      <directionalLight position={[5, 6, -5]} intensity={0.6 * intensity} castShadow={false} />

      {/* Subtle ambient light to prevent pure black shadows - INCREASED */}
      <ambientLight intensity={0.2 * intensity} />
    </>
  );
}
