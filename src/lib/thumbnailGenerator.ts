import * as THREE from 'three';

interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
}

/**
 * Generates a thumbnail image from mesh data using Three.js
 * @param meshData - The mesh data containing vertices, indices, and normals
 * @param width - Thumbnail width (default: 256)
 * @param height - Thumbnail height (default: 256)
 * @returns Promise<Blob> - The generated thumbnail as a PNG blob
 */
export async function generateThumbnail(
  meshData: MeshData,
  width: number = 256,
  height: number = 256
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    try {
      // Create off-screen renderer
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
        alpha: true,
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(1);
      renderer.setClearColor(0xf8f9fa, 1);

      // Create scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8f9fa);

      // Create camera
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

      // Create geometry from mesh data
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(meshData.vertices, 3)
      );
      geometry.setIndex(meshData.indices);
      
      if (meshData.normals && meshData.normals.length > 0) {
        geometry.setAttribute(
          'normal',
          new THREE.Float32BufferAttribute(meshData.normals, 3)
        );
      } else {
        geometry.computeVertexNormals();
      }

      // Create material - professional CAD-like appearance
      const material = new THREE.MeshStandardMaterial({
        color: 0x7c9bc4,
        metalness: 0.3,
        roughness: 0.6,
        flatShading: false,
      });

      // Create mesh
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      // Calculate bounding box and center the model
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox!;
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);
      mesh.position.sub(center);

      // Calculate camera position to fit the model
      const size = new THREE.Vector3();
      boundingBox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      const cameraDistance = maxDim / (2 * Math.tan(fov / 2)) * 1.8;

      // Position camera at an isometric-like angle
      camera.position.set(
        cameraDistance * 0.7,
        cameraDistance * 0.5,
        cameraDistance * 0.7
      );
      camera.lookAt(0, 0, 0);

      // Add lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight1.position.set(1, 2, 1.5);
      scene.add(directionalLight1);

      const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
      directionalLight2.position.set(-1, -0.5, -1);
      scene.add(directionalLight2);

      // Render
      renderer.render(scene, camera);

      // Get image as blob
      renderer.domElement.toBlob(
        (blob) => {
          // Clean up
          geometry.dispose();
          material.dispose();
          renderer.dispose();

          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Failed to generate thumbnail blob'));
          }
        },
        'image/png',
        0.9
      );
    } catch (error) {
      reject(error);
    }
  });
}
