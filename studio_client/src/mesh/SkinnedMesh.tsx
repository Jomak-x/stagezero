import React from "react";
import * as THREE from "three";
import { createStandardMaterial } from "./MeshUtils";
import { SkinnedMeshMessage } from "../WebsocketMessages";
import { OutlinesIfHovered } from "../OutlinesIfHovered";
import { ViewerContext } from "../ViewerContext";
import { useFrame, useThree } from "@react-three/fiber";

/**
 * Component for rendering skinned meshes with animations
 */
export const SkinnedMesh = React.forwardRef<
  THREE.SkinnedMesh,
  SkinnedMeshMessage & { children?: React.ReactNode }
>(function SkinnedMesh(
  { children, ...message },
  ref: React.ForwardedRef<THREE.SkinnedMesh>,
) {
  const viewer = React.useContext(ViewerContext)!;
  const maxAnisotropy = useThree((state) => state.gl.capabilities.getMaxAnisotropy());

  // Load the glTF maps separately: only base color is encoded in sRGB.
  const [textures, setTextures] = React.useState<{
    color: THREE.Texture | null;
    normal: THREE.Texture | null;
    metallicRoughness: THREE.Texture | null;
  }>({ color: null, normal: null, metallicRoughness: null });
  React.useLayoutEffect(() => {
    const urls: string[] = [];
    const maps: THREE.Texture[] = [];
    const loader = new THREE.TextureLoader();
    let disposed = false;
    const load = (bytes: Uint8Array<ArrayBuffer> | null | undefined, srgb: boolean) => {
      if (!bytes) return null;
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
      urls.push(url);
      const map = loader.load(
        url,
        (loaded) => {
          URL.revokeObjectURL(url);
          if (disposed) loaded.dispose();
        },
        undefined,
        () => URL.revokeObjectURL(url),
      );
      map.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      map.anisotropy = Math.min(8, maxAnisotropy);
      maps.push(map);
      return map;
    };
    setTextures({
      color: load(message.props.texture_png, true),
      normal: load(message.props.normal_texture_png, false),
      metallicRoughness: load(message.props.metallic_roughness_texture_png, false),
    });
    return () => {
      disposed = true;
      for (const map of maps) map.dispose();
      for (const url of urls) URL.revokeObjectURL(url);
    };
  }, [
    message.props.texture_png,
    message.props.normal_texture_png,
    message.props.metallic_roughness_texture_png,
    maxAnisotropy,
  ]);

  // Create material based on props.
  const material = React.useMemo(() => {
    const material = createStandardMaterial(message.props);
    if (message.props.texture_png && material instanceof THREE.MeshStandardMaterial) {
      material.map = textures.color;
      material.normalMap = textures.normal;
      material.roughnessMap = textures.metallicRoughness;
      material.metalnessMap = textures.metallicRoughness;
      material.metalness = message.props.metallic_factor ?? 0;
      material.roughness = message.props.roughness_factor ?? 1;
      const factor = message.props.base_color_factor ?? [1, 1, 1, 1];
      material.color.multiply(new THREE.Color().setRGB(factor[0], factor[1], factor[2]));
      material.opacity *= factor[3];
      material.transparent ||= factor[3] < 1;
    }
    return material;
  }, [
    textures,
    message.props.material,
    message.props.color,
    message.props.metallic_factor,
    message.props.roughness_factor,
    message.props.base_color_factor,
    message.props.wireframe,
    message.props.opacity,
    message.props.flat_shading,
    message.props.side,
  ]);

  // Reference to bones for animation updates.
  const bonesRef = React.useRef<THREE.Bone[]>();

  // Create geometry and skeleton using memoization.
  const { geometry, skeleton } = React.useMemo(() => {
    // Setup geometry.
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.vertices.buffer.slice(
            message.props.vertices.byteOffset,
            message.props.vertices.byteOffset +
              message.props.vertices.byteLength,
          ),
        ),
        3,
      ),
    );
    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(
          message.props.faces.buffer.slice(
            message.props.faces.byteOffset,
            message.props.faces.byteOffset + message.props.faces.byteLength,
          ),
        ),
        1,
      ),
    );
    if (message.props.uv) {
      const uv = message.props.uv;
      geometry.setAttribute("uv", new THREE.BufferAttribute(
        new Float32Array(uv.buffer.slice(uv.byteOffset, uv.byteOffset + uv.byteLength)), 2,
      ));
    }
    if (message.props.normals) {
      const normals = message.props.normals;
      geometry.setAttribute("normal", new THREE.BufferAttribute(
        new Float32Array(normals.buffer.slice(normals.byteOffset, normals.byteOffset + normals.byteLength)), 3,
      ));
    } else {
      geometry.computeVertexNormals();
    }
    geometry.computeBoundingSphere();

    // Setup skinned mesh bones.
    const bone_wxyzs = new Float32Array(
      message.props.bone_wxyzs.buffer.slice(
        message.props.bone_wxyzs.byteOffset,
        message.props.bone_wxyzs.byteOffset +
          message.props.bone_wxyzs.byteLength,
      ),
    );
    const bone_positions = new Float32Array(
      message.props.bone_positions.buffer.slice(
        message.props.bone_positions.byteOffset,
        message.props.bone_positions.byteOffset +
          message.props.bone_positions.byteLength,
      ),
    );

    const bones: THREE.Bone[] = [];
    bonesRef.current = bones;
    for (let i = 0; i < bone_positions.length / 3; i++) {
      bones.push(new THREE.Bone());
    }
    const boneInverses: THREE.Matrix4[] = [];
    const xyzw_quat = new THREE.Quaternion();
    bones.forEach((bone, i) => {
      xyzw_quat.set(
        bone_wxyzs[i * 4 + 1],
        bone_wxyzs[i * 4 + 2],
        bone_wxyzs[i * 4 + 3],
        bone_wxyzs[i * 4 + 0],
      );

      const boneInverse = new THREE.Matrix4();
      boneInverse.makeRotationFromQuaternion(xyzw_quat);
      boneInverse.setPosition(
        bone_positions[i * 3 + 0],
        bone_positions[i * 3 + 1],
        bone_positions[i * 3 + 2],
      );
      boneInverse.invert();
      boneInverses.push(boneInverse);

      bone.setRotationFromQuaternion(xyzw_quat);
      bone.position.set(
        bone_positions[i * 3 + 0],
        bone_positions[i * 3 + 1],
        bone_positions[i * 3 + 2],
      );
    });
    const skeleton = new THREE.Skeleton(bones, boneInverses);

    geometry.setAttribute(
      "skinIndex",
      new THREE.BufferAttribute(
        new Uint16Array(
          message.props.skin_indices.buffer.slice(
            message.props.skin_indices.byteOffset,
            message.props.skin_indices.byteOffset +
              message.props.skin_indices.byteLength,
          ),
        ),
        4,
      ),
    );
    geometry.setAttribute(
      "skinWeight",
      new THREE.BufferAttribute(
        new Float32Array(
          message.props.skin_weights!.buffer.slice(
            message.props.skin_weights!.byteOffset,
            message.props.skin_weights!.byteOffset +
              message.props.skin_weights!.byteLength,
          ),
        ),
        4,
      ),
    );

    skeleton.init();
    return { geometry, skeleton };
  }, [
    message.props.normals,
    message.props.uv,
    message.props.vertices.buffer,
    message.props.faces.buffer,
    message.props.skin_indices.buffer,
    message.props.skin_weights?.buffer,
    message.props.bone_wxyzs.buffer,
    message.props.bone_positions.buffer,
  ]);

  // Handle initialization and cleanup.
  // Get mutable once.
  const viewerMutable = viewer.mutable.current;

  // Clean up geometry and skeleton when they change (they're created together).
  React.useEffect(() => {
    const state = viewerMutable.skinnedMeshState[message.name];
    if (state !== undefined) state.initialized = false;
    return () => {
      if (skeleton) skeleton.dispose();
      if (geometry) geometry.dispose();
    };
  }, [skeleton, geometry, message.name, viewerMutable.skinnedMeshState]);

  // Clean up material when it changes.
  React.useEffect(() => {
    return () => {
      if (material) material.dispose();
    };
  }, [material]);

  // Check if we should render a shadow mesh.
  const shadowOpacity =
    typeof message.props.receive_shadow === "number"
      ? message.props.receive_shadow
      : 0.0;

  // Create shadow material for shadow mesh.
  const shadowMaterial = React.useMemo(() => {
    if (shadowOpacity === 0.0) return null;
    return new THREE.ShadowMaterial({
      opacity: shadowOpacity,
      color: 0x000000,
      depthWrite: false,
    });
  }, [shadowOpacity]);

  // Update bone transforms for animation.
  useFrame(() => {
    const state = viewerMutable.skinnedMeshState[message.name];
    const bones = bonesRef.current;
    // Removal messages clear mutable state before React unmounts the mesh.
    // A render frame may still run during that character-switch boundary.
    if (state === undefined) return;
    if (skeleton !== undefined && bones !== undefined) {
      if (!state.initialized) {
        const parentNode = viewerMutable.nodeRefFromName[message.name];
        if (parentNode === undefined) return;
        bones.forEach((bone) => {
          parentNode.add(bone);
        });
        state.initialized = true;
      }

      // Only update bones if dirty flag is set.
      if (state.dirty) {
        bones.forEach((bone, i) => {
          const wxyz = state.poses[i].wxyz;
          const position = state.poses[i].position;
          bone.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
          bone.position.set(position[0], position[1], position[2]);
        });
        state.dirty = false; // Reset dirty flag after update.
      }
    }
  });

  return (
    <skinnedMesh
      ref={ref}
      geometry={geometry}
      material={material}
      skeleton={skeleton}
      castShadow={message.props.cast_shadow}
      receiveShadow={message.props.receive_shadow === true}
      frustumCulled={false}
    >
      <OutlinesIfHovered
        enableCreaseAngle={geometry.attributes.position.count < 1024}
      />
      {shadowMaterial && shadowOpacity > 0 ? (
        <skinnedMesh
          geometry={geometry}
          material={shadowMaterial}
          skeleton={skeleton}
          receiveShadow
          frustumCulled={false}
        />
      ) : null}
      {children}
    </skinnedMesh>
  );
});
