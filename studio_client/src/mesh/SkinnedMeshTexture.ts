import * as THREE from "three";

/** Optional PNG maps transported with a generated human's skinned mesh. */
export type SkinnedMeshTextureProps = {
  texture_png?: Uint8Array<ArrayBuffer>;
  normal_texture_png?: Uint8Array<ArrayBuffer> | null;
  metallic_roughness_texture_png?: Uint8Array<ArrayBuffer> | null;
};

type TextureLoaderLike = Pick<THREE.TextureLoader, "load">;
type UrlApi = Pick<typeof URL, "createObjectURL" | "revokeObjectURL">;

/** Attach GLB maps after image decoding, and release every resource on replacement. */
export function attachSkinnedMeshTextures(
  material: THREE.MeshStandardMaterial,
  props: SkinnedMeshTextureProps,
  loader: TextureLoaderLike = new THREE.TextureLoader(),
  urls: UrlApi = URL,
): () => void {
  const objectUrls: string[] = [];
  const textures: THREE.Texture[] = [];
  let active = true;
  const attach = (
    bytes: Uint8Array<ArrayBuffer> | null | undefined,
    apply: (texture: THREE.Texture) => void,
    srgb = false,
  ) => {
    if (!bytes?.byteLength) return;
    const url = urls.createObjectURL(new Blob([bytes], { type: "image/png" }));
    objectUrls.push(url);
    textures.push(loader.load(url, (texture) => {
      if (!active) return;
      texture.flipY = true; // Trimesh exports the original glTF UV orientation.
      if (srgb) texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      apply(texture);
      material.needsUpdate = true;
    }));
  };
  attach(props.texture_png, (texture) => { material.map = texture; }, true);
  attach(props.normal_texture_png, (texture) => { material.normalMap = texture; });
  attach(props.metallic_roughness_texture_png, (texture) => {
    material.metalnessMap = texture;
    material.roughnessMap = texture;
  });
  return () => {
    active = false;
    if (textures.includes(material.map!)) material.map = null;
    if (textures.includes(material.normalMap!)) material.normalMap = null;
    if (textures.includes(material.metalnessMap!)) material.metalnessMap = null;
    if (textures.includes(material.roughnessMap!)) material.roughnessMap = null;
    material.needsUpdate = true;
    objectUrls.forEach((url) => urls.revokeObjectURL(url));
    textures.forEach((texture) => texture.dispose());
  };
}
