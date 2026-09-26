import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";

import { attachSkinnedMeshTextures } from "./SkinnedMeshTexture.ts";

test("skinned human PNG maps retain glTF material channels and release on replacement", () => {
  const material = new THREE.MeshStandardMaterial();
  const pending = [];
  const disposed = [];
  const revoked = [];
  const loader = {
    load(url, onLoad) {
      const texture = new THREE.Texture();
      texture.dispose = () => disposed.push(texture);
      pending.push({ url, onLoad, texture });
      return texture;
    },
  };
  const urls = {
    createObjectURL(blob) {
      assert.equal(blob.type, "image/png");
      return `blob:mesh-${pending.length}`;
    },
    revokeObjectURL(url) { revoked.push(url); },
  };
  const bytes = new Uint8Array([137, 80, 78, 71]);
  const cleanup = attachSkinnedMeshTextures(material, {
    texture_png: bytes, normal_texture_png: bytes,
    metallic_roughness_texture_png: bytes,
  }, loader, urls);
  assert.equal(pending.length, 3);
  pending.forEach(({ onLoad, texture }) => onLoad(texture));
  assert.equal(material.map, pending[0].texture);
  assert.equal(material.normalMap, pending[1].texture);
  assert.equal(material.metalnessMap, pending[2].texture);
  assert.equal(material.roughnessMap, pending[2].texture);
  assert.equal(material.map.colorSpace, THREE.SRGBColorSpace);
  assert.equal(material.normalMap.colorSpace, THREE.NoColorSpace);
  assert.ok(pending.every(({ texture }) => texture.flipY));

  cleanup();
  assert.equal(material.map, null);
  assert.equal(material.normalMap, null);
  assert.equal(material.metalnessMap, null);
  assert.equal(material.roughnessMap, null);
  assert.deepEqual(disposed, pending.map(({ texture }) => texture));
  assert.deepEqual(revoked, pending.map(({ url }) => url));
  pending[0].onLoad(pending[0].texture);
  assert.equal(material.map, null, "a late image decode must not reattach a disposed texture");
  material.dispose();
});
