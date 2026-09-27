import * as THREE from "three";

/** Small, static accents in the original metre-scale environment coordinates.
 * Crossing dressing is added before its parent is compressed in X and Z. */
type Kind = "crossing" | "city" | "station";
type Mat = THREE.MeshStandardMaterial;
let boxGeometry: THREE.BoxGeometry = null!;
let cylinderGeometry: THREE.CylinderGeometry = null!;
let coneGeometry: THREE.ConeGeometry = null!;
function createPalette() { return {
  ink: new THREE.MeshStandardMaterial({ color: "#28444e", roughness: .72 }),
  cream: new THREE.MeshStandardMaterial({ color: "#fff2d3", roughness: .87 }),
  coral: new THREE.MeshStandardMaterial({ color: "#f36f5e", roughness: .82 }),
  orange: new THREE.MeshStandardMaterial({ color: "#eda445", roughness: .8 }),
  yellow: new THREE.MeshStandardMaterial({ color: "#ffd363", roughness: .8 }),
  turquoise: new THREE.MeshStandardMaterial({ color: "#42babc", roughness: .8 }),
  blue: new THREE.MeshStandardMaterial({ color: "#557fca", roughness: .78 }),
  violet: new THREE.MeshStandardMaterial({ color: "#aa7abc", roughness: .8 }),
  green: new THREE.MeshStandardMaterial({ color: "#518e66", roughness: .9 }),
  leaf: new THREE.MeshStandardMaterial({ color: "#4a935f", roughness: .92 }),
  soil: new THREE.MeshStandardMaterial({ color: "#665042", roughness: 1 }),
  metal: new THREE.MeshStandardMaterial({ color: "#768e92", metalness: .38, roughness: .58 }),
}; }
let palette: ReturnType<typeof createPalette> = null!;
let brights: Mat[] = [];
let primitiveMeshes: THREE.Mesh[] = [];

function primitive(parent: THREE.Object3D, geometry: THREE.BufferGeometry, mat: THREE.Material,
  x: number, y: number, z: number, sx: number, sy: number, sz: number) {
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz);
  mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh);
  primitiveMeshes.push(mesh);
  return mesh;
}
function box(parent: THREE.Object3D, mat: THREE.Material, x: number, y: number, z: number,
  width: number, height: number, depth: number) {
  return primitive(parent, boxGeometry, mat, x, y, z, width, height, depth);
}
function cylinder(parent: THREE.Object3D, mat: THREE.Material, x: number, y: number, z: number,
  radius: number, height: number) {
  return primitive(parent, cylinderGeometry, mat, x, y, z, radius, height, radius);
}
function local(parent: THREE.Object3D, x: number, z: number, yaw = 0) {
  const group = new THREE.Group(); group.position.set(x, 0, z); group.rotation.y = yaw;
  parent.add(group); return group;
}

function signTexture(title: string, subtitle: string, background: string, foreground = "#fff5db") {
  const canvas = document.createElement("canvas"); canvas.width = 512; canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D is required for street signage");
  ctx.fillStyle = background; ctx.fillRect(0, 0, 512, 512);
  ctx.strokeStyle = foreground; ctx.lineWidth = 7; ctx.strokeRect(27, 27, 458, 458);
  ctx.fillStyle = foreground; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = "bold 84px Arial, sans-serif";
  const words = title.split(" ");
  if (words.length > 1) {
    ctx.fillText(words[0], 256, 206, 430); ctx.fillText(words.slice(1).join(" "), 256, 308, 430);
  } else ctx.fillText(title, 256, 242, 430);
  ctx.font = "bold 27px Arial, sans-serif"; ctx.fillText(subtitle, 256, 431, 430);
  const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 4; return map;
}

// A sign faces local +Z. A 180-degree parent rotation makes it face the south approach.
function bladeSign(parent: THREE.Object3D, x: number, y: number, z: number,
  title: string, subtitle: string, color: string, width = 2.35, height = 4.1) {
  const sign = new THREE.Group(); sign.position.set(x, y, z); parent.add(sign);
  box(sign, palette.ink, 0, 0, -.12, width + .2, height + .2, .19);
  const map = signTexture(title, subtitle, color);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(width, height),
    new THREE.MeshStandardMaterial({ map, roughness: .75, side: THREE.DoubleSide,
      emissiveMap: map, emissive: "#ffffff", emissiveIntensity: .08 }));
  face.position.z = .012; sign.add(face);
  box(sign, palette.metal, 0, -height / 2 - .13, .11, width + .4, .12, .38);
}

function stripedAwning(parent: THREE.Object3D, x: number, z: number, width: number,
  colors: [Mat, Mat]) {
  // Mounted just above existing shop awnings; no ground-level footprint.
  const awning = local(parent, x, z);
  box(awning, palette.cream, 0, 2.75, .51, width, .055, 1.28);
  const stripes = 8;
  for (let i = 0; i < stripes; i++) {
    const xx = -width / 2 + (i + .5) * width / stripes;
    box(awning, colors[i % 2], xx, 2.79, .51, width / stripes - .02, .018, 1.27);
    box(awning, colors[i % 2], xx, 2.56, 1.13, width / stripes - .02, .37, .055);
  }
}

function flowerPlanter(parent: THREE.Object3D, x: number, z: number, width = 1.7, color = palette.coral) {
  const planter = local(parent, x, z);
  box(planter, color, 0, .35, 0, width, .7, .68);
  box(planter, palette.soil, 0, .705, 0, width - .12, .04, .56);
  for (let i = 0; i < 5; i++) {
    const fx = -width * .39 + i * width * .195;
    const fz = i % 2 ? -.13 : .13;
    cylinder(planter, palette.leaf, fx, .93, fz, .025, .43);
    primitive(planter, coneGeometry, palette.leaf, fx - .09, .9, fz, .12, .22, .08).rotation.z = .8;
    cylinder(planter, brights[(i + (color === palette.coral ? 2 : 0)) % brights.length], fx, 1.17, fz, .105, .09);
  }
}

function bunting(parent: THREE.Object3D, ax: number, az: number, bx: number, bz: number, y = 9.7) {
  const a = new THREE.Vector3(ax, y, az), b = new THREE.Vector3(bx, y, bz);
  const cable = cylinder(parent, palette.metal, 0, 0, 0, .018, a.distanceTo(b));
  cable.position.copy(a).add(b).multiplyScalar(.5);
  cable.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  const length = a.distanceTo(b), count = Math.floor(length / 1.8);
  for (let i = 1; i < count; i++) {
    const t = i / count, x = ax + (bx - ax) * t, z = az + (bz - az) * t;
    const flag = primitive(parent, coneGeometry, brights[i % brights.length], x, y - .42, z,
      .38, .82, .06); flag.rotation.z = Math.PI;
  }
}

function vendingMachine(parent: THREE.Object3D, x: number, z: number, color: Mat) {
  const machine = local(parent, x, z);
  box(machine, color, 0, 1.03, 0, 1.05, 2.06, .68);
  box(machine, palette.cream, 0, 1.51, .351, .84, .75, .025);
  for (let row = 0; row < 2; row++) for (let col = 0; col < 5; col++) {
    box(machine, brights[(row * 2 + col) % brights.length], -.31 + col * .155,
      1.72 - row * .3, .372, .105, .2, .018);
  }
  box(machine, palette.ink, -.14, .72, .359, .57, .18, .025);
  box(machine, palette.metal, .34, .9, .36, .14, .34, .028);
}

function cafeTable(parent: THREE.Object3D, x: number, z: number, canopy: Mat) {
  const setting = local(parent, x, z);
  cylinder(setting, palette.ink, 0, .5, 0, .045, 1);
  cylinder(setting, palette.cream, 0, .83, 0, .62, .075);
  cylinder(setting, palette.ink, 0, 2.07, 0, .04, 2.4);
  const cover = primitive(setting, coneGeometry, canopy, 0, 2.67, 0, 1.85, .64, 1.85);
  cover.material.side = THREE.DoubleSide;
  cylinder(setting, palette.cream, 0, 2.37, 0, 1.78, .045);
  for (const side of [-1, 1]) {
    box(setting, palette.ink, side * 1.05, .35, 0, .06, .7, .55);
    box(setting, palette.cream, side * 1.05, .72, 0, .48, .065, .5);
  }
}

function batchSolidMeshes(root: THREE.Group) {
  // Baking local transforms into instance matrices also collapses the many tiny
  // flower, awning, and vending meshes to one draw per geometry/material pair.
  root.updateWorldMatrix(true, true);
  const inverseRoot = root.matrixWorld.clone().invert();
  const batches = new Map<string, { geometry: THREE.BufferGeometry; material: THREE.Material; matrices: THREE.Matrix4[] }>();
  for (const mesh of primitiveMeshes) {
    const key = `${mesh.geometry.uuid}/${(mesh.material as THREE.Material).uuid}`;
    let batch = batches.get(key);
    if (!batch) {
      batch = { geometry: mesh.geometry, material: mesh.material as THREE.Material, matrices: [] };
      batches.set(key, batch);
    }
    batch.matrices.push(inverseRoot.clone().multiply(mesh.matrixWorld));
    mesh.parent?.remove(mesh);
  }
  for (const batch of batches.values()) {
    const instances = new THREE.InstancedMesh(batch.geometry, batch.material, batch.matrices.length);
    for (let i = 0; i < batch.matrices.length; i++) instances.setMatrixAt(i, batch.matrices[i]);
    instances.instanceMatrix.needsUpdate = true;
    instances.computeBoundingSphere();
    instances.castShadow = true; instances.receiveShadow = true;
    root.add(instances);
  }
  primitiveMeshes = [];
}

export function addStreetDressing(group: THREE.Group, kind: Kind): void {
  // Scene switching disposes every geometry and material below the environment
  // group. Allocate these per call so the next scene never reuses disposed GPU data.
  boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 12);
  coneGeometry = new THREE.ConeGeometry(1, 1, 8);
  palette = createPalette();
  brights = [palette.coral, palette.yellow, palette.turquoise, palette.orange, palette.violet];
  primitiveMeshes = [];
  const dressing = new THREE.Group(); dressing.name = `street-dressing-${kind}`;
  group.add(dressing); group = dressing;
  if (kind === "crossing") {
    // Original crossing coordinates: 44 m road, curb at 22 m, facades near 30 m.
    // Facade details are above walking height; freestanding objects stay outside the 22 m roadway.
    for (const side of [-1, 1]) {
      const facade = local(group, 0, side * 31.3, side < 0 ? 0 : Math.PI);
      for (const x of [-43, 43]) {
        stripedAwning(facade, x, 0, 8.6,
          x < 0 ? [palette.coral, palette.cream] : [palette.turquoise, palette.cream]);
        flowerPlanter(group, x - 4, side * 29.7, 1.5, palette.orange);
        flowerPlanter(group, x + 4, side * 29.7, 1.5, palette.turquoise);
      }
      // Vertical signs sit on the secondary storefronts, away from the corner landmark.
      bladeSign(facade, -68, 10, 3.9, "MORI", "MARKET & MORE", "#ef8068");
      bladeSign(facade, 68, 12, 3.9, "NAMI", "TEA HOUSE", "#43aeb8");
      bunting(group, -28, side * 56, 28, side * 56, 10.5);
    }
    for (const side of [-1, 1]) {
      vendingMachine(group, side * 53.5, -29, side < 0 ? palette.coral : palette.blue);
      vendingMachine(group, side * 55, -29, side < 0 ? palette.turquoise : palette.orange);
    }
    batchSolidMeshes(dressing);
    return;
  }

  if (kind === "city") {
    for (const [x, colors] of [[-51, [palette.coral, palette.cream]],
      [44, [palette.turquoise, palette.cream]], [69, [palette.orange, palette.cream]]] as [number, [Mat, Mat]][]) {
      stripedAwning(group, x, -16.5, 7.4, colors);
    }
    bladeSign(group, 45, 10.4, -16.6, "SORA", "OPTICS", "#e97c5b", 2.3, 3.9);
    bladeSign(group, 69, 9.6, -16.5, "MORI", "FLOWERS", "#4caaa5", 2.3, 3.8);
    for (const x of [-78, -58, 55, 79]) flowerPlanter(group, x, -15.8, 1.7,
      x < 0 ? palette.orange : palette.turquoise);
    vendingMachine(group, 79, -15.9, palette.coral);
    vendingMachine(group, 80.2, -15.9, palette.blue);
    // The courtyard's outer edge holds the café setting; the 5 m alley remains clear.
    cafeTable(group, -58, -62, palette.coral);
    cafeTable(group, -52, -63, palette.turquoise);
    flowerPlanter(group, -64, -63, 1.8, palette.orange);
    flowerPlanter(group, -45, -63, 1.8, palette.violet);
    bunting(group, -67, -69, -31, -69, 8.8);
    batchSolidMeshes(dressing);
    return;
  }

  // Station: color follows the entry canopy and retail edges, leaving the 18 m level route clear.
  for (const x of [-34, -27, 27, 34]) flowerPlanter(group, x, 9, 2, x < 0 ? palette.coral : palette.turquoise);
  for (const x of [-53, 53]) {
    vendingMachine(group, x, 3, palette.orange);
    vendingMachine(group, x + 1.2, 3, palette.blue);
  }
  cafeTable(group, -49, 23, palette.coral);
  cafeTable(group, 49, 23, palette.turquoise);
  bunting(group, -39, -19, -11, -19, 8.6);
  bunting(group, 11, -19, 39, -19, 8.6);
  cylinder(group, palette.metal, -47, 4, -12, .07, 8);
  cylinder(group, palette.metal, 47, 4, -12, .07, 8);
  bladeSign(group, -47, 8, -12, "KOMA", "EXPRESS", "#e78061");
  bladeSign(group, 47, 8, -12, "MORI", "FLOWERS", "#4caeab");
  batchSolidMeshes(dressing);
}
