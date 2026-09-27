import * as THREE from "three";

/** Original, deterministic, metre-scale architecture. All geometry is authored here;
 * the local advertising atlas contains original generated raster artwork. No external models or network requests. */
export type EnvironmentKind = "crossing" | "city" | "station";
export interface EnvironmentCamera {
  id: string; label: string; position: [number, number, number];
  target: [number, number, number]; fov?: number;
}
export interface EnvironmentResult {
  group: THREE.Group; cameras: EnvironmentCamera[];
  lighting: { background: string; fogNear: number; fogFar: number;
    sun: [number, number, number]; sunColor: string; sunIntensity: number;
    ambientIntensity: number; exposure: number };
  metadata: {
    title: string; description: string; units: string; provenance: string;
    bounds: { min: [number, number, number]; max: [number, number, number] };
    walkableRects: { id: string; min: [number, number]; max: [number, number]; y: number }[];
    spawnExitGates: { id: string; position: [number, number, number]; width: number; direction: [number, number, number] }[];
    interactionAnchors: { id: string; label: string; position: [number, number, number] }[];
    clearWidths: Record<string, number>; limitations: string[];
  };
}
type Mat = THREE.MeshStandardMaterial;
const UP = new THREE.Vector3(0, 1, 0);
let randomState = 49;
function random() { randomState = (1664525 * randomState + 1013904223) >>> 0; return randomState / 4294967296; }
function material(color: string, roughness = .8, metalness = 0, emissive?: string): Mat {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness,
    emissive: emissive || "#000000", emissiveIntensity: emissive ? .36 : 0 });
}
function canvasTexture(width: number, height: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("Canvas 2D is required for environment artwork");
  draw(ctx); const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace; texture.anisotropy = 8; return texture;
}
function surface(kind: "asphalt" | "paving" | "concrete" | "wood", repeat = 1) {
  const texture = canvasTexture(512, 512, ctx => {
    ctx.fillStyle = { asphalt: "#454a4c", paving: "#bab8ae", concrete: "#b8b3a8", wood: "#a77d55" }[kind];
    ctx.fillRect(0, 0, 512, 512);
    for (let i = 0; i < 24000; i++) {
      const v = Math.floor(100 + random() * 100);
      ctx.fillStyle = `rgba(${v},${v},${v},${kind === "asphalt" ? .16 : .08})`;
      ctx.fillRect(random() * 512, random() * 512, 1 + random() * 2, 1 + random() * 2);
    }
    if (kind === "paving") {
      ctx.strokeStyle = "#96988f"; ctx.lineWidth = 2;
      for (let y = 0; y <= 512; y += 64) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(512, y); ctx.stroke();
        for (let x = (y / 64) % 2 ? -64 : 0; x < 512; x += 128) {
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 64); ctx.stroke();
        }
      }
    }
    if (kind === "wood") {
      for (let x = 0; x < 512; x += 32) {
        ctx.fillStyle = "rgba(45,32,20,.16)"; ctx.fillRect(x, 0, 2, 512);
        for (let i = 0; i < 9; i++) { ctx.fillStyle = "rgba(60,40,25,.09)"; ctx.fillRect(x + random() * 30, 0, .6, 512); }
      }
    }
    if (kind === "asphalt") {
      ctx.strokeStyle = "rgba(20,23,26,.23)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, 182); ctx.lineTo(102, 171); ctx.lineTo(190, 203); ctx.lineTo(268, 195); ctx.stroke();
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(repeat, repeat);
  return new THREE.MeshStandardMaterial({ map: texture, roughness: kind === "wood" ? .74 : .94 });
}
function windowMaterial(quadrant: number): Mat {
  const map = new THREE.TextureLoader().load(new URL("./assets/windows.png", import.meta.url).href);
  map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 8;
  // Inset within the original atlas quadrants avoids sampling their border pixels.
  map.repeat.set(.488, .488); map.offset.set((quadrant % 2) * .5 + .006, quadrant < 2 ? .506 : .006);
  return new THREE.MeshStandardMaterial({map, color: "#d5dfe0", roughness: .3, metalness: .18,
    emissiveMap: map, emissive: "#ffffff", emissiveIntensity: quadrant === 3 ? .28 : .08});
}
class Builder {
  group = new THREE.Group();
  readonly BOX = new THREE.BoxGeometry(1, 1, 1);
  readonly CYLINDER = new THREE.CylinderGeometry(1, 1, 1, 24);
  readonly SPHERE = new THREE.IcosahedronGeometry(1, 1);
  white = material("#eee9d8"); dark = material("#263335", .56, .4);
  metal = material("#697678", .4, .65); concrete = surface("concrete");
  paving = surface("paving"); wood = surface("wood");
  glass = material("#7d9d9e", .22, .46);
  glassDark = material("#324c54", .2, .52);
  warm = material("#f8dca3", .55, 0, "#ffc379");
  leaf = [material("#3c5345"), material("#516448"), material("#687154")];
  accents = [material("#ccc7b8"), material("#a9adb0"), material("#766d63"), material("#d9d6c9"), material("#8d9c9d")];
  windows = [windowMaterial(0), windowMaterial(1), windowMaterial(0), windowMaterial(2), windowMaterial(3)];
  instancedCount = 0;
  private paintStrips: {x:number;z:number;angle:number;width:number}[] = [];
  mesh(parent: THREE.Object3D, geometry: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number, sx = 1, sy = 1, sz = 1, ry = 0) {
    const mesh = new THREE.Mesh(geometry, mat); mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz); mesh.rotation.y = ry;
    mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
  }
  box(parent: THREE.Object3D, mat: THREE.Material, x: number, y: number, z: number, w: number, h: number, d: number, ry = 0) {
    return this.mesh(parent, this.BOX, mat, x, y, z, w, h, d, ry);
  }
  cyl(parent: THREE.Object3D, mat: THREE.Material, x: number, y: number, z: number, r: number, h: number) {
    return this.mesh(parent, this.CYLINDER, mat, x, y, z, r, h, r);
  }
  beam(parent: THREE.Object3D, mat: THREE.Material, a: THREE.Vector3, b: THREE.Vector3, width: number) {
    const mesh = this.box(parent, mat, 0, 0, 0, width, a.distanceTo(b), width);
    mesh.position.copy(a).add(b).multiplyScalar(.5); mesh.quaternion.setFromUnitVectors(UP, b.clone().sub(a).normalize()); return mesh;
  }
  local(x: number, z: number, rotation = 0) { const g = new THREE.Group(); g.position.set(x, 0, z); g.rotation.y = rotation; this.group.add(g); return g; }
  plane(mat: Mat, x: number, z: number, w: number, d: number, y = .005) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat); mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, y, z); mesh.receiveShadow = true; this.group.add(mesh); return mesh;
  }
  text(parent: THREE.Object3D, words: string, x: number, y: number, z: number, w: number, h: number,
    background = "#213b3c", foreground = "#f2efdf", small = "", style: "shop" | "poster" | "wayfinding" = "shop") {
    const canvasHeight = Math.max(64, Math.min(2048, Math.round(1024 * h / w)));
    const texture = canvasTexture(1024, canvasHeight, ctx => {
      ctx.fillStyle = background; ctx.fillRect(0, 0, 1024, canvasHeight);
      if (style === "poster") {
        ctx.strokeStyle = foreground; ctx.lineWidth = 20; ctx.beginPath(); ctx.arc(760, 205, 174, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = .5; ctx.fillStyle = foreground; ctx.fillRect(610, 350, 410, 12); ctx.globalAlpha = 1;
        ctx.textAlign = "left"; ctx.font = "bold 94px Arial, sans-serif";
        const parts = words.split(" "); parts.forEach((word, index) => ctx.fillText(word, 66, 150 + index * 99));
        ctx.font = "26px Arial, sans-serif"; ctx.fillText(small || "A NEW PERSPECTIVE / 2026", 66, 460);
      } else {
        const margin = Math.min(22, canvasHeight * .08);
        ctx.strokeStyle = foreground; ctx.globalAlpha = .42; ctx.lineWidth = 2;
        ctx.strokeRect(18, margin, 988, canvasHeight - margin * 2); ctx.globalAlpha = 1;
        ctx.fillStyle = foreground; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        if (h / w > 2) {
          const letters = words.replace(/ /g, "").split("");
          const fs = Math.min(460, canvasHeight * .7 / letters.length);
          ctx.font = `600 ${fs}px Arial, sans-serif`;
          letters.forEach((letter, i) => ctx.fillText(letter, 512, canvasHeight * .12 + (i + .5) * canvasHeight * .72 / letters.length));
        } else {
          const fontSize = Math.min(920 / Math.max(words.length * .61, 5), canvasHeight * (small ? .38 : .54));
          ctx.font = `600 ${fontSize}px Arial, sans-serif`;
          ctx.fillText(words, 512, canvasHeight * (small ? .4 : .51), 920);
          if (small) {
            ctx.font = `${Math.min(fontSize * .36, canvasHeight * .16)}px Arial, sans-serif`;
            ctx.fillText(small, 512, canvasHeight * .73, 900);
          }
        }
      }
    });
    const mat = new THREE.MeshStandardMaterial({ map: texture, roughness: .7, emissiveMap: texture,
      emissive: "#ffffff", emissiveIntensity: .12 });
    this.box(parent, this.dark, x, y, z - .065, w + .16, h + .16, .16);
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); panel.position.set(x, y, z + .023); parent.add(panel);
  }
  billboard(parent: THREE.Object3D, quadrant: number, x: number, y: number, z: number, w: number, h: number) {
    const texture = new THREE.TextureLoader().load(new URL("./assets/billboards.png", import.meta.url).href);
    texture.colorSpace = THREE.SRGBColorSpace; texture.anisotropy = 8;
    texture.repeat.set(.5, .5); texture.offset.set(quadrant % 2 * .5, quadrant < 2 ? .5 : 0);
    const mat = new THREE.MeshStandardMaterial({ map: texture, roughness: .65, emissiveMap: texture,
      emissive: "#ffffff", emissiveIntensity: .1 });
    this.box(parent, this.dark, x, y, z - .08, w + .35, h + .35, .24);
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); panel.position.set(x, y, z + .06); parent.add(panel);
    this.box(parent, this.metal, x, y - h / 2 - .15, z + .25, w + .5, .16, .6);
  }
  pavement(x: number, z: number, w: number, d: number) {
    this.box(this.group, this.concrete, x, -.13, z, w, .25, d);
    const m = this.paving.clone(); m.map = this.paving.map!.clone(); m.map.repeat.set(w / 5, d / 5); m.map.needsUpdate = true;
    this.plane(m, x, z, w, d, .004);
    // Flush curbs keep the entire public circulation plane at y=0.
    this.box(this.group, this.white, x, -.01, z - d / 2 + .12, w, .035, .24);
    this.box(this.group, this.white, x - w / 2 + .12, -.01, z, .24, .035, d);
  }
  zebra(ax: number, az: number, bx: number, bz: number, width: number) {
    const dx = bx - ax, dz = bz - az, length = Math.hypot(dx, dz), angle = Math.atan2(dx, dz);
    for (let t = .65; t < length - .5; t += 1.6) this.paintStrips.push({
      x: ax + dx*t/length, z: az + dz*t/length, width, angle });
  }
  tactile(ax: number, az: number, bx: number, bz: number, width = .38) {
    const mat = material("#b8a568"); const dx = bx - ax, dz = bz - az, length = Math.hypot(dx, dz);
    this.box(this.group, mat, (ax + bx) / 2, .015, (az + bz) / 2, width, .025, length, Math.atan2(dx, dz));
  }
  lamp(x: number, z: number, angle = 0, tall = false) {
    const p = this.local(x, z, angle); const h = tall ? 8.4 : 5.7;
    this.cyl(p, this.dark, 0, h / 2, 0, .085, h); this.cyl(p, this.metal, 0, .22, 0, .16, .44);
    this.beam(p, this.dark, new THREE.Vector3(0, h, 0), new THREE.Vector3(1.5, h + .2, 0), .075);
    this.box(p, this.dark, 1.3, h + .17, 0, .75, .15, .38); this.box(p, this.warm, 1.3, h + .07, 0, .65, .035, .28);
  }
  bench(x: number, z: number, angle = 0) {
    const g = this.local(x, z, angle);
    for (const a of [-.9, .9]) { this.box(g, this.dark, a, .26, 0, .08, .5, .52); this.box(g, this.dark, a, .72, -.23, .06, .68, .06); }
    for (let i = 0; i < 4; i++) this.box(g, this.wood, 0, .51, -.24 + i * .15, 2.3, .055, .11);
    for (let i = 0; i < 3; i++) this.box(g, this.wood, 0, .74 + i * .15, -.27, 2.3, .1, .055);
  }
  tree(x: number, z: number, size = 1) {
    this.box(this.group, this.concrete, x, .16, z, 2.4 * size, .32, 2.4 * size);
    this.box(this.group, material("#4f5142"), x, .327, z, 2.05 * size, .014, 2.05 * size);
    this.cyl(this.group, material("#665b4b"), x, 1.8 * size, z, .16 * size, 3.6 * size);
    for (let i = 0; i < 7; i++) {
      const a = i * 2.4, y = 3.4 + (i % 3) * .57;
      this.mesh(this.group, this.SPHERE, this.leaf[i % 3], x + Math.cos(a) * .85 * size,
        y * size, z + Math.sin(a) * .8 * size, 1.45 * size, 1.2 * size, 1.3 * size);
    }
  }
  bollard(x: number, z: number) { this.cyl(this.group, this.dark, x, .4, z, .09, .8); this.cyl(this.group, this.white, x, .64, z, .095, .06); }
  signal(x: number, z: number, rotation: number) {
    const g = this.local(x, z, rotation);
    this.cyl(g, this.dark, 0, 2.7, 0, .09, 5.4); this.box(g, this.dark, 1.7, 5.3, 0, 3.5, .09, .09);
    this.box(g, this.dark, 3.2, 4.92, 0, 1.18, .4, .3);
    for (let i = 0; i < 3; i++) {
      const m = material(i === 0 ? "#bb473e" : i === 1 ? "#776b44" : "#41675c", .3, .1, i === 0 ? "#dc4437" : undefined);
      const disc = this.cyl(g, m, 2.8 + i * .39, 4.92, .175, .115, .025); disc.rotation.x = Math.PI / 2;
      this.box(g, this.dark, 2.8 + i * .39, 5.075, .24, .3, .06, .3);
    }
    this.box(g, this.dark, 0, 2.2, .16, .38, .64, .28);
    this.box(g, material("#85d3a0", .4, 0, "#68c688"), 0, 2.12, .312, .15, .2, .018);
    this.box(g, this.metal, .12, 1.05, .12, .18, .22, .16);
  }
  roof(g: THREE.Object3D, w: number, d: number, h: number) {
    this.box(g, this.concrete, 0, h + .15, 0, w + .3, .3, d + .3);
    for (const x of [-w / 2, w / 2]) this.box(g, this.accents[0], x, h + .55, 0, .18, .8, d);
    for (const z of [-d / 2, d / 2]) this.box(g, this.accents[0], 0, h + .55, z, w, .8, .18);
    for (let i = 0; i < Math.max(1, Math.floor(w / 8)); i++) {
      const x = -w / 3 + i * 4.5;
      this.box(g, this.metal, x, h + .8, -2, 2.4, 1.3, 2);
      this.cyl(g, this.dark, x, h + 1.48, -2, .66, .08);
      for (let j = 0; j < 6; j++) this.box(g, this.dark, x, h + .35 + j * .17, -.98, 2.15, .055, .025);
      this.box(g, this.metal, x + .8, h + .38, 1, .8, .4, 3);
    }
    this.cyl(g, this.metal, w * .22, h + 1.7, -d * .25, .045, 3.4);
  }
  shop(g: THREE.Object3D, x: number, front: number, w: number, name: string, index: number, open: boolean) {
    const colors = ["#294341", "#714b3f", "#a28b68", "#33485b", "#674e59"];
    const facade = front + .15;
    this.box(g, this.dark, x - w / 2 + .09, 1.85, facade, .18, 3.7, .28);
    this.box(g, this.dark, x + w / 2 - .09, 1.85, facade, .18, 3.7, .28);
    this.box(g, this.dark, x, 3.52, facade, w, .17, .3);
    this.text(g, name, x, 3.03, facade + .2, w - .35, .68, colors[index % 5], "#f1ead9", "");
    if (!open) {
      this.box(g, this.glassDark, x, 1.36, facade -.035, w - .25, 2.65, .07);
      for (let j = -1; j <= 1; j++) this.box(g, this.metal, x + j * w / 3, 1.36, facade + .03, .065, 2.65, .08);
      this.box(g, this.warm, x, 2.4, facade + .01, w - .4, .13, .08);
      // An inset shelf display gives depth instead of a flat painted storefront.
      for (let j = 0; j < 4; j++) this.box(g, this.wood, x - w / 2 + .6 + j * (w - 1.2) / 3, .65, facade + .06, .36, .7 + (j % 2) * .3, .18);
      this.box(g, this.metal, x + .3, 1.2, facade + .1, .035, .45, .06);
    } else {
      // Doorway is entirely open and flush: no collision or opaque front wall.
      const depth = 7;
      this.box(g, this.wood, x, -.06, front - depth / 2, w - .25, .1, depth);
      this.box(g, material("#d5c8b2"), x, 1.9, front - depth, w, 3.8, .18);
      for (const side of [-1, 1]) this.box(g, this.concrete, x + side * (w / 2 - .13), 1.9, front - depth / 2, .18, 3.8, depth);
      this.box(g, this.white, x, 3.65, front - depth / 2, w, .15, depth);
      this.box(g, this.warm, x, 3.55, front - depth / 2, w - .7, .05, .14);
      this.text(g, name.includes("BOOK") ? "READ / REST / REPEAT" : "GOOD DAYS BEGIN HERE", x, 2.5, front - depth + .12, w - 1, .65, "#d5c8b2", "#414d45");
      if (name.includes("BOOK")) this.bookshop(g, x, front, w);
      else this.cafe(g, x, front, w);
      this.box(g, this.dark, x, 2.68, facade, w - .3, .08, .12);
      // Framed side windows preserve the central 2.2 m portal.
      for (const side of [-1, 1]) {
        const sideWidth = Math.max(.3, (w - 2.3) / 2);
        const glass = new THREE.MeshPhysicalMaterial({ color: "#a9c0bb", transparent: true, opacity: .2, roughness: .12, metalness: .05, depthWrite: false });
        this.box(g, glass, x + side * (1.15 + sideWidth / 2), 1.28, facade, sideWidth, 2.5, .04);
        this.box(g, this.metal, x + side * 1.15, 1.3, facade, .07, 2.6, .09);
      }
    }
    const awning = this.box(g, material(colors[index % 5]), x, 2.64, facade + .55, w - .18, .08, 1.15);
    awning.rotation.x = .13;
    this.box(g, material(colors[index % 5]), x, 2.51, facade + 1.1, w - .18, .22, .04);
  }
  cafe(g: THREE.Object3D, x: number, front: number, w: number) {
    this.box(g, this.wood, x, .57, front - 5.7, w - 1.2, 1.14, .9);
    this.box(g, this.white, x, 1.17, front - 5.7, w - 1, .1, 1.05);
    this.box(g, this.metal, x + w / 4, 1.45, front - 5.8, .85, .5, .5);
    for (let i = 0; i < 4; i++) this.cyl(g, this.white, x - .9 + i * .22, 1.31, front - 5.45, .06, .16);
    for (const side of [-1, 1]) for (let row = 0; row < 2; row++) {
      const tx = x + side * w * .28, tz = front - 1.8 - row * 2;
      this.cyl(g, this.dark, tx, .36, tz, .045, .72); this.cyl(g, this.wood, tx, .75, tz, .55, .07);
      for (const offset of [-.72, .72]) {
        this.box(g, this.wood, tx, .44, tz + offset, .44, .065, .44);
        this.box(g, this.wood, tx, .75, tz + offset + Math.sign(offset) * .2, .44, .5, .055);
        for (const a of [-.16, .16]) for (const b of [-.16, .16]) this.box(g, this.dark, tx + a, .21, tz + offset + b, .028, .42, .028);
      }
      this.cyl(g, this.white, tx + .1, .835, tz, .065, .12);
      this.cyl(g, this.dark, tx, 3.15, tz, .018, .65); this.cyl(g, this.warm, tx, 2.82, tz, .22, .13);
    }
  }
  bookshop(g: THREE.Object3D, x: number, front: number, w: number) {
    const covers = [material("#496468"), material("#b46c4b"), material("#d2b16d"), material("#6d655d"), material("#849583")];
    for (const side of [-1, 1]) {
      const sx = x + side * (w / 2 - .55);
      this.box(g, this.wood, sx, 1.15, front - 4.1, .72, 2.3, 4.8);
      for (let shelf = 0; shelf < 4; shelf++) {
        this.box(g, this.dark, sx - side * .38, .45 + shelf * .49, front - 4.1, .035, .36, 4.5);
        for (let j = 0; j < 22; j++) this.box(g, covers[j % 5], sx - side * .43, .46 + shelf * .49, front - 6.15 + j * .19, .09, .26 + random() * .12, .12);
      }
    }
    this.box(g, this.wood, x, 1.12, front - 6.7, w - 1, 2.24, .32);
    for(let shelf=0;shelf<4;shelf++){
      this.box(g,this.wood,x,.2+shelf*.51,front-6.35,w-1,.065,.5);
      const count=Math.floor((w-1.4)/.23);
      for(let i=0;i<count;i++) this.box(g,covers[(i+shelf*2)%covers.length],x-(w-1.4)/2+i*.23,.43+shelf*.51,front-6.32,.16,.32+random()*.09,.26);
    }
    this.box(g, this.wood, x, .67, front - 3.1, 1.2, .1, 1.8);
    for (const offset of [-.45, .45]) this.box(g, this.dark, x + offset, .33, front - 3.1, .06, .66, 1.45);
    for (let i = 0; i < 5; i++) this.box(g, covers[i], x + (i % 2 ? .28 : -.28), .77, front - 3.6 + i * .25, .36, .12, .27);
  }
  building(x: number, z: number, w: number, d: number, floors: number, rotation: number, index: number,
    names: string[] = [], open = false) {
    const g = this.local(x, z, rotation), h = floors * 3.25 + 4, front = d / 2;
    const shell = this.accents[index % this.accents.length];
    this.box(g, shell, 0, (h + 3.7) / 2, 0, w, h - 3.7, d);
    if (!open) this.box(g, shell, 0, 1.8, -.2, w, 3.6, d - .4);
    const bays = Math.max(2, Math.floor(w / 2.8));
    for (let floor = 0; floor < floors; floor++) {
      const yy = 5.7 + floor * 3.25;
      this.box(g, this.concrete, 0, yy - 1.5, front + .13, w + .18, .18, .38);
      for (let bay = 0; bay < bays; bay++) {
        const xx = -w / 2 + (bay + .5) * w / bays;
        const ww = w / bays - .5;
        this.box(g, this.dark, xx, yy, front + .035, ww + .12, 2.22, .11);
        this.box(g, this.windows[Math.floor(random() * 4.6)], xx, yy, front + .098, ww, 2.08, .025);
        this.box(g, this.metal, xx, yy, front + .13, .045, 2.1, .045);
        this.box(g, this.metal, xx, yy -.27, front + .13, ww, .045, .045);
        this.box(g, this.concrete, xx, yy - 1.13, front + .22, ww + .22, .12, .5);
        if (index % 4 === 1 && floor > 0) {
          this.box(g, this.dark, xx, yy - .8, front + .58, ww + .2, .055, .055);
          for (const offset of [-.4, 0, .4]) this.box(g, this.metal, xx + offset * ww, yy - 1.0, front + .58, .035, .42, .035);
        }
      }
      // Side elevations are modeled as carefully as the street facade.
      for (const side of [-1, 1]) for (let bay = 0; bay < Math.floor(d / 3); bay++) {
        const zz = -d / 2 + (bay + .5) * d / Math.floor(d / 3);
        this.box(g, this.dark, side * (w / 2 + .025), yy, zz, .08, 2.1, 1.75);
        this.box(g, this.windows[(bay + floor + index) % 4], side * (w / 2 + .074), yy, zz, .025, 1.97, 1.62);
        this.box(g, this.concrete, side * (w / 2 + .19), yy - 1.13, zz, .4, .12, 1.97);
      }
    }
    const shops = open && names.length ? names.length : Math.max(1, Math.floor(w / 6.5));
    for (let i = 0; i < shops; i++) this.shop(g, -w / 2 + (i + .5) * w / shops, front, w / shops,
      names[i % Math.max(1, names.length)] || ["MORI MARKET", "STUDIO NINE", "KOMA COFFEE", "DAILY GOODS", "NORTH RECORDS"][index % 5], index + i, open);
    this.roof(g, w, d, h);
    // Exposed services, hanging sign and inset building entrance.
    this.box(g, this.metal, w / 2 - .15, h / 2, front + .08, .08, h - 1, .1);
    if (index % 3 === 0) {
      const blade = new THREE.Group(); blade.position.set(w / 2 + .25, 8, front -.15); blade.rotation.y = Math.PI / 2; g.add(blade);
      this.text(blade, ["KOMA", "NORTH", "HIKARI", "SORA"][index % 4], 0, 0, 0, 1.8, 5, "#e4ddc7", "#344447", "OPEN DAILY");
    }
    return g;
  }
  finish() {
    // Merge all crossing paint in a single mask; overlapping meshes shimmer at overhead distances.
    if(this.paintStrips.length){
      const minX=Math.min(...this.paintStrips.map(p=>p.x-p.width)),maxX=Math.max(...this.paintStrips.map(p=>p.x+p.width));
      const minZ=Math.min(...this.paintStrips.map(p=>p.z-p.width)),maxZ=Math.max(...this.paintStrips.map(p=>p.z+p.width));
      const width=maxX-minX,depth=maxZ-minZ;
      const map=canvasTexture(2048,2048,ctx=>{
        ctx.clearRect(0,0,2048,2048);ctx.scale(2048/width,2048/depth);ctx.translate(-minX,-minZ);ctx.fillStyle='#e9e6d7';
        for(const p of this.paintStrips){ctx.save();ctx.translate(p.x,p.z);ctx.rotate(-p.angle);ctx.fillRect(-p.width/2,-.39,p.width,.78);ctx.restore();}
      });
      const paint=new THREE.MeshStandardMaterial({map,roughness:.95,transparent:true,alphaTest:.2,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-4,polygonOffsetUnits:-4});
      const decal=this.plane(paint,(minX+maxX)/2,(minZ+maxZ)/2,width,depth,.009);decal.renderOrder=1;
    }
    // Consolidate repeated solid geometry into GPU instances. Custom artwork stays separate.
    this.group.updateMatrixWorld(true);
    const batches = new Map<string, { geometry: THREE.BufferGeometry; material: THREE.Material; matrices: THREE.Matrix4[] }>();
    const remove: THREE.Mesh[] = [];
    this.group.traverse(object => {
      if (!(object instanceof THREE.Mesh) || (object.geometry !== this.BOX && object.geometry !== this.CYLINDER && object.geometry !== this.SPHERE)) return;
      if (Array.isArray(object.material) || object.material.transparent) return;
      const key = `${object.geometry.uuid}/${object.material.uuid}`;
      let batch = batches.get(key); if (!batch) { batch = { geometry: object.geometry, material: object.material, matrices: [] }; batches.set(key, batch); }
      batch.matrices.push(object.matrixWorld.clone()); remove.push(object);
    });
    remove.forEach(mesh => mesh.removeFromParent());
    for (const { geometry, material: mat, matrices } of batches.values()) {
      const instances = new THREE.InstancedMesh(geometry, mat, matrices.length);
      matrices.forEach((matrix, i) => instances.setMatrixAt(i, matrix));
      instances.castShadow = true; instances.receiveShadow = true; instances.computeBoundingSphere();
      this.group.add(instances); this.instancedCount += matrices.length;
    }
  }
}

function ground(b: Builder, size = 1000) {
  b.plane(surface("asphalt", size / 5), 0, 0, size, size, -.018);
  // Recessed storm grates and manhole covers away from pedestrian routes.
  for (let k = -3; k <= 3; k++) for (const sign of [-1, 1]) {
    b.box(b.group, b.dark, k * 29 + 6, -.005, sign * 21.25, .8, .025, .38);
    for (let i = 0; i < 6; i++) b.box(b.group, b.metal, k * 29 + 5.68 + i * .13, .01, sign * 21.25, .04, .025, .32);
  }
}
function roadLines(b: Builder, axis: "x" | "z", start: number, end: number, offset = 0) {
  const paint = material("#cac8b8");
  for (let i = start; i < end; i += 7) {
    if (axis === "z") b.box(b.group, paint, offset, .008, i, .14, .016, 3);
    else b.box(b.group, paint, i, .008, offset, 3, .016, .14);
  }
}
function baseMetadata(title: string, description: string): EnvironmentResult["metadata"] {
  return { title, description, units: "metres; Y up; public ground Y=0",
    provenance: "Original procedural geometry and original canvas graphics authored in EnvironmentScene.ts. No scanned, generated or third-party geometry. Advertising photographs and upper-floor window detail use original AI-generated raster atlases; other surfaces and signs use original canvas graphics. Upper-floor glazing detail is a baked reflection/interior texture, not simulated rooms. All business names and advertisements are fictional.",
    bounds: { min: [-130, 0, -130], max: [130, 64, 130] }, walkableRects: [], spawnExitGates: [], interactionAnchors: [], clearWidths: {},
    limitations: ["Fictional architectural study inspired by Japanese urban districts; not a survey or reconstruction of a real place.",
      "Visual environment only: no actors, traffic, crowd generation, navigation mesh, collision physics or automatic door behavior.",
      "Walkable rectangles and gates are broad planning regions and do not subtract furniture or building footprints; they are not a baked navigation solution. Public circulation is flat at Y=0; station steps are visual geometry alongside a level bypass."] };
}
function crossing(b: Builder) {
  ground(b);
  const shops = [["KOMA COFFEE", "SEN BOOKS"], ["NORTH RECORDS", "MORI MARKET"], ["FORM & FIELD", "NAMI TEA"], ["SORA OPTICS", "DAILY GOODS"]];
  let index = 0;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    b.pavement(sx * 74, sz * 74, 104, 104);
    const rot = sz === -1 ? 0 : Math.PI;
    // Unequal heights and setbacks make four individually legible corners.
    b.building(sx * 43, sz * 42, 24, 20, [7, 5, 6, 8][index], rot, index, shops[index]);
    for (let i = 0; i < 4; i++) {
      const width = [16, 20, 15, 21][i];
      b.building(sx * (65 + i * 19.5), sz * (38 + (i % 2) * 2), width, 21, 4 + ((i + index) % 5), rot, index + i + 1);
      b.building(sx * 39, sz * (67 + i * 20), 18, 22, 4 + ((i * 2 + index) % 4), sx < 0 ? Math.PI / 2 : -Math.PI / 2, index + i + 4);
    }
    for (let i = 0; i < 3; i++) {
      b.building(sx * (68 + i * 22), sz * 76, 20, 25, 7 + (i % 3), rot, index + i + 8);
    }
    b.signal(sx * 23.5, sz * 25.3, sz < 0 ? 0 : Math.PI);
    b.lamp(sx * 26.5, sz * 49, sx < 0 ? 0 : Math.PI, true);
    b.tree(sx * 27.7, sz * 65, 1.05); b.bench(sx * 28, sz * 60, sx < 0 ? Math.PI / 2 : -Math.PI / 2);
    for (let i = 0; i < 3; i++) b.bollard(sx * (25 + i * 2), sz * 24.2);
    b.tactile(sx * 24.8, sz * 30, sx * 24.8, sz * 110);
    b.tactile(sx * 30, sz * 24.8, sx * 110, sz * 24.8);
    index++;
  }
  // Low-cost distant street frontage extends the district deep into atmosphere.
  // These are actual three-dimensional masses and glazing, not background images.
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    b.pavement(sx * 228, sz * 74, 204, 104);
    b.pavement(sx * 74, sz * 228, 104, 204);
    for (let i = 0; i < 9; i++) for (const axis of [0, 1]) {
      const along = 147 + i * 23;
      const g = b.local(axis === 0 ? sx * along : sx * 40,
        axis === 0 ? sz * 40 : sz * along,
        axis === 0 ? (sz < 0 ? 0 : Math.PI) : (sx < 0 ? Math.PI / 2 : -Math.PI / 2));
      const h = 18 + ((i * 7 + (sx + sz + 2)) % 6) * 3.2;
      b.box(g, b.accents[i % 5], 0, h / 2, 0, 21.5, h, 24);
      b.box(g, b.concrete, 0, h + .18, 0, 22, .36, 24.4);
      for (let y = 5; y < h - 1; y += 3.2) {
        b.box(g, b.glass, 0, y, 12.06, 18, 1.9, .1);
        for (let x = -8; x <= 8; x += 2.7) b.box(g, b.metal, x, y, 12.15, .09, 1.95, .1);
        b.box(g, b.concrete, 0, y - 1.08, 12.16, 21.6, .16, .3);
      }
      b.box(g, b.dark, 0, 1.6, 12.08, 19, 2.8, .12);
      b.box(g, b.accents[(i + 2) % 5], 0, 3.2, 12.15, 21, .6, .2);
    }
  }
  for (const sign of [-1, 1]) for (const offset of [-9, 0, 9]) {
    roadLines(b, "x", sign < 0 ? -340 : 125, sign < 0 ? -125 : 340, offset);
    roadLines(b, "z", sign < 0 ? -340 : 125, sign < 0 ? -125 : 340, offset);
  }
  // Primary landmark: a tall rounded glazed corner, with a sculptural screen crown.
  const corner = b.local(-39, -38);
  const curved = new THREE.CylinderGeometry(12, 12, 27, 28, 1, false, 0, Math.PI / 2);
  const glass = b.mesh(corner, curved, material("#88a7ad", .32, .26), 0, 20.5, 0); glass.rotation.y = 0;
  for (let level = 0; level <= 8; level++) {
    const arc = new THREE.TorusGeometry(12.25, .14, 4, 28, Math.PI / 2);
    const ring = new THREE.Mesh(arc, b.metal); ring.rotation.x = Math.PI / 2; ring.position.y = 7 + level * 3.3; corner.add(ring);
  }
  for (let i = 0; i <= 12; i++) {
    const angle = i / 12 * Math.PI / 2;
    b.box(corner, b.metal, Math.sin(angle) * 12.17, 20.5, Math.cos(angle) * 12.17, .08, 27, .09, angle);
  }
  b.billboard(corner, 0, 0, 28.5, 12.3, 18, 12);
  b.text(corner, "KOMA", 0, 36, 10.2, 13, 2.5, "#e6dfcf", "#314b4a", "CROSSING DISTRICT");
  const east = b.local(43, -42);
  for (const x of [-7.4, 7.4]) b.box(east, b.metal, x, 27, 9.8, .2, 12, .2);
  b.billboard(east, 1, 0, 23, 11.1, 18, 12);
  b.text(east, "SORA", 0, 30, 11.1, 16, 3.5, "#384c59", "#e8e9dc", "ARTS & CULTURE");
  const south = b.local(43, 42, Math.PI);
  b.billboard(south, 2, 0, 20, 11.1, 15, 10);
  // Four conventional crossings and the two complete centre-spanning scramble paths.
  b.zebra(-22, -17.7, 22, -17.7, 5.3); b.zebra(-22, 17.7, 22, 17.7, 5.3);
  b.zebra(-17.7, -22, -17.7, 22, 5.3); b.zebra(17.7, -22, 17.7, 22, 5.3);
  b.zebra(-24, -24, 24, 24, 4.8); b.zebra(-24, 24, 24, -24, 4.8);
  for (const sign of [-1, 1]) {
    roadLines(b, "z", sign < 0 ? -125 : 31, sign < 0 ? -30 : 125);
    roadLines(b, "x", sign < 0 ? -125 : 31, sign < 0 ? -30 : 125);
    for (const offset of [-9, 9]) {
      roadLines(b, "z", sign < 0 ? -125 : 34, sign < 0 ? -32 : 125, offset);
      roadLines(b, "x", sign < 0 ? -125 : 34, sign < 0 ? -32 : 125, offset);
    }
    b.box(b.group, b.white, 0, .012, sign * 29, 20, .02, .5);
    b.box(b.group, b.white, sign * 29, .012, 0, .5, .02, 20);
  }
  const metadata = baseMetadata("Koma Crossing", "A broad fictional Tokyo-inspired scramble crossing framed by layered shopfronts, rounded glazing, billboards and continuous city streets.");
  metadata.bounds = { min: [-330, 0, -330], max: [330, 64, 330] };
  metadata.walkableRects = [{ id: "scramble", min: [-24, -24], max: [24, 24], y: 0 },
    ...[-1, 1].flatMap(s => [
      { id: `east-west-${s}`, min: [-125, s < 0 ? -30 : 22] as [number, number], max: [125, s < 0 ? -22 : 30] as [number, number], y: 0 },
      { id: `north-south-${s}`, min: [s < 0 ? -30 : 22, -125] as [number, number], max: [s < 0 ? -22 : 30, 125] as [number, number], y: 0 }])];
  metadata.spawnExitGates = [
    { id: "north", position: [25, 0, -115], width: 6, direction: [0, 0, 1] },
    { id: "south", position: [-25, 0, 115], width: 6, direction: [0, 0, -1] },
    { id: "east", position: [115, 0, 25], width: 6, direction: [-1, 0, 0] },
    { id: "west", position: [-115, 0, -25], width: 6, direction: [1, 0, 0] }];
  metadata.interactionAnchors = [{ id: "scramble-centre", label: "Crossing centre", position: [0, 0, 0] }, { id: "corner-plaza", label: "Koma corner", position: [-26, 0, -27] }];
  metadata.clearWidths = { roadway: 44, cornerSidewalk: 8, zebra: 5.3, diagonalZebra: 4.8 };
  const cameras: EnvironmentCamera[] = [
    { id: "hero", label: "Crossing · hero", position: [7, 23, 61], target: [-1, 6, -9], fov: 57 },
    { id: "street", label: "Scramble · eye level", position: [22, 1.72, 25], target: [-19, 9, -27], fov: 66 },
    { id: "centre", label: "In the crossing", position: [0, 1.7, 3], target: [-33, 14, -37], fov: 71 },
    { id: "overhead", label: "Crossing geometry", position: [1, 110, 14], target: [0, 0, 0], fov: 48 },
    { id: "corner", label: "Corner shops", position: [4, 1.7, -26], target: [-39, 4.3, -29], fov: 62 },
    { id: "avenue", label: "Along the avenue", position: [-71, 2.2, 24], target: [23, 8, -26], fov: 55 },
  ];
  return { metadata, cameras };
}

function city(b: Builder) {
  ground(b, 1000);
  // Connected east-west high street, north-south side street, and a rear courtyard.
  b.pavement(0, -30, 180, 36); b.pavement(0, 30, 180, 36);

  b.building(-23, -25, 24, 16, 4, 0, 0, ["KOMA COFFEE", "SEN BOOKS"], true);
  b.building(-51, -26, 22, 18, 5, 0, 2, ["FORM & FIELD", "NAMI TEA"]);
  b.building(-73, -25, 18, 16, 3, 0, 1, ["MORI MARKET"]);
  b.building(21, -26, 19, 18, 6, 0, 3, ["NORTH RECORDS", "STUDIO NINE"]);
  b.building(44, -26, 22, 18, 4, 0, 4, ["SORA OPTICS", "DAILY GOODS"]);
  b.building(69, -27, 24, 20, 5, 0, 1, ["ATELIER 04", "MORI FLOWERS"]);
  for (let i = 0; i < 6; i++) b.building(-67 + i * 26, 26 + (i % 2), 23, 17, 3 + i % 4, Math.PI, i + 1,
    [["KOMA BAKERY", "THREAD & NEEDLE"], ["QUIET CORNER", "SORA GOODS"], ["MORI MARKET", "NAMI TEA"]][i % 3]);
  b.pavement(0, -62, 23, 82);
  for (const sign of [-1, 1]) for (let i = 0; i < 3; i++) b.building(sign * 22, -52 - i * 22, 19, 18,
    3 + i, sign < 0 ? Math.PI / 2 : -Math.PI / 2, i + 5);
  // A genuinely open courtyard reached by a 7 m alley, with planted edges.
  b.pavement(-48, -57, 42, 25);
  b.building(-49, -80, 35, 16, 3, 0, 2, ["COURTYARD STUDIOS", "THE PAPER ROOM"]);
  for (const x of [-62, -35]) { b.tree(x, -56, .95); b.bench(x + 2.6, -56, Math.PI / 2); }
  for (let x = -70; x <= 75; x += 24) {
    b.lamp(x, -13.3); b.lamp(x + 7, 13.3, Math.PI);
    b.tree(x + 8, -14.3, .8); b.bench(x + 4.8, -14.4);
  }
  // Menus and bicycle racks sit in furnishing bands, clear of the shop entrances.
  const menu = b.local(-33, -15.5, .12);
  b.box(menu, b.wood, 0, .66, 0, .72, 1.3, .1);
  b.text(menu, "COFFEE / TEA", 0, .72, .066, .61, .96, "#344642", "#ece4ca", "BAKED HERE DAILY");
  for (let i = 0; i < 5; i++) {
    const x = 31 + i * 1.15;
    b.beam(b.group, b.metal, new THREE.Vector3(x, 0, -14), new THREE.Vector3(x, .8, -14), .05);
    b.box(b.group, b.metal, x + .32, .8, -14, .7, .05, .05);
    b.box(b.group, b.metal, x + .64, .4, -14, .05, .8, .05);
  }
  roadLines(b, "x", -105, 105);
  b.zebra(-6.5, -12, -6.5, 12, 4); b.zebra(7, -12, 7, 12, 4);
  b.tactile(-95, -13, 95, -13); b.tactile(-95, 13, 95, 13);
  const metadata = baseMetadata("Mori High Street", "A connected neighborhood street with open furnished coffee shop and bookshop, a side street and a quiet planted rear courtyard.");
  metadata.bounds = { min: [-115, 0, -115], max: [115, 35, 115] };
  metadata.walkableRects = [
    { id: "street", min: [-100, -16], max: [100, 16], y: 0 },
    { id: "side-street", min: [-11.5, -102], max: [11.5, 12], y: 0 },
    { id: "cafe", min: [-34.5, -24], max: [-23.5, -16], y: 0 },
    { id: "bookshop", min: [-22.5, -24], max: [-11.5, -16], y: 0 },
    { id: "courtyard-alley", min: [-40, -66], max: [-35, -12], y: 0 },
    { id: "courtyard", min: [-67, -66], max: [-29, -46], y: 0 }];
  metadata.spawnExitGates = [
    { id: "west", position: [-100, 0, -13.5], width: 5, direction: [1, 0, 0] },
    { id: "east", position: [100, 0, 13.5], width: 5, direction: [-1, 0, 0] },
    { id: "north", position: [0, 0, -100], width: 10, direction: [0, 0, 1] }];
  metadata.interactionAnchors = [
    { id: "cafe", label: "Koma Coffee counter", position: [-29, 0, -21.6] },
    { id: "books", label: "Sen Books display", position: [-17, 0, -20.1] },
    { id: "courtyard", label: "Mori courtyard", position: [-49, 0, -55] }];
  metadata.clearWidths = { mainRoad: 24, sidewalk: 5, shopDoorway: 2.3, sideStreet: 23, courtyardAlley: 5 };
  const cameras: EnvironmentCamera[] = [
    { id: "hero", label: "High street · hero", position: [6, 13, 10], target: [-25, 5, -25], fov: 62 },
    { id: "street", label: "Neighborhood walk", position: [-2, 1.7, -8], target: [-29, 2.5, -19], fov: 65 },
    { id: "cafe", label: "Inside Koma Coffee", position: [-29, 1.65, -18.3], target: [-29, 1.5, -23], fov: 79 },
    { id: "bookshop", label: "Inside Sen Books", position: [-20, 1.65, -18.1], target: [-15.4, 1.4, -22.2], fov: 70 },
    { id: "courtyard", label: "Rear courtyard", position: [-48, 3.2, -45], target: [-49, 3, -74], fov: 65 },
    { id: "overhead", label: "Street & courtyard plan", position: [-14, 99, 13], target: [-18, 0, -30], fov: 58 },
  ];
  return { metadata, cameras };
}

function station(b: Builder) {
  ground(b, 1000);
  b.pavement(0, -13, 118, 108);
  const station = b.local(0, -47);
  const glazing = new THREE.MeshPhysicalMaterial({ color: "#9bb9bc", roughness: .14, metalness: .12,
    transparent: true, opacity: .24, depthWrite: false, side: THREE.DoubleSide });
  const steel = material("#aeb9b9", .36, .7);
  // Grand glazed hall: transparent facade, real frame, roof trusses and a visible concourse.
  b.box(station, b.concrete, 0, -.18, 0, 82, .32, 33);
  b.box(station, b.accents[1], 0, 6.8, -16, 82, 13.6, .6);
  for (let x = -40; x <= 40; x += 5) {
    b.box(station, steel, x, 6.7, 15, .2, 13.4, .25);
    b.box(station, steel, x, 6.7, -15, .2, 13.4, .25);
    b.beam(station, steel, new THREE.Vector3(x, 12, -16), new THREE.Vector3(x, 16.2, 0), .24);
    b.beam(station, steel, new THREE.Vector3(x, 16.2, 0), new THREE.Vector3(x, 12, 16), .24);
    b.beam(station, steel, new THREE.Vector3(x, 11, -15), new THREE.Vector3(x, 11, 15), .16);
    for (let z = -12; z <= 12; z += 6) b.beam(station, steel, new THREE.Vector3(x, 11, z), new THREE.Vector3(x, 16 - Math.abs(z) * .25, z + 3), .1);
    if (Math.abs(x) > 9) b.box(station, glazing, x + 2.5, 6.7, 15.02, 4.8, 12.8, .08);
  }
  for (let y = 3.6; y < 14; y += 3.2) b.box(station, steel, 0, y, 15.1, 82, .14, .2);
  for (const side of [-1, 1]) {
    const roof = b.box(station, glazing, 0, 14.1, side * 8, 83, .13, 16.55); roof.rotation.x = side * .255;
    b.box(station, steel, 0, 12, side * 16, 84, .23, .24);
    b.box(station, steel, 0, 16.2, 0, 84, .2, .2);
  }
  b.text(station, "HIKARI STATION", 0, 10.2, 15.35, 24, 2, "#234044", "#f0ead7", "CITY RAIL / EAST CONCOURSE");
  b.text(station, "01—04   ↑   PLATFORMS", 0, 4.3, 6.6, 12, 1.1, "#243d40", "#efead4", "TICKETS   •   CITY EXIT");
  for (const side of [-1, 1]) {
    b.box(station, b.concrete, side * 40.5, 6, 0, .5, 12, 32);
    for (let i = 0; i < 5; i++) {
      b.box(station, b.dark, side * (12 + i * 3), .62, 2, .55, 1.24, 1.3);
      b.box(station, b.warm, side * (12 + i * 3), 1.3, 2, .38, .07, .7);
      b.box(station, b.metal, side * (12 + i * 3), 1.6, 2.15, .25, .38, .07);
    }
    for (let i = 0; i < 4; i++) {
      b.box(station, b.dark, side * 31, 1.1, -9 + i * 2, 1.4, 2.2, .6);
      b.box(station, b.warm, side * 31, 1.5, -8.65 + i * 2, .9, .8, .04);
    }
  }
  // Extended lightweight entry canopy and its V-shaped column structure.
  b.box(b.group, steel, 0, 6.25, -23, 61, .18, 21);
  for (let x = -28; x <= 28; x += 7) {
    if (Math.abs(x) >= 14) b.beam(b.group, steel, new THREE.Vector3(x, 0, -18), new THREE.Vector3(x - 1.7, 6.2, -18), .16);
    if (Math.abs(x) >= 14) b.beam(b.group, steel, new THREE.Vector3(x, 0, -18), new THREE.Vector3(x + 1.7, 6.2, -18), .16);
    b.box(b.group, b.dark, x, 6.13, -23, .12, .18, 20);
    b.box(b.group, b.warm, x, 6.02, -23, .12, .035, 14);
  }
  // Decorative raised terrace and stairs live beside an unobstructed 18 m level route.
  for (const side of [-1, 1]) {
    b.box(b.group, b.concrete, side * 41, .75, -11, 15, 1.5, 15);
    for (let step = 0; step < 6; step++) b.box(b.group, b.concrete, side * 41, .125 + step * .125, -.4 - step * .5, 15, .25 + step * .25, .5);
    for (let i = 0; i < 3; i++) b.tree(side * (32 + i * 10), 15, .95);
    for (let i = 0; i < 4; i++) b.bench(side * (19 + i * 10), 17, Math.PI);
    b.lamp(side * 13, 4, side < 0 ? 0 : Math.PI); b.lamp(side * 52, -3, side < 0 ? 0 : Math.PI);
    b.building(side * 77, -44, 27, 32, 6, side < 0 ? Math.PI / 2 : -Math.PI / 2, side < 0 ? 2 : 4, ["HIKARI HOTEL", "CITY ROOMS"]);
    b.building(side * 82, 0, 32, 25, 5, side < 0 ? Math.PI / 2 : -Math.PI / 2, side < 0 ? 1 : 3, ["FORM & FIELD", "NORTH RECORDS"]);
    b.building(side * 69, 72, 33, 24, 5, Math.PI, side < 0 ? 0 : 3);
  }
  // Clock pavilion is visible from both plaza and approach.
  b.cyl(b.group, b.dark, 11, 3.9, 9, .14, 7.8);
  const clock = b.local(11, 9);
  const clockTex = canvasTexture(512, 512, ctx => {
    ctx.fillStyle = "#e9e6d7"; ctx.beginPath(); ctx.arc(256, 256, 247, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#33484a"; ctx.lineWidth = 10;
    for (let i = 0; i < 12; i++) { const a = i / 12 * Math.PI * 2; ctx.beginPath(); ctx.moveTo(256 + Math.sin(a) * 208, 256 + Math.cos(a) * 208); ctx.lineTo(256 + Math.sin(a) * 228, 256 + Math.cos(a) * 228); ctx.stroke(); }
    ctx.lineWidth = 14; ctx.beginPath(); ctx.moveTo(256, 256); ctx.lineTo(170, 194); ctx.moveTo(256, 256); ctx.lineTo(352, 142); ctx.stroke();
  });
  const dial = new THREE.Mesh(new THREE.CircleGeometry(.88, 48), new THREE.MeshStandardMaterial({ map: clockTex, emissiveMap: clockTex, emissive: "#ffffff", emissiveIntensity: .2 }));
  dial.position.set(0, 7.8, .13); clock.add(dial); const back = dial.clone(); back.rotation.y = Math.PI; back.position.z = -.13; clock.add(back);
  // Compact original kiosks with open service counters.
  for (const side of [-1, 1]) {
    const kiosk = b.local(side * 22, 4, side < 0 ? .25 : -.25);
    b.box(kiosk, b.wood, 0, 1.25, 0, 5, 2.5, 3.2);
    b.box(kiosk, b.dark, 0, 1.63, 1.64, 4.45, 1.15, .04);
    b.box(kiosk, b.white, 0, 1.05, 1.9, 4.7, .12, .65);
    b.box(kiosk, b.dark, 0, 2.75, .2, 5.7, .18, 4);
    b.text(kiosk, side < 0 ? "KOMA EXPRESS" : "MORI FLOWERS", 0, 2.35, 1.8, 4.6, .45, "#2e4747", "#ede6d3");
  }
  // Arrival road and clearly outlined, empty bus bays.
  b.pavement(0, 58, 114, 10);
  for (let i = -2; i <= 2; i++) {
    b.box(b.group, b.white, i * 20, .015, 40, 17.5, .025, .16);
    b.box(b.group, b.white, i * 20 - 8.7, .015, 44, .16, .025, 8);
    const shelter = b.local(i * 20, 56, Math.PI);
    for (const x of [-3, 3]) b.box(shelter, steel, x, 1.55, 0, .1, 3.1, .1);
    b.box(shelter, glazing, 0, 1.6, -.2, 6, 2.8, .06);
    b.box(shelter, steel, 0, 3.1, .55, 6.6, .12, 2.2);
    b.text(shelter, `BUS ${i + 3}`, 0, 2.6, .2, 1.4, .55, "#243e42", "#efe8d5");
    b.bench(i * 20, 56, Math.PI);
  }
  b.zebra(0, 37, 0, 54, 6); b.tactile(0, 31, 0, -31, .5);
  for (const side of [-1, 1]) for (let i = 0; i < 6; i++) b.bollard(side * (6 + i * 8), 37);
  const metadata = baseMetadata("Hikari Station Plaza", "A dusk arrival plaza with a glazed rail hall, exposed roof trusses, deep entry canopy, kiosks, bus shelters and an uninterrupted level approach.");
  metadata.walkableRects = [
    { id: "plaza", min: [-58, -3], max: [58, 36], y: 0 },
    { id: "level-approach", min: [-9, -37], max: [9, 32], y: 0 },
    { id: "concourse", min: [-39, -61], max: [39, -32], y: 0 },
    { id: "bus-walk", min: [-55, 53], max: [55, 62], y: 0 }];
  metadata.spawnExitGates = [
    { id: "rail", position: [0, 0, -58], width: 16, direction: [0, 0, 1] },
    { id: "bus", position: [0, 0, 61], width: 6, direction: [0, 0, -1] },
    { id: "west", position: [-55, 0, 26], width: 8, direction: [1, 0, 0] },
    { id: "east", position: [55, 0, 26], width: 8, direction: [-1, 0, 0] }];
  metadata.interactionAnchors = [
    { id: "station", label: "East concourse", position: [0, 0, -42] },
    { id: "kiosk", label: "Koma Express", position: [-22, 0, 7] },
    { id: "bus", label: "Bus connection", position: [0, 0, 55] }];
  metadata.clearWidths = { levelApproach: 18, plaza: 116, mainEntrance: 18, busCrossing: 6 };
  const cameras: EnvironmentCamera[] = [
    { id: "hero", label: "Station · dusk arrival", position: [48, 24, 43], target: [0, 7, -27], fov: 59 },
    { id: "street", label: "Plaza · eye level", position: [9, 1.72, 32], target: [0, 8, -37], fov: 64 },
    { id: "canopy", label: "Under the canopy", position: [-8, 1.72, -13], target: [1, 6, -40], fov: 71 },
    { id: "hall", label: "Inside the concourse", position: [4, 1.72, -41], target: [-24, 8, -51], fov: 72 },
    { id: "bus", label: "Bus connections", position: [-40, 3.5, 66], target: [0, 3, 18], fov: 57 },
    { id: "overhead", label: "Station & plaza plan", position: [6, 118, 53], target: [0, 0, -10], fov: 58 },
  ];
  return { metadata, cameras };
}

export function buildEnvironment(kind: EnvironmentKind): EnvironmentResult {
  randomState = kind === "crossing" ? 239 : kind === "city" ? 827 : 1069;
  const b = new Builder(); b.group.name = `environment-${kind}`;
  const result = kind === "crossing" ? crossing(b) : kind === "city" ? city(b) : station(b);
  b.finish();
  b.group.userData.provenance = result.metadata.provenance;
  b.group.userData.instancedObjectCount = b.instancedCount;
  return { group: b.group, ...result,
    lighting: kind === "station"
      ? { background: "#89969f", fogNear: 130, fogFar: 310, sun: [-70, 38, 65], sunColor: "#ffd0a1", sunIntensity: 2.6, ambientIntensity: 1.15, exposure: 1.05 }
      : { background: "#cbd8dc", fogNear: 115, fogFar: 310, sun: [-65, 100, 55], sunColor: "#fff2dc", sunIntensity: 3.0, ambientIntensity: 1.2, exposure: 1.0 } };
}
