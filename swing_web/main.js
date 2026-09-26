import * as THREE from '/vendor/three/build/three.module.js';

// The visible canvas is also the MediaRecorder source. Every frame composites
// the live WebGL scene with the controls and telemetry that the viewer sees.
const canvas = document.querySelector('#stage');
const ctx = canvas.getContext('2d', { alpha: false });
const container = document.querySelector('.stage-wrap');
const access = document.querySelector('#access-controls');
const loading = document.querySelector('#loading');
const loadingDetail = document.querySelector('#loading-detail');
const connection = document.querySelector('#connection');
const pulse = document.querySelector('.pulse');
const clock = new THREE.Clock();
const scene = new THREE.Scene();
scene.background = new THREE.Color('#172739');
scene.fog = new THREE.FogExp2('#183247', 0.016);
const camera = new THREE.PerspectiveCamera(48, 16 / 9, 0.05, 220);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.55;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.setAttribute('aria-hidden', 'true');
renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;opacity:0;pointer-events:none';
container.prepend(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xcfe7ff, 0x21324a, 2.4));
const sun = new THREE.DirectionalLight(0xffd3aa, 3.0);
sun.position.set(-8, 17, -15); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -30; sun.shadow.camera.right = 30;
sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30;
sun.shadow.camera.near = 1; sun.shadow.camera.far = 75;
sun.shadow.bias = -0.0003;
scene.add(sun);
const rim = new THREE.DirectionalLight(0x64c7ff, 1.8); rim.position.set(7, 7, 13); scene.add(rim);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(150, 150), new THREE.MeshStandardMaterial({ color: '#1a2933', roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.035; ground.receiveShadow = true; scene.add(ground);
const road = new THREE.Mesh(new THREE.PlaneGeometry(8.4, 44), new THREE.MeshStandardMaterial({ color: '#27323e', roughness: 1 }));
road.rotation.x = -Math.PI / 2; road.position.set(0, -0.015, -12); road.receiveShadow = true; scene.add(road);
const stripeMat = new THREE.MeshBasicMaterial({ color: '#ccbf91', transparent: true, opacity: .53 });
for (let z = -32; z < 10; z += 2.5) {
  const stripe = new THREE.Mesh(new THREE.PlaneGeometry(.075, 1.05), stripeMat);
  stripe.rotation.x = -Math.PI / 2; stripe.position.set(0, .003, z); scene.add(stripe);
}

const state = {
  sceneReady: false, rigReady: false, connected: false, snapshot: null,
  runId: null,
  carryAt: null,
  lastStateAt: 0, lastPollAt: 0, lastPaintedFrame: -1, ackedCommand: null,
  localStalls: 0, events: [], phase: 'READY', cameraMode: 'CINEMATIC',
  commandBusy: false, command: '', commandAt: 0, error: '',
  recorder: null, recordStart: 0, recordedChunks: [], recordingStatus: '',
  pointerX: 0, pointerY: 0, dragStart: null, orbitOffset: 0,
};
const actors = new Map();
const sceneMeshes = [];
let rig = null;
let currentAnchor = null;
let stageW = 1280, stageH = 720, pixelScale = 1;
const scratchV = new THREE.Vector3();
const desiredCamera = new THREE.Vector3(4.6, 3.6, 8.8);
const lookAt = new THREE.Vector3(0, 3.0, -8);
camera.position.copy(desiredCamera);

function resize() {
  stageW = Math.max(300, container.clientWidth); stageH = Math.max(300, container.clientHeight);
  pixelScale = Math.min(window.devicePixelRatio || 1, 1.5);
  canvas.width = Math.round(stageW * pixelScale); canvas.height = Math.round(stageH * pixelScale);
  canvas.style.width = `${stageW}px`; canvas.style.height = `${stageH}px`;
  ctx.setTransform(pixelScale, 0, 0, pixelScale, 0, 0);
  renderer.setSize(stageW, stageH, false);
  camera.aspect = stageW / stageH; camera.updateProjectionMatrix();
  layoutButtons();
}
new ResizeObserver(resize).observe(container);

function toColor(bytes, fallback = '#6f8192') {
  if (!Array.isArray(bytes) || bytes.length < 3) return new THREE.Color(fallback);
  return new THREE.Color().setRGB(bytes[0] / 255, bytes[1] / 255, bytes[2] / 255, THREE.SRGBColorSpace);
}
function setGeometry(doc) {
  for (const mesh of sceneMeshes) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  sceneMeshes.length = 0;
  const meshes = doc.meshes || [];
  if (!meshes.length) throw new Error('Scene endpoint returned no generated geometry');
  for (const part of meshes) {
    const raw = part.positions || part.vertices;
    const faces = part.indices || part.faces;
    if (!raw?.length || !faces?.length) continue;
    const positions = Array.isArray(raw[0]) ? raw.flat() : raw;
    const indices = Array.isArray(faces[0]) ? faces.flat() : faces;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({ color: toColor(part.color), roughness: .86, metalness: .04, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true; mesh.receiveShadow = true;
    scene.add(mesh); sceneMeshes.push(mesh);
  }
  state.sceneReady = sceneMeshes.length > 0;
}

function flatMatrix(row) {
  return Array.isArray(row?.[0]) ? row.flat() : row;
}
function prepareRig(doc) {
  const vertices = doc.bind_vertices;
  const faces = doc.faces;
  const indices = doc.lbs_indices;
  const weights = doc.lbs_weights;
  const inverses = doc.bind_rig_transform_inv;
  if (!vertices?.length || !faces?.length || !indices?.length || !weights?.length || !inverses?.length) {
    throw new Error('CoreSkin rig is missing required geometry or skin weights');
  }
  const vertexCount = vertices.length;
  const influences = indices[0].length;
  const flatVerts = new Float32Array(Array.isArray(vertices[0]) ? vertices.flat() : vertices);
  const flatJoint = new Uint8Array(Array.isArray(indices[0]) ? indices.flat() : indices);
  const flatWeights = new Float32Array(Array.isArray(weights[0]) ? weights.flat() : weights);
  const local = new Float32Array(vertexCount * influences * 3);
  for (let i = 0; i < vertexCount; i++) {
    const vx = flatVerts[3*i], vy = flatVerts[3*i+1], vz = flatVerts[3*i+2];
    for (let k = 0; k < influences; k++) {
      const j = flatJoint[i*influences+k];
      const m = flatMatrix(inverses[j]);
      const q = (i*influences+k)*3;
      local[q] = m[0]*vx+m[1]*vy+m[2]*vz+m[3];
      local[q+1] = m[4]*vx+m[5]*vy+m[6]*vz+m[7];
      local[q+2] = m[8]*vx+m[9]*vy+m[10]*vz+m[11];
    }
  }
  rig = { vertexCount, influences, vertices: flatVerts, joints: flatJoint, weights: flatWeights, local,
    faces: Array.isArray(faces[0]) ? faces.flat() : faces };
  state.rigReady = true;
  makeActor('spider'); makeActor('mj');
}

function costumeColor(which, v, joint) {
  const [x,y,z] = v;
  let rgb;
  if (which === 'spider') {
    if (joint === 6 || y > 1.58) rgb = [0.83,.045,.085];
    else if (joint >= 19 && joint <= 26) rgb = y < .29 ? [.82,.045,.09] : [.055,.20,.56];
    else if (joint >= 7 && joint <= 18) rgb = y > 1.27 || Math.abs(x) > .36 ? [.83,.045,.085] : [.06,.18,.48];
    else rgb = Math.abs(x) < .105 && z > -.10 ? [.85,.055,.092] : [.06,.20,.54];
    // Woven web pattern stays attached to the exact skinned surface.
    if (rgb[0] > .5 && y > .9 && y < 1.87) {
      const vertical = Math.abs(Math.sin(Math.atan2(x, z+.025)*8)) < .065;
      const rings = Math.abs(Math.sin(y*56)) < .07;
      if (vertical || rings) rgb = rgb.map(c => c*.44);
    }
  } else {
    if (joint === 6 || y > 1.61) rgb = [.88,.52,.39];
    else if (joint >= 19 && joint <= 26) rgb = y < .16 ? [.21,.13,.12] : [.16,.22,.38];
    else if (joint === 10 || joint === 16 || joint === 11 || joint === 17) rgb = [.88,.52,.39];
    else if (y > 1.02) rgb = Math.abs(x) < .08 && z > 0 ? [.97,.86,.72] : [.54,.15,.22];
    else rgb = [.18,.24,.39];
  }
  return rgb;
}

function makeActor(which) {
  const positions = new Float32Array(rig.vertexCount*3);
  const colors = new Float32Array(rig.vertexCount*3);
  const reveal=[];
  for (let i=0; i<rig.vertexCount; i++) {
    let best = 0, weight = -1;
    for (let k=0; k<rig.influences; k++) {
      const w=rig.weights[i*rig.influences+k];
      if (w>weight) { weight=w; best=rig.joints[i*rig.influences+k]; }
    }
    const bind=[rig.vertices[i*3],rig.vertices[i*3+1],rig.vertices[i*3+2]];
    colors.set(costumeColor(which, bind, best), i*3);
    if(which==='spider' && best===6 && bind[1]>1.63 && bind[1]<1.78 && bind[2]>-.01){
      const fade=Math.min(1,Math.max(0,(1.78-bind[1])/.08))*Math.min(1,Math.max(0,(bind[2]+.01)/.04));
      if(fade>.001)reveal.push([i,fade]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(rig.faces); geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ vertexColors:true, roughness:.82, metalness:.02, side:THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry,material); mesh.frustumCulled = false; mesh.castShadow = true; mesh.receiveShadow = true;
  scene.add(mesh);
  const face = new THREE.Group();
  const whites = new THREE.MeshStandardMaterial({color:which==='spider'?'#e9faff':'#f4eadc',roughness:.25,emissive:which==='spider'?'#82b6de':'#000000',emissiveIntensity:which==='spider'?.35:0});
  let mouth=null;const eyes=[];
  if (which === 'spider') {
    for (const side of [-1,1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(1,12,8), whites);
      eye.position.set(side*.052,.09,.151); eye.scale.set(.049,.025,.012); eye.rotation.z=side*.27; face.add(eye);
      eyes.push(eye);
    }
    mouth=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),new THREE.MeshStandardMaterial({color:'#a95657',roughness:.8}));
    mouth.position.set(0,.015,.151);mouth.scale.set(.024,.005,.004);mouth.visible=false;face.add(mouth);
  } else {
    const hairMat = new THREE.MeshStandardMaterial({color:'#54252a',roughness:1});
    const hair = new THREE.Mesh(new THREE.SphereGeometry(1,16,12),hairMat);
    hair.scale.set(.113,.112,.125); hair.position.set(0,.13,-.004); face.add(hair);
    const bangs=new THREE.Mesh(new THREE.SphereGeometry(1,16,10),hairMat);
    bangs.scale.set(.092,.031,.039);bangs.position.set(-.012,.15,.11);bangs.rotation.z=-.16;face.add(bangs);
    for (const side of [-1,1]) {
      const lock=new THREE.Mesh(new THREE.CapsuleGeometry(.035,.31,4,8),hairMat);
      lock.position.set(side*.094,-.142,-.028);lock.rotation.z=side*.16;face.add(lock);
    }
    const eyeMat=new THREE.MeshStandardMaterial({color:'#251b1e',roughness:.45});
    for(const side of [-1,1]){
      const eye=new THREE.Mesh(new THREE.SphereGeometry(1,10,8),eyeMat);
      eye.position.set(side*.038,.09,.151);eye.scale.set(.007,.008,.004);face.add(eye);
      eyes.push(eye);
    }
    mouth=new THREE.Mesh(new THREE.SphereGeometry(1,10,8),new THREE.MeshStandardMaterial({color:'#a34d56',roughness:.9}));
    mouth.position.set(0,.015,.151);mouth.scale.set(.025,.005,.004);face.add(mouth);
  }
  scene.add(face);
  let chest=null;
  if(which==='spider'){
    chest=new THREE.Group();
    const symbol=new THREE.Mesh(new THREE.SphereGeometry(1,10,8),new THREE.MeshStandardMaterial({color:'#0b1420'}));
    symbol.position.set(0,-.035,.185);symbol.scale.set(.034,.053,.008);chest.add(symbol);
    scene.add(chest);
  }
  actors.set(which, { mesh,face,chest,mouth,eyes,reveal,baseColors:colors.slice(),lastFrame:-1 });
}

function matrix3(row) {
  const flat = flatMatrix(row);
  return flat?.length === 9 ? flat : [1,0,0,0,1,0,0,0,1];
}
function poseActor(which, actor, frame) {
  const model = actors.get(which);
  if (!model || !actor?.positions?.length || !actor?.rotations?.length || model.lastFrame===frame) return;
  const positions = model.mesh.geometry.attributes.position.array;
  const joints = actor.positions, rotations = actor.rotations;
  const scale = Number(actor.scale) || 1;
  const rot = rotations.map(matrix3);
  for (let i=0; i<rig.vertexCount; i++) {
    let x=0,y=0,z=0;
    for (let k=0; k<rig.influences; k++) {
      const a=i*rig.influences+k, w=rig.weights[a]; if (w<.0001) continue;
      const j=rig.joints[a], b=a*3, m=rot[j], p=joints[j];
      if (!m || !p) continue;
      const lx=rig.local[b]*scale, ly=rig.local[b+1]*scale, lz=rig.local[b+2]*scale;
      x+=w*(m[0]*lx+m[1]*ly+m[2]*lz+p[0]);
      y+=w*(m[3]*lx+m[4]*ly+m[5]*lz+p[1]);
      z+=w*(m[6]*lx+m[7]*ly+m[8]*lz+p[2]);
    }
    positions[3*i]=x; positions[3*i+1]=y; positions[3*i+2]=z;
  }
  model.mesh.geometry.attributes.position.needsUpdate=true;
  model.mesh.geometry.computeVertexNormals();
  model.face.position.fromArray(joints[6]);
  const m=rot[6], rm=new THREE.Matrix4().set(m[0],m[1],m[2],0,m[3],m[4],m[5],0,m[6],m[7],m[8],0,0,0,0,1);
  model.face.quaternion.setFromRotationMatrix(rm);
  model.face.scale.setScalar(scale);
  const kiss=Number(state.snapshot?.kiss_amount||0);
  if(model.mouth){
    model.mouth.visible=which==='mj'||kiss>.04;
    model.mouth.scale.set(.024+kiss*.004,.005+kiss*.003,.004+kiss*.011);
    model.mouth.position.z=.151+kiss*.004;
  }
  if(which==='mj')for(const eye of model.eyes){
    const blink=Math.pow(Math.max(0,Math.sin((state.snapshot?.time||0)*2.8+1.8)),30);
    eye.scale.y=.008*(1-.88*blink);
  }
  if(which==='spider' && model.reveal.length){
    const colors=model.mesh.geometry.attributes.color.array;
    const skin=[.88,.52,.39];
    for(const [index,fade] of model.reveal){
      const q=index*3,a=fade*kiss;
      for(let axis=0;axis<3;axis++)colors[q+axis]=model.baseColors[q+axis]*(1-a)+skin[axis]*a;
    }
    model.mesh.geometry.attributes.color.needsUpdate=true;
  }
  if(model.chest){
    model.chest.position.fromArray(joints[4]);
    const c=rot[4],cm=new THREE.Matrix4().set(c[0],c[1],c[2],0,c[3],c[4],c[5],0,c[6],c[7],c[8],0,0,0,0,1);
    model.chest.quaternion.setFromRotationMatrix(cm);model.chest.scale.setScalar(scale);
  }
  model.lastFrame=frame;
}

const webMaterial = new THREE.LineBasicMaterial({ color:'#e6f4ff', transparent:true, opacity:.94, linewidth:2 });
const webGlowMaterial = new THREE.LineBasicMaterial({ color:'#89caff', transparent:true, opacity:.27, linewidth:5 });
const web = new THREE.Line(new THREE.BufferGeometry(),webMaterial);
const webGlow = new THREE.Line(new THREE.BufferGeometry(),webGlowMaterial);
scene.add(webGlow,web);
function setWeb(anchor, hand) {
  const visible=Array.isArray(anchor)&&anchor.length===3&&Array.isArray(hand)&&hand.length===3;
  web.visible=visible; webGlow.visible=visible; if(!visible)return;
  const start=new THREE.Vector3().fromArray(hand),end=new THREE.Vector3().fromArray(anchor);
  const points=[start,end];
  web.geometry.dispose();webGlow.geometry.dispose();
  web.geometry=new THREE.BufferGeometry().setFromPoints(points);
  webGlow.geometry=new THREE.BufferGeometry().setFromPoints(points);
  currentAnchor=end;
}

function actorList(snapshot) {
  if(Array.isArray(snapshot.actors)) return snapshot.actors;
  if(Array.isArray(snapshot.positions)) return snapshot.positions.map((p,i)=>({id:i?'mj':'spider',positions:p,rotations:snapshot.rotations?.[i],scale:i?.94:1}));
  return [];
}
function applyState(snapshot) {
  if(!snapshot || !rig) return;
  if(snapshot.error) throw new Error(`Motion engine: ${snapshot.error}`);
  if(snapshot.run_id && snapshot.run_id!==state.runId){
    state.runId=snapshot.run_id;state.ackedCommand=null;state.lastPaintedFrame=-1;state.carryAt=null;
    for(const model of actors.values())model.lastFrame=-1;
  }
  if(snapshot.carrying && state.carryAt===null)state.carryAt=snapshot.time||0;
  if(!snapshot.carrying)state.carryAt=null;
  state.snapshot=snapshot; state.connected=true; state.lastStateAt=performance.now();
  state.error='';
  pulse.classList.add('online');connection.textContent='Motion engine online';
  const list=actorList(snapshot);
  for(let i=0;i<list.length;i++) poseActor(i?'mj':'spider',list[i],snapshot.frame);
  const hero=list[0];
  setWeb(snapshot.web_anchor,snapshot.web_hand||hero?.positions?.[16]||hero?.joints?.[16]);
  state.phase=String(snapshot.phase||'LIVE').toUpperCase();
  if(Array.isArray(snapshot.events)) state.events=snapshot.events.slice(-5);
  if(loading && state.sceneReady && state.rigReady) loading.classList.add('done');
}

async function getJson(url) {
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
async function initialize() {
  try {
    loadingDetail.textContent='Loading generated city geometry…';
    setGeometry(await getJson('/api/scene'));
    loadingDetail.textContent='Loading exact CoreSkin mesh and rig…';
    prepareRig(await getJson('/api/rig'));
    loadingDetail.textContent='Connecting to live Core27 poses…';
    await pollState();
  } catch(e) {
    loadingDetail.textContent=e.message;
    connection.textContent='Scene unavailable'; console.error(e);
  }
}
let polling=false;
async function pollState() {
  if(polling) return; polling=true;
  try { applyState(await getJson('/api/state')); }
  catch(e) { state.connected=false;state.error=e.message;pulse.classList.remove('online');connection.textContent='Reconnecting to motion engine';if(e.message.startsWith('Motion engine:')){loading.classList.remove('done');loadingDetail.textContent=e.message;} }
  finally {polling=false;state.lastPollAt=performance.now();}
}

async function command(action) {
  if(state.commandBusy || !state.rigReady) return;
  state.commandBusy=true;state.command=action;state.commandAt=performance.now();
  try {
    const res=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
    if(!res.ok) throw new Error(await res.text());
    await res.json();
  } catch(e) {state.events=[`Command failed: ${e.message}`];}
  finally {state.commandBusy=false;}
}

let buttonRects=[];
const buttonDefs=[
  {id:'swing',label:'↗  START SWING',action:()=>command('swing'),key:'S'},
  {id:'left',label:'←  STEER LEFT',action:()=>command('left'),key:'←'},
  {id:'right',label:'STEER RIGHT  →',action:()=>command('right'),key:'→'},
  {id:'carry',label:'CARRY MJ',action:()=>command('carry'),key:'C'},
  {id:'land',label:'LAND',action:()=>command('land'),key:'L'},
  {id:'kiss',label:'KISS',action:()=>command('kiss'),key:'K'},
  {id:'camera',label:'◉  CAMERA',action:()=>cycleCamera(),key:'SPACE'},
  {id:'reset',label:'↺  RESET',action:()=>command('reset'),key:'X'},
  {id:'record',label:'●  RECORD',action:()=>toggleRecording(),key:'R'},
];
for(const def of buttonDefs) {
  const b=document.createElement('button');b.type='button';b.textContent=def.label;b.setAttribute('aria-label',def.id==='record'?'Record actual canvas video':def.label);
  b.dataset.action=def.id;b.addEventListener('click',def.action);access.appendChild(b);def.node=b;
}
function layoutButtons() {
  const mobile=stageW<850, margin=mobile?14:30, gap=mobile?6:8, h=mobile?29:39;
  buttonRects=[];
  if(mobile){
    const first=buttonDefs.slice(0,4),second=buttonDefs.slice(4);
    for(const [row,defs] of [[0,first],[1,second]]){
      const base=(stageW-2*margin-gap*(defs.length-1))/defs.length;
      defs.forEach((def,i)=>buttonRects.push({def,x:margin+i*(base+gap),y:stageH-88+row*35,w:base,h}));
    }
  } else {
    const base=Math.min(132,(stageW-2*margin-gap*(buttonDefs.length-1))/buttonDefs.length);
    buttonDefs.forEach((def,i)=>buttonRects.push({def,x:margin+i*(base+gap),y:stageH-83,w:base,h}));
  }
  for(const r of buttonRects){r.def.node.style.left=`${r.x}px`;r.def.node.style.top=`${r.y}px`;r.def.node.style.width=`${r.w}px`;r.def.node.style.height=`${r.h}px`;}
}
function cycleCamera(){
  const modes=['CINEMATIC','CHASE','CLOSE'];
  state.cameraMode=modes[(modes.indexOf(state.cameraMode)+1)%modes.length];
}
resize();
window.addEventListener('keydown',e=>{
  const key=e.key.toLowerCase();
  if(['arrowleft','arrowright',' '].includes(key)) e.preventDefault();
  if(key==='arrowleft'||key==='a')command('left');
  if(key==='arrowright'||key==='d')command('right');
  if(key==='s')command('swing');if(key==='x')command('reset');
  if(key==='c')command('carry');if(key==='l')command('land');if(key==='k')command('kiss');
  if(key==='r')toggleRecording();
  if(key===' ')cycleCamera();
});
canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect();state.pointerX=e.clientX-r.left;state.pointerY=e.clientY-r.top;if(state.dragStart){state.orbitOffset+=(e.movementX||0)*.005;}});
canvas.addEventListener('pointerdown',e=>{const r=canvas.getBoundingClientRect();state.dragStart={x:e.clientX-r.left,y:e.clientY-r.top};});
canvas.addEventListener('pointerup',e=>{const r=canvas.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;if(state.dragStart && Math.abs(x-state.dragStart.x)+Math.abs(y-state.dragStart.y)<8){const hit=buttonRects.find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);hit?.def.action();}state.dragStart=null;});

async function toggleRecording(){
  if(state.recorder && state.recorder.state==='recording'){state.recorder.stop();return;}
  if(state.snapshot?.recording_sealed){state.recordingStatus='Reset before capturing another measured take';return;}
  if(!canvas.captureStream||!window.MediaRecorder){state.recordingStatus='Recording unavailable in this browser';return;}
  const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(x=>MediaRecorder.isTypeSupported(x));
  if(!mime){state.recordingStatus='WebM recording unsupported';return;}
  state.recordedChunks=[];const rec=new MediaRecorder(canvas.captureStream(30),{mimeType:mime,videoBitsPerSecond:8_000_000});
  rec.ondataavailable=e=>{if(e.data.size)state.recordedChunks.push(e.data);};
  rec.onstop=async()=>{
    state.recordingStatus='Saving actual live capture…';
    const blob=new Blob(state.recordedChunks,{type:mime});
    if(blob.size===0){state.recordingStatus='No video frames were captured';return;}
    const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='interactive-run.webm';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),60000);
    try{const res=await fetch('/api/recording?name=interactive-run.webm',{method:'POST',headers:{'Content-Type':mime},body:blob});if(!res.ok)throw new Error(`HTTP ${res.status}`);state.recordingStatus='Live video saved';}
    catch(e){state.recordingStatus=`Downloaded locally · server save failed (${e.message})`;}
  };
  rec.start(1000);state.recorder=rec;state.recordStart=performance.now();state.recordingStatus='Recording live canvas and controls';
}

function cameraUpdate(dt) {
  const s=state.snapshot; const root=s?.root || s?.actors?.[0]?.positions?.[0] || [0,2,-8];
  const phase=String(s?.phase||''); const carryClose=s?.carrying && state.carryAt!==null && (s.time-state.carryAt)<4.5;
  const isClose=/carry|pickup|land|kiss|final|embrace/i.test(phase)||carryClose;
  const mode=state.cameraMode;
  let x=root[0],y=root[1],z=root[2];
  if(!Number.isFinite(y))y=2;
  const endingPortrait=/kiss|landed|settling/i.test(phase);
  const angle=(Number(s?.yaw)||0)+state.orbitOffset;
  const side=(distance,front)=>[Math.cos(angle)*distance+Math.sin(angle)*front,
    -Math.sin(angle)*distance+Math.cos(angle)*front];
  let targetY=isClose?y+.65:y+.5, targetZ=z-1.2, targetX=x;
  if(mode==='CLOSE'&&endingPortrait){
    const hero=s?.actors?.[0]?.positions?.[6],mj=s?.actors?.[1]?.positions?.[6];
    const mid=hero&&mj?[(hero[0]+mj[0])/2,(hero[1]+mj[1])/2,(hero[2]+mj[2])/2]:[x,y+.75,z];
    const [dx,dz]=side(2.1,0);
    desiredCamera.set(mid[0]+dx,mid[1]+.45,mid[2]+dz);
    targetX=mid[0];targetY=mid[1];targetZ=mid[2];
  }
  else if(mode==='CLOSE'){
    const [dx,dz]=side(2.4,3.25);
    desiredCamera.set(x+dx,Math.max(2.1,y+1.0),z+dz);
    targetY=y+.2;targetZ=z;
  }
  else if(mode==='CHASE') desiredCamera.set(x+Math.sin(state.orbitOffset)*6,Math.max(2.5,y+2.3),z+7.8*Math.cos(state.orbitOffset));
  else if(isClose)desiredCamera.set(x+3.25+Math.sin(state.orbitOffset)*.8,Math.max(2.1,y+1.4),z+4.35);
  else desiredCamera.set(x+8.5+Math.sin(state.orbitOffset)*1.2,Math.max(5,y+4.8),z+12.0);
  if(mode!=='CLOSE'&&s?.carrying){targetY=y+.2;targetZ=z;}
  scratchV.set(targetX,targetY,targetZ);
  const alpha=1-Math.exp(-Math.min(dt,.08)*(isClose||mode==='CLOSE'?3.5:1.8));
  camera.position.lerp(desiredCamera,alpha);lookAt.lerp(scratchV,alpha);camera.lookAt(lookAt);
  camera.fov=THREE.MathUtils.lerp(camera.fov,mode==='CLOSE'?(endingPortrait?32:38):isClose?41:48,alpha);camera.updateProjectionMatrix();
}

function roundRect(x,y,w,h,r,fill,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,r);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function label(text,x,y,size=11,color='#d5e1ef',weight=500,align='left'){ctx.font=`${weight} ${size}px Inter, Arial, sans-serif`;ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(text,x,y);}
function mono(text,x,y,size=10,color='#91a2b8',align='left'){ctx.font=`500 ${size}px 'DM Mono', monospace`;ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(text,x,y);}
function trim(value,max=43){const str=String(value||'');return str.length>max?str.slice(0,max-1)+'…':str;}
function hud(now){
  const w=stageW,h=stageH,mobile=w<850,edge=mobile?15:29;
  const gradient=ctx.createLinearGradient(0,0,0,h);gradient.addColorStop(0,'#08121b96');gradient.addColorStop(.24,'#08121b00');gradient.addColorStop(.64,'#08121b00');gradient.addColorStop(1,'#071018e8');ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
  roundRect(edge,20,mobile?164:218,51,8,'#0a1526d9','#56718777');
  mono('SCENE / 001',edge+15,39,10,'#eab4ad');label('THE SKYLINE',edge+15,61,mobile?15:19,'#fff',800);
  const status=state.connected?(state.phase||'LIVE'):'CONNECTING';
  roundRect(w-edge-(mobile?150:207),20,mobile?150:207,51,8,'#0a1526d9','#56718777');
  ctx.fillStyle=state.connected?'#3dd5b2':'#e7a557';ctx.beginPath();ctx.arc(w-edge-(mobile?131:183),38,4,0,Math.PI*2);ctx.fill();
  mono(status,w-edge-(mobile?120:171),42,mobile?8:10,'#e7f0fb');
  mono(`FRAME ${String(state.snapshot?.frame??0).padStart(4,'0')}  ·  ${state.cameraMode}`,w-edge-12,61,mobile?7:9,'#9ab2ca','right');
  if(!mobile){
    const metrics=state.snapshot?.metrics||{};
    const latency=Number(metrics.latency_ms??metrics.end_to_end_ms??0);
    const stalls=Number(metrics.stalls??0);
    const gripFields=metrics.pose||{};
    const grip=gripFields.active?Math.max(gripFields.mj_right_shoulder_m||0,gripFields.mj_left_shoulder_m||0,gripFields.hero_thigh_m||0):null;
    roundRect(edge,86,218,113,8,'#0a1526c5','#56718755');
    mono('LIVE PERFORMANCE',edge+14,106,9,'#b9ccdd');
    mono(`LATENCY     ${latency?Math.round(latency)+' ms':'—'}`,edge+14,127,10,'#e8f5ff');
    mono(`STALLS      ${stalls} engine / ${state.localStalls} view`,edge+14,147,9,stalls||state.localStalls?'#ffbd8a':'#8de1be');
    mono(`GRIP        ${grip===null?'—':grip.toFixed(3)+' m'}`,edge+14,167,10,'#e8f5ff');
    const collisions=metrics.collision_corrections??metrics.collisions??0;
    mono(`AVOIDED HITS ${collisions}`,edge+14,187,10,collisions?'#ffbd8a':'#8de1be');
    if(state.events.length){roundRect(w-edge-244,87,244,52,8,'#0a1526bf','#56718755');mono('DIRECTOR EVENT',w-edge-230,107,9,'#eab4ad');label(trim(state.events.at(-1)?.text||state.events.at(-1)?.message||state.events.at(-1),35),w-edge-230,126,11,'#d8e6f5');}
  }
  const rec=state.recorder?.state==='recording';
  if(rec){roundRect(w/2-74,25,148,29,15,'#821c27dc');ctx.fillStyle='#ff626b';ctx.beginPath();ctx.arc(w/2-54,39,4,0,Math.PI*2);ctx.fill();mono(`REC  ${new Date(now-state.recordStart).toISOString().slice(14,19)}`,w/2-41,43,10,'#fff');}
  if(state.recordingStatus){mono(trim(state.recordingStatus,mobile?32:60),w/2,mobile?h-120:h-101,mobile?8:10,'#e6edf9','center');}
  const action=state.command && now-state.commandAt<1500;
  if(action){roundRect(w/2-108,h*.22,216,36,18,'#172e42de','#8ec8e877');mono(state.command.toUpperCase()+' COMMAND SENT',w/2,h*.22+23,10,'#e7f7ff','center');}
  const barY=mobile?h-105:h-105;
  mono('LIVE DIRECTION',edge,barY,mobile?8:10,'#ced8e6');
  if(!mobile)mono('CLICK A CONTROL OR USE THE KEYBOARD',w-edge,barY,10,'#a6b6c6','right');
  for(const r of buttonRects){
    const {def,x,y,w:bw,h:bh}=r;const hovered=state.pointerX>=x&&state.pointerX<=x+bw&&state.pointerY>=y&&state.pointerY<=y+bh;
    const isRec=def.id==='record'&&rec;
    const active=hovered||isRec||(state.command===def.id&&now-state.commandAt<420);
    roundRect(x,y,bw,bh,6,isRec?'#ad2635':active?'#354f69':'#1b2b3ee8',isRec?'#f57983':active?'#a5d5f0':'#607c937d');
    const title=def.id==='record'?(rec?'■ STOP & SAVE':'● RECORD'):def.label;
    mono(title,x+bw/2,y+bh/2+(mobile?3:4),mobile?8:9,active?'#ffffff':'#d5e5f4','center');
  }
  mono(state.connected?'● LIVE ENGINE':'○ WAITING FOR ENGINE',edge,h-13,mobile?8:9,state.connected?'#70deb9':'#e5a16b');
  mono('REAL TIME · REAL POSES · REAL CAPTURE',w-edge,h-13,mobile?7:9,'#849ab0','right');
}

let lastFrameAt=performance.now();
function animate(now){
  requestAnimationFrame(animate);
  const frameMs=now-lastFrameAt,dt=Math.min(.1,frameMs/1000);lastFrameAt=now;
  if(state.rigReady&&now-state.lastPollAt>50)pollState();
  if(state.connected&&now-state.lastStateAt>1200){state.connected=false;state.localStalls++;pulse.classList.remove('online');connection.textContent='Motion stream stalled';}
  cameraUpdate(dt);
  renderer.render(scene,camera);
  ctx.setTransform(pixelScale,0,0,pixelScale,0,0);
  ctx.drawImage(renderer.domElement,0,0,stageW,stageH);
  hud(now);
  const snap=state.snapshot;
  if(snap&&snap.frame!==state.lastPaintedFrame){
    state.lastPaintedFrame=snap.frame;
    const id=snap.last_command_id;
    if(Number.isInteger(id)&&id>0&&id!==state.ackedCommand){
      state.ackedCommand=id;
      fetch('/api/telemetry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command_id:id,frame:snap.frame,client_frame_ms:frameMs,stall_count:state.localStalls})}).catch(()=>{});
    }
  }
}
requestAnimationFrame(animate);
initialize();
