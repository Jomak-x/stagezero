import * as THREE from '/three/build/three.module.js';

const canvas = document.querySelector('#stage');
const ctx = canvas.getContext('2d', { alpha: false });
const wrap = document.querySelector('.stage-wrap');
const controls = document.querySelector('#accessible-controls');
const clipSelect = document.querySelector('#clip-select');
const actionPrompt = document.querySelector('#action-prompt');
const generateButton = document.querySelector('#generate');
const frameInput = document.querySelector('#frame-input');
const loading = document.querySelector('#loading');
const loadingDetail = document.querySelector('#loading-detail');

const threeCanvas = document.createElement('canvas');
threeCanvas.style.cssText = 'position:absolute;inset:0;opacity:0;pointer-events:none';
threeCanvas.setAttribute('aria-hidden', 'true');
wrap.prepend(threeCanvas);
const renderer = new THREE.WebGLRenderer({ canvas: threeCanvas, antialias: true, powerPreference:'high-performance', preserveDrawingBuffer:true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.38;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#a4aaa1');
scene.fog = new THREE.Fog('#b6bdb3', 18, 66);
const camera = new THREE.PerspectiveCamera(48, 16/9, .05, 150);
camera.position.set(3.8,3.3,8);
scene.add(new THREE.HemisphereLight(0xe5eeec, 0x63746c, 2.15));
const sun = new THREE.DirectionalLight(0xffe2bd, 3.0);
sun.position.set(-8,15,7);sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-25;sun.shadow.camera.right=25;
sun.shadow.camera.top=25;sun.shadow.camera.bottom=-25;sun.shadow.camera.near=1;sun.shadow.camera.far=60;
sun.shadow.bias=-.00025;sun.shadow.normalBias=.02;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x9cc6d4, .7);fill.position.set(7,8,-10);scene.add(fill);

const ui = {
  w:1200,h:700,dpr:1, clips:[], clip:null, clipId:'', frame:0,playhead:0,
  playing:false,loadingClip:false,sceneReady:false,rigReady:false,ready:false,
  camera:'FRONT',cameraInitialized:false,orbit:0,pointerX:0,pointerY:0,dragging:false,dragStart:null,
  record:null,recordAt:0,chunks:[],recordStatus:'',job:null,jobStatus:'',
  recordStartedWall:null,recordEndedWall:null,recordEndedPerf:null,rafSamples:[],rafGapCount:0,excludedLoadingFrames:0,
  generationLog:[],activeContinuation:null,pendingVisibleContinuations:[],bufferHoldStart:null,
  wantedResume:false,live:false,lastSubmittedPrompt:'',queuedPrompt:null,liveEvents:[],
  sourceError:'',sceneProvenance:'',lastTick:performance.now(),
};
const sceneMeshes=[];
const actors=[];
const cameraGoal=new THREE.Vector3(3.8,3.3,8);
const targetGoal=new THREE.Vector3(0,1.2,-4);
const cameraTarget=new THREE.Vector3(0,1.2,-4);

function resize(){
  ui.w=Math.max(320,wrap.clientWidth);ui.h=Math.max(320,wrap.clientHeight);
  ui.dpr=Math.min(devicePixelRatio||1,1.5);
  canvas.width=Math.round(ui.w*ui.dpr);canvas.height=Math.round(ui.h*ui.dpr);
  canvas.style.width=`${ui.w}px`;canvas.style.height=`${ui.h}px`;
  ctx.setTransform(ui.dpr,0,0,ui.dpr,0,0);
  renderer.setSize(ui.w,ui.h,false);
  camera.aspect=ui.w/ui.h;camera.updateProjectionMatrix();
  layoutButtons();
}
new ResizeObserver(resize).observe(wrap);

async function json(url,options){
  const response=await fetch(url,{cache:'no-store',...options});
  if(!response.ok){let detail='';try{detail=(await response.json()).error||'';}catch{}throw new Error(`${response.status} ${detail||url}`);}
  return response.json();
}
function flatten(rows){return Array.isArray(rows?.[0])?rows.flat():rows;}
function color(bytes){return new THREE.Color().setRGB((bytes?.[0]??160)/255,(bytes?.[1]??160)/255,(bytes?.[2]??160)/255,THREE.SRGBColorSpace);}

function buildScene(doc){
  const batches=doc.meshes||[];
  if(!batches.length)throw new Error('Generated scene contains no mesh geometry');
  for(const item of batches){
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(flatten(item.positions),3));
    geometry.setIndex(flatten(item.indices||item.faces));
    geometry.computeVertexNormals();
    const material=new THREE.MeshStandardMaterial({color:color(item.color),roughness:.92,metalness:0,side:THREE.DoubleSide});
    const mesh=new THREE.Mesh(geometry,material);
    mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh);sceneMeshes.push(mesh);
  }
  ui.sceneReady=true;
  ui.sceneProvenance=doc.provenance?.source||doc.provenance?.source_path||'Generated Market Square geometry';
  if(doc.camera?.position)camera.position.fromArray(doc.camera.position);
  if(doc.camera?.look_at){cameraTarget.fromArray(doc.camera.look_at);camera.lookAt(cameraTarget);}
}

function loadTexture(url,srgb=false){
  if(!url)return Promise.resolve(null);
  return new Promise((resolve,reject)=>{
    new THREE.TextureLoader().load(url,texture=>{
      // Trimesh's extracted UVs use the bottom-origin convention. Its CPU
      // reference samples image row 1-v, matching TextureLoader's Y flip.
      texture.flipY=true;
      texture.colorSpace=srgb?THREE.SRGBColorSpace:THREE.NoColorSpace;
      texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
      texture.needsUpdate=true;resolve(texture);
    },undefined,reject);
  });
}

async function buildActor(index,rig){
  const required=['vertices','faces','normals','uv','skin_indices','skin_weights','rest_positions'];
  for(const key of required)if(!rig[key]?.length)throw new Error(`Generated character ${index+1} lacks ${key}`);
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(flatten(rig.vertices),3));
  geometry.setAttribute('normal',new THREE.Float32BufferAttribute(flatten(rig.normals),3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(flatten(rig.uv),2));
  geometry.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(flatten(rig.skin_indices),4));
  geometry.setAttribute('skinWeight',new THREE.Float32BufferAttribute(flatten(rig.skin_weights),4));
  geometry.setIndex(flatten(rig.faces));
  geometry.computeBoundingSphere();
  const [map,normalMap,metallicRoughness]=await Promise.all([
    loadTexture(rig.texture_url,true),loadTexture(rig.normal_texture_url),loadTexture(rig.metallic_roughness_texture_url)
  ]);
  const properties=rig.material||{};
  const base=properties.base_color_factor||[1,1,1,1];
  const material=new THREE.MeshStandardMaterial({
    color:new THREE.Color().setRGB(base[0],base[1],base[2],THREE.LinearSRGBColorSpace),
    map,normalMap,metalnessMap:metallicRoughness,roughnessMap:metallicRoughness,
    metalness:properties.metallic_factor??0,roughness:properties.roughness_factor??.9,
    side:THREE.DoubleSide,transparent:base[3]<.99,opacity:base[3],
  });
  const bones=rig.rest_positions.map((p,i)=>{
    const bone=new THREE.Bone();bone.name=rig.bone_names?.[i]||`bone-${i}`;
    bone.matrixAutoUpdate=false;bone.matrixWorldAutoUpdate=false;
    bone.matrixWorld.makeTranslation(p[0],p[1],p[2]);
    return bone;
  });
  const inverses=rig.rest_positions.map(p=>new THREE.Matrix4().makeTranslation(-p[0],-p[1],-p[2]));
  const skeleton=new THREE.Skeleton(bones,inverses);
  const mesh=new THREE.SkinnedMesh(geometry,material);
  mesh.bindMode='detached';mesh.bind(skeleton,new THREE.Matrix4());
  mesh.frustumCulled=false;mesh.castShadow=true;mesh.receiveShadow=true;
  scene.add(mesh);
  actors[index]={mesh,bones,skeleton,rig,lastFrame:-1};
  // Material and UVs come directly from this model-generated GLB.
  if(!map)throw new Error(`Generated character ${index+1} has no base-color texture`);
}

function matrix3(rotation){return Array.isArray(rotation?.[0])?rotation.flat():rotation;}
function applyNativeFrame(index,frame){
  const actor=actors[index];
  const p=ui.clip?.fitted_positions?.[index]?.[frame];
  const r=ui.clip?.fitted_rotations?.[index]?.[frame];
  if(!actor||!p||!r||actor.lastFrame===frame)return;
  for(let j=0;j<actor.bones.length;j++){
    const m=matrix3(r[j]),v=p[j];
    actor.bones[j].matrixWorld.set(m[0],m[1],m[2],v[0],m[3],m[4],m[5],v[1],m[6],m[7],m[8],v[2],0,0,0,1);
  }
  actor.skeleton.update();
  actor.lastFrame=frame;
}
function showFrame(frame){
  if(!ui.clip)return;
  const committedEnd=ui.job&&ui.activeContinuation?ui.activeContinuation.prefix_frames-1:ui.clip.frames-1;
  ui.frame=Math.max(0,Math.min(committedEnd,Math.floor(frame)));
  for(let i=0;i<Math.min(actors.length,ui.clip.actors);i++)applyNativeFrame(i,ui.frame);
  for(let i=0;i<actors.length;i++)actors[i].mesh.visible=i<ui.clip.actors;
  if(document.activeElement!==frameInput)frameInput.value=String(ui.frame);
  frameInput.max=String(committedEnd);
}
function markRenderedContinuations(){
  if(!ui.pendingVisibleContinuations.length)return;
  const renderedAt=performance.now();
  ui.pendingVisibleContinuations=ui.pendingVisibleContinuations.filter(entry=>{
    if(ui.frame<entry.prefix_frames)return true;
    entry.first_visible_appended_frame_at_utc=new Date().toISOString();
    entry.first_visible_frame=ui.frame;
    entry.first_visible_perf=renderedAt;
    entry.first_visible_latency_ms=Math.round(renderedAt-entry.submitted_perf);
    entry.command_to_visible_ms=Math.round(renderedAt-entry.command_received_perf);
    entry.first_visible_outcome='shown';
    return false;
  });
}
function reconcilePendingVisible(committedPrefix){
  ui.pendingVisibleContinuations=ui.pendingVisibleContinuations.filter(entry=>{
    if(entry.prefix_frames<committedPrefix)return true;
    entry.first_visible_outcome='replaced_before_visible';
    entry.replaced_before_visible_at_utc=new Date().toISOString();
    return false;
  });
}

function cleanLabel(text){return String(text||'Untitled motion').replace(/[_-]+/g,' ').replace(/\s+/g,' ').trim();}
function liveUnavailableReason(clip){
  if(!clip)return null;
  if(clip.actors>1)return 'Paired · replay only (contact unsupported)';
  if(/kimodo/i.test(`${clip.label||''} ${clip.source||''} ${clip.metadata?.source||''}`))
    return 'Kimodo · replay only (Core-incompatible)';
  return null;
}
function isDiagnostic(clip){return /fail|diagnostic|ablation|stress|probe|interrupted/i.test(`${clip.label} ${clip.source}`);}
function preferredClip(clips){return clips.find(c=>c.featured)||clips.find(c=>/staged.?fight/i.test(c.label))||clips.find(c=>!isDiagnostic(c)&&c.actors===2)||clips.find(c=>!isDiagnostic(c))||clips[0];}
async function refreshClips(){
  const clips=await json('/api/grounded/clips');
  ui.clips=Array.isArray(clips)?clips:[];
  clipSelect.replaceChildren();
  for(const clip of ui.clips){const option=document.createElement('option');option.value=clip.id;option.textContent=`${cleanLabel(clip.label)} · ${clip.frames} frames`;clipSelect.appendChild(option);}
  if(!ui.clips.length)throw new Error('No saved native Core27 clips are available');
}
function mergeLiveTail(tail,source,sourceId,start,id){
  if(tail.id!==id||tail.source_clip_id&&tail.source_clip_id!==sourceId)
    throw new Error('Generated tail belongs to a different clip or source');
  if(!Number.isInteger(start)||start<1||start>source.frames||tail.start_frame!==start)
    throw new Error('Generated tail starts outside the committed native prefix');
  if(!Number.isInteger(tail.frames)||tail.frames<=start||tail.actors!==source.actors||tail.fps!==source.fps)
    throw new Error('Generated tail has incompatible frame or actor metadata');
  if(tail.metadata?.preserved_prefix_frames!=null&&tail.metadata.preserved_prefix_frames!==start)
    throw new Error('Generated tail prefix metadata differs from the committed frame');
  const merged={...tail};
  for(const key of ['positions','rotations','fitted_positions','fitted_rotations']){
    const oldActors=source[key],newActors=tail[key];
    if(!Array.isArray(oldActors)||!Array.isArray(newActors)||oldActors.length!==source.actors||newActors.length!==source.actors)
      throw new Error(`Generated tail lacks compatible ${key}`);
    merged[key]=oldActors.map((oldFrames,actor)=>{
      const newFrames=newActors[actor];
      if(!Array.isArray(oldFrames)||oldFrames.length<start||!Array.isArray(newFrames)||newFrames.length!==tail.frames-start)
        throw new Error(`Generated tail ${key} has an invalid frame count`);
      return oldFrames.slice(0,start).concat(newFrames);
    });
  }
  return merged;
}
async function selectClip(id,{preserveFrame=false,resume=false,liveSwap=false,liveStart=null}={}){
  if(ui.loadingClip)return;
  const sourceClip=liveSwap?ui.clip:null;
  const sourceId=liveSwap?ui.clipId:null;
  if(liveSwap&&(!sourceClip||ui.activeContinuation?.source_clip_id!==sourceId||ui.activeContinuation?.prefix_frames!==liveStart))
    throw new Error('Live source changed before the generated tail request');
  if(!liveSwap){
    ui.loadingClip=true;ui.sourceError='';loading.classList.remove('done');
    loadingDetail.textContent='Loading exact native model frames…';
  }
  try{
    const url=`/api/grounded/clip?id=${encodeURIComponent(id)}`+(liveSwap?`&start=${liveStart}`:'');
    const response=await json(url);
    if(liveSwap&&ui.clipId!==sourceId)throw new Error('Live source changed while fetching generated frames');
    const data=liveSwap?mergeLiveTail(response,sourceClip,sourceId,liveStart,id):response;
    if(!data.fitted_positions?.length||!data.fitted_rotations?.length)throw new Error('Clip has no fitted transforms');
    // A live replacement retains the current frame and fractional playhead
    // after the fetch. The old committed prefix remains visible meanwhile.
    const currentFrame=ui.frame,currentPlayhead=ui.playhead,currentPlaying=ui.playing;
    if(liveSwap){reconcilePendingVisible(ui.activeContinuation?.prefix_frames??0);ui.job=null;}
    ui.clip=data;ui.clipId=id;
    ui.frame=preserveFrame?Math.min(currentFrame,data.frames-1):0;
    ui.playhead=liveSwap?currentPlayhead:ui.frame/data.fps;
    ui.playing=liveSwap?currentPlaying:resume;
    for(const actor of actors)actor.lastFrame=-1;
    showFrame(ui.frame);
    clipSelect.value=id;
    const liveRestriction=liveUnavailableReason(data);
    generateButton.disabled=Boolean(liveRestriction);
    if(liveRestriction){ui.jobStatus=liveRestriction;ui.live=false;}
    if(!liveSwap){
      for(const entry of ui.pendingVisibleContinuations)entry.first_visible_outcome='abandoned_by_clip_change';
      ui.pendingVisibleContinuations=[];
      loading.classList.add('done');
    }
  }catch(e){
    ui.sourceError=e.message;loadingDetail.textContent=e.message;
    if(liveSwap)throw e;
    ui.playing=false;
  }
  finally{if(!liveSwap)ui.loadingClip=false;}
}
async function initialize(){
  try{
    loadingDetail.textContent='Loading generated Market Square…';
    buildScene(await json('/api/grounded/scene'));
    loadingDetail.textContent='Loading two textured generated humans…';
    const rigs=await Promise.all([json('/api/grounded/rig?actor=0'),json('/api/grounded/rig?actor=1')]);
    await buildActor(0,rigs[0]);await buildActor(1,rigs[1]);
    ui.rigReady=true;
    loadingDetail.textContent='Indexing saved native Core27 clips…';
    await refreshClips();
    ui.ready=true;
    const queryClip=new URLSearchParams(location.search).get('clip');
    await selectClip(ui.clips.some(c=>c.id===queryClip)?queryClip:preferredClip(ui.clips).id);
  }catch(e){ui.sourceError=e.message;loadingDetail.textContent=e.message;console.error(e);}
}

function step(delta){
  if(!ui.clip)return;
  ui.playing=false;ui.wantedResume=false;showFrame(ui.frame+delta);ui.playhead=ui.frame/ui.clip.fps;
}
function togglePlay(){
  if(!ui.clip||ui.loadingClip)return;
  if(ui.frame>=ui.clip.frames-1&&!ui.job){showFrame(0);ui.playhead=0;}
  ui.playing=!ui.playing;
  if(!ui.playing)ui.wantedResume=false;
}
function reset(){ui.playing=false;ui.wantedResume=false;ui.playhead=0;showFrame(0);}
function cycleCamera(){ui.camera=ui.camera==='STAGE'?'FRONT':ui.camera==='FRONT'?'CLOSE':'STAGE';}
async function adjacent(direction){
  if(!ui.clips.length||ui.job)return;
  if(ui.live)toggleLive();
  const index=ui.clips.findIndex(c=>c.id===ui.clipId);
  const next=(index+direction+ui.clips.length)%ui.clips.length;
  await selectClip(ui.clips[next].id);
}

function toggleLive(){
  const restriction=liveUnavailableReason(ui.clip);
  if(restriction){ui.jobStatus=restriction;return;}
  ui.live=!ui.live;
  ui.liveEvents.push({at_utc:new Date().toISOString(),at_perf:performance.now(),enabled:ui.live,clip_id:ui.clipId,frame:ui.frame});
  ui.jobStatus=ui.live?(ui.lastSubmittedPrompt?'LIVE ON · native actions will refill near the end':'LIVE ON · submit the first action'):
    (ui.job?'LIVE OFF · finishing current generation':'LIVE OFF · playback will end');
}
function maybeAutoContinue(){
  if(!ui.live||!ui.playing||!ui.clip||liveUnavailableReason(ui.clip)||ui.loadingClip||ui.job||!ui.lastSubmittedPrompt)return;
  if(ui.clip.frames-1-ui.frame<=60)void requestContinuation(ui.lastSubmittedPrompt,'live');
}
generateButton.addEventListener('click',()=>requestContinuation());
actionPrompt.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();requestContinuation();}});
async function requestContinuation(promptOverride=null,origin='manual',receivedCommand=null){
  const prompt=String(promptOverride??actionPrompt.value).trim();
  if(!prompt||!ui.clip)return;
  const restriction=liveUnavailableReason(ui.clip);
  if(restriction){ui.jobStatus=restriction;return;}
  const command=receivedCommand||{prompt,received_perf:performance.now(),received_at_utc:new Date().toISOString()};
  if(ui.job){
    if(origin==='manual'){
      ui.queuedPrompt=command;
      ui.jobStatus='Native generation running · newer action queued';
    }
    return;
  }
  const latestBoundary=Math.floor(ui.clip.frames/4)*4;
  const commitFrame=Math.min(latestBoundary,Math.max(40,Math.ceil((ui.frame+60)/4)*4));
  if(commitFrame<40||ui.frame>=commitFrame){ui.jobStatus='No unplayed four-frame boundary remains';return;}
  ui.lastSubmittedPrompt=prompt;
  const submittedPerf=performance.now();
  const entry={source_clip_id:ui.clipId,prompt,origin,prefix_frames:commitFrame,
    commit_frame:commitFrame,replaced_unplayed_frames:ui.clip.frames-commitFrame,
    command_received_at_utc:command.received_at_utc,command_received_perf:command.received_perf,
    submitted_at_utc:new Date().toISOString(),submitted_perf:submittedPerf,
    queue_wait_ms:Math.round(submittedPerf-command.received_perf),
    job_ready_at_utc:null,job_ready_latency_ms:null,first_visible_appended_frame_at_utc:null,
    first_visible_latency_ms:null,command_to_visible_ms:null,first_visible_outcome:'pending',buffer_hold_seconds:0};
  ui.generationLog.push(entry);ui.activeContinuation=entry;ui.job='submitting';
  clipSelect.disabled=true;frameInput.max=String(commitFrame-1);
  ui.jobStatus='Submitting native continuation…';
  try{
    const response=await json('/api/grounded/continue',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip_id:ui.clipId,prompt,commit_frame:commitFrame})});
    ui.job=response.job_id||response.id;
    if(!ui.job)throw new Error('Continuation job returned no ID');
    ui.jobStatus='Generating next native Core frames…';
    while(ui.job){
      await new Promise(resolve=>setTimeout(resolve,650));
      const status=await json(`/api/grounded/job?id=${encodeURIComponent(ui.job)}`);
      if(status.status==='complete'){
        if(status.preserved_prefix_frames!==commitFrame)throw new Error('Generated prefix differs from the committed native history');
        if(ui.frame>=commitFrame)throw new Error('Generation arrived after the committed playback boundary');
        entry.job_ready_at_utc=new Date().toISOString();
        entry.job_ready_perf=performance.now();
        entry.job_ready_latency_ms=Math.round(entry.job_ready_perf-entry.submitted_perf);
        entry.command_to_job_ready_ms=Math.round(entry.job_ready_perf-entry.command_received_perf);
        entry.preserved_prefix_frames=status.preserved_prefix_frames;
        const nextId=status.clip_id||status.result?.clip_id;
        if(!nextId)throw new Error('Completed continuation has no clip ID');
        await refreshClips();
        ui.jobStatus='Loading native continuation into playback…';
        await selectClip(nextId,{preserveFrame:true,liveSwap:true,liveStart:commitFrame});
        // The buffer is usable only once the new clip is installed. This
        // includes any time spent fetching fitted frames after job-ready.
        if(ui.bufferHoldStart!==null){entry.buffer_hold_seconds=+(entry.buffer_hold_seconds+(performance.now()-ui.bufferHoldStart)/1000).toFixed(3);ui.bufferHoldStart=null;}
        ui.wantedResume=false;
        entry.result_clip_id=nextId;
        entry.completed_perf=performance.now();
        ui.pendingVisibleContinuations.push(entry);
        ui.activeContinuation=null;
        ui.jobStatus='Native continuation ready';
        if(ui.clipId!==nextId)throw new Error('Generated clip could not be loaded');
        break;
      }
      if(status.status==='failed'||status.status==='error')throw new Error(status.error||'Native generation failed');
      ui.jobStatus=`Native Core generation · ${cleanLabel(status.status||'running')}`;
    }
  }catch(e){ui.jobStatus=`Continuation failed: ${e.message}`;entry.error=e.message;entry.completed_perf=performance.now();ui.job=null;ui.wantedResume=false;
    if(ui.live){ui.live=false;ui.liveEvents.push({at_utc:new Date().toISOString(),at_perf:performance.now(),enabled:false,reason:'generation error',clip_id:ui.clipId,frame:ui.frame});}
    ui.queuedPrompt=null;
    if(ui.bufferHoldStart!==null){entry.buffer_hold_seconds=+(entry.buffer_hold_seconds+(performance.now()-ui.bufferHoldStart)/1000).toFixed(3);ui.bufferHoldStart=null;}
    ui.activeContinuation=null;
  }
  finally{
    clipSelect.disabled=false;
    for(const option of clipSelect.options)option.disabled=false;
    frameInput.max=String((ui.clip?.frames||1)-1);
    if(!entry.error&&ui.queuedPrompt){
      const queued=ui.queuedPrompt;ui.queuedPrompt=null;
      void requestContinuation(queued.prompt,'queued',queued);
    }
  }
}

let buttonRects=[];
const buttonDefs=[
  {id:'previous',label:'◀  PREV CLIP',action:()=>adjacent(-1)},
  {id:'play',label:'▶  PLAY',action:togglePlay},
  {id:'next',label:'NEXT CLIP  ▶',action:()=>adjacent(1)},
  {id:'reset',label:'↺  RESTART',action:reset},
  {id:'camera',label:'◉  CAMERA',action:cycleCamera},
  {id:'live',label:'↻  LIVE OFF',action:toggleLive},
  {id:'record',label:'●  RECORD',action:toggleRecord},
];
for(const def of buttonDefs){
  const button=document.createElement('button');button.type='button';button.textContent=def.label;
  button.setAttribute('aria-label',def.label);button.addEventListener('click',def.action);
  controls.appendChild(button);def.node=button;
}
function layoutButtons(){
  const compact=ui.w<760,margin=compact?15:30,gap=compact?6:10,h=compact?33:42;
  const y=ui.h-(compact?59:75);
  const width=Math.min(140,(ui.w-margin*2-gap*(buttonDefs.length-1))/buttonDefs.length);
  buttonRects=buttonDefs.map((def,i)=>({def,x:margin+i*(width+gap),y,w:width,h}));
  for(const b of buttonRects){Object.assign(b.def.node.style,{left:`${b.x}px`,top:`${b.y}px`,width:`${b.w}px`,height:`${b.h}px`});}
  const rightCard=compact?172:236;
  frameInput.style.left=`${ui.w-margin-rightCard+14}px`;
  frameInput.style.top='45px';
  frameInput.style.width=compact?'78px':'98px';
}
resize();
clipSelect.addEventListener('change',()=>{if(ui.live)toggleLive();selectClip(clipSelect.value);});
function commitFrameInput(){if(frameInput.value.trim()==='')return;const frame=Number(frameInput.value);if(Number.isFinite(frame)&&ui.clip){ui.playing=false;ui.wantedResume=false;showFrame(frame);ui.playhead=ui.frame/ui.clip.fps;}}
frameInput.addEventListener('input',commitFrameInput);
frameInput.addEventListener('change',commitFrameInput);
canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect();ui.pointerX=e.clientX-r.left;ui.pointerY=e.clientY-r.top;if(ui.dragging)ui.orbit+=(e.movementX||0)*.007;});
canvas.addEventListener('pointerdown',e=>{const r=canvas.getBoundingClientRect();ui.dragging=true;ui.dragStart=[e.clientX-r.left,e.clientY-r.top];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',e=>{ui.dragging=false;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  const click=ui.dragStart&&Math.abs(x-ui.dragStart[0])+Math.abs(y-ui.dragStart[1])<9;ui.dragStart=null;
  if(!click)return;
  if(y>=ui.h-145&&y<ui.h-98&&ui.clip){const t=Math.max(0,Math.min(1,(x-30)/(ui.w-60)));ui.playing=false;ui.wantedResume=false;showFrame(Math.round(t*(ui.clip.frames-1)));ui.playhead=ui.frame/ui.clip.fps;return;}
  buttonRects.find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h)?.def.action();
});
window.addEventListener('keydown',e=>{
  if(document.activeElement===actionPrompt||document.activeElement===frameInput||document.activeElement===clipSelect)return;
  if([' ','ArrowLeft','ArrowRight'].includes(e.key))e.preventDefault();
  if(e.key===' ')togglePlay();if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);
  if(e.key.toLowerCase()==='c')cycleCamera();if(e.key.toLowerCase()==='r')toggleRecord();
});

function toggleRecord(){
  if(ui.record?.state==='recording'){ui.recordEndedWall=new Date().toISOString();ui.recordEndedPerf=performance.now();ui.record.stop();return;}
  if(!canvas.captureStream||!window.MediaRecorder){ui.recordStatus='Recording unavailable in this browser';return;}
  const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(type=>MediaRecorder.isTypeSupported(type));
  if(!mime){ui.recordStatus='WebM capture unsupported';return;}
  ui.chunks=[];
  const recorder=new MediaRecorder(canvas.captureStream(30),{mimeType:mime,videoBitsPerSecond:9_000_000});
  recorder.ondataavailable=e=>{if(e.data.size)ui.chunks.push(e.data);};
  recorder.onstop=async()=>{
    ui.recordStatus='Saving the actual canvas recording…';
    const blob=new Blob(ui.chunks,{type:mime});
    if(!blob.size){ui.recordStatus='No video frames captured';return;}
    const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='native-motion-review.webm';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),60000);
    const samples=[...ui.rafSamples].sort((a,b)=>a-b);
    const captureEnd=ui.recordEndedPerf??performance.now();
    const continuations=ui.generationLog.filter(entry=>{
      const during=value=>value!=null&&value>=ui.recordAt&&value<=captureEnd;
      return during(entry.submitted_perf)||during(entry.job_ready_perf)||during(entry.first_visible_perf)||
        (entry.submitted_perf<ui.recordAt&&(entry.completed_perf==null||entry.completed_perf>=ui.recordAt));
    }).map(entry=>{
      const row=Object.fromEntries(Object.entries(entry).filter(([key])=>!key.endsWith('_perf')));
      const submittedDuring=entry.submitted_perf>=ui.recordAt&&entry.submitted_perf<=captureEnd;
      const visibleDuring=entry.first_visible_perf!=null&&entry.first_visible_perf>=ui.recordAt&&entry.first_visible_perf<=captureEnd;
      row.recording_scope=submittedDuring?'submitted_during_recording':'preroll_overlap';
      row.submitted_at_recording_ms=Math.round(entry.submitted_perf-ui.recordAt);
      row.first_visible_in_recording=visibleDuring;
      if(!visibleDuring){
        row.first_visible_appended_frame_at_utc=null;row.first_visible_frame=null;
        row.first_visible_latency_ms=null;row.command_to_visible_ms=null;
        row.first_visible_outcome=entry.first_visible_outcome==='pending'?'pending_at_recording_end':'outside_recording';
      }
      return row;
    });
    const telemetry={recording_started_at_utc:ui.recordStartedWall,recording_ended_at_utc:ui.recordEndedWall||new Date().toISOString(),
      duration_ms:Math.round(captureEnd-ui.recordAt),clip_id:ui.clipId,source_frames:ui.clip?.frames||0,
      recording_bytes:blob.size,renderer:'WebGL scene and Canvas2D controls captured from the visible canvas',
      raf:{sample_count:samples.length,gaps_over_100_ms:ui.rafGapCount,
        max_gap_ms:samples.length?+samples.at(-1).toFixed(2):null,
        p95_gap_ms:samples.length?+samples[Math.min(samples.length-1,Math.floor(samples.length*.95))].toFixed(2):null,
        excluded_loading_frames:ui.excludedLoadingFrames},
      live_events:ui.liveEvents.filter(event=>event.at_perf>=ui.recordAt&&event.at_perf<=captureEnd)
        .map(({at_perf,...event})=>({...event,at_recording_ms:Math.round(at_perf-ui.recordAt)})),
      continuations};
    const outcomes=await Promise.allSettled([
      json('/api/grounded/recording',{method:'POST',headers:{'Content-Type':mime},body:blob}),
      json('/api/grounded/telemetry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(telemetry)}),
    ]);
    ui.recordStatus=outcomes.every(result=>result.status==='fulfilled')?'Actual canvas video and telemetry saved':
      `Video downloaded; server save: ${outcomes.map(result=>result.status==='fulfilled'?'ok':result.reason.message).join(' / ')}`;
  };
  recorder.start(1000);ui.record=recorder;ui.recordAt=performance.now();ui.recordStartedWall=new Date().toISOString();
  ui.recordEndedWall=null;ui.recordEndedPerf=null;ui.rafSamples=[];ui.rafGapCount=0;ui.excludedLoadingFrames=0;
  ui.recordStatus='Recording exact native playback and controls';
}

function updateCamera(dt){
  if(!ui.clip)return;
  const p=ui.clip.fitted_positions;
  const a=p?.[0]?.[ui.frame]?.[0],b=ui.clip.actors>1?p?.[1]?.[ui.frame]?.[0]:null;
  const cx=a&&b?(a[0]+b[0])/2:a?.[0]||0;
  const cy=a&&b?(a[1]+b[1])/2:a?.[1]||.9;
  const cz=a&&b?(a[2]+b[2])/2:a?.[2]||0;
  let baseAngle=0,pairDistance=0;
  if(a&&b){
    const dx=b[0]-a[0],dz=b[2]-a[2];pairDistance=Math.hypot(dx,dz);
    if(pairDistance>.05){let px=dz/pairDistance,pz=-dx/pairDistance;if(pz<0){px=-px;pz=-pz;}baseAngle=Math.atan2(px,pz);}
  }
  const angle=baseAngle+ui.orbit,side=Math.sin(angle),front=Math.cos(angle);
  const distance=Math.max(ui.camera==='CLOSE'?3.2:ui.camera==='FRONT'?3.8:5.25,pairDistance*(ui.camera==='CLOSE'?1.3:1.5));
  const lateral=ui.camera==='STAGE'?.45:0;
  cameraGoal.set(cx+lateral*front+distance*side,cy+1,cz-lateral*side+distance*front);
  targetGoal.set(cx,cy-.05,cz);
  if(!ui.cameraInitialized){camera.position.copy(cameraGoal);cameraTarget.copy(targetGoal);ui.cameraInitialized=true;}
  const ease=1-Math.exp(-Math.min(dt,.07)*2.8);
  camera.position.lerp(cameraGoal,ease);cameraTarget.lerp(targetGoal,ease);
  camera.lookAt(cameraTarget);
  camera.fov=THREE.MathUtils.lerp(camera.fov,ui.camera==='CLOSE'?43:48,ease);camera.updateProjectionMatrix();
}

function rect(x,y,w,h,r,fill,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,r);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.lineWidth=1;ctx.strokeStyle=stroke;ctx.stroke();}}
function text(value,x,y,size=12,color='#f1f4ee',weight=600,align='left'){
  ctx.font=`${weight} ${size}px Inter,ui-sans-serif,sans-serif`;ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(String(value),x,y);
}
function mono(value,x,y,size=10,color='#b8c9bc',align='left'){
  ctx.font=`600 ${size}px ui-monospace,Menlo,monospace`;ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(String(value),x,y);
}
function clampText(value,max=42){const str=String(value||'');return str.length>max?str.slice(0,max-1)+'…':str;}
function wrapText(value,maxChars=49){const words=String(value||'').split(/\s+/),lines=[''];for(const word of words){let i=lines.length-1;if((lines[i]+' '+word).trim().length>maxChars&&lines[i]){lines.push(word);}else lines[i]=(lines[i]+' '+word).trim();if(lines.length===3)break;}return lines;}
function drawHud(now){
  const w=ui.w,h=ui.h,mobile=w<760,margin=mobile?15:30;
  const shade=ctx.createLinearGradient(0,0,0,h);shade.addColorStop(0,'#071410be');shade.addColorStop(.25,'#0714100a');shade.addColorStop(.57,'#07141000');shade.addColorStop(1,'#06120fe8');ctx.fillStyle=shade;ctx.fillRect(0,0,w,h);
  rect(margin,20,mobile?210:330,mobile?76:113,5,'#0e201aca','#a9c1ac55');
  mono('GENERATED CORE27 MOTION  /  17-BONE RETARGET',margin+14,39,mobile?8:10,'#d3e7ce');
  text(clampText(cleanLabel(ui.clip?.label||'Loading clip'),mobile?24:34),margin+14,mobile?64:67,mobile?15:20,'#f6faf3',800);
  if(!mobile){
    mono(`${ui.clip?.actors||0} GENERATED HUMAN${ui.clip?.actors===1?'':'S'}  ·  ${ui.clip?.fps||20} FPS`,margin+14,88,10,'#dce9d8');
    mono(clampText(ui.clip?.source||'Saved native source',42),margin+14,108,10,'#adcfb7');
  }
  const right=mobile?172:236;
  rect(w-margin-right,20,right,mobile?76:113,5,'#0e201aca','#a9c1ac55');
  mono('EXACT SOURCE FRAME',w-margin-right+14,39,mobile?8:10,'#d3e7ce');
  text(String(ui.frame).padStart(4,'0'),w-margin-right+14,69,mobile?22:29,'#fff',800);
  mono(`/ ${String(Math.max(0,(ui.clip?.frames||1)-1)).padStart(4,'0')}`,w-margin-14,67,mobile?9:11,'#b2c9b8','right');
  mono(ui.camera+' CAMERA  ·  '+(ui.playing?'PLAYING':'PAUSED'),w-margin-14,mobile?88:104,mobile?8:10,'#dce9d8','right');
  if(!mobile&&ui.clip){
    const boxY=147;
    rect(margin,boxY,Math.min(390,w*.34),ui.clip.prompt?100:71,5,'#0e201aa8','#a9c1ac44');
    mono('MODEL DIRECTION',margin+14,boxY+21,9,'#bad4b8');
    const lines=wrapText(ui.clip.prompt||ui.clip.metadata?.prompt||'Saved native body motion',50);
    lines.slice(0,3).forEach((line,i)=>text(line,margin+14,boxY+45+i*18,11,'#edf5ec',500));
    mono(`SOURCE: ${clampText(ui.clip.source||'native Core27',43)}`,margin+14,boxY+(ui.clip.prompt?88:61),8,'#b6cdb6');
  }
  if(!mobile){
    const request=actionPrompt.value.trim();
    const requestState=liveUnavailableReason(ui.clip)?'REPLAY ONLY':
      !request?'NOT SUBMITTED':ui.queuedPrompt?.prompt===request?'QUEUED':
      ui.lastSubmittedPrompt===request?(ui.job?'SUBMITTED · GENERATING':'SUBMITTED'):'NOT SUBMITTED';
    const cardY=ui.clip?.prompt?260:231;
    const cardWidth=Math.min(390,w*.34);
    rect(margin,cardY,cardWidth,62,5,'#0e201ac4','#a9c1ac55');
    mono(`NEXT REQUEST  ·  ${requestState}`,margin+14,cardY+19,9,'#cce6c6');
    wrapText(request||'No next action entered',Math.floor((cardWidth-28)/6)).slice(0,2)
      .forEach((line,i)=>text(line,margin+14,cardY+40+i*15,10,'#edf5ec',500));
  }
  if(ui.jobStatus){rect(w/2-178,26,356,38,19,'#274c3fdc','#a9d8ba88');mono(clampText(ui.jobStatus,47),w/2,49,10,'#ebfff0','center');}
  if(ui.queuedPrompt){rect(w/2-178,68,356,25,12,'#173a31df','#a9d8ba77');mono(`NEXT QUEUED: ${clampText(ui.queuedPrompt.prompt,35)}`,w/2,84,9,'#e6ffe8','center');}
  if(ui.record?.state==='recording'){
    rect(w/2-70,70,140,29,15,'#8a2826e0','#e78682');
    mono(`● REC ${new Date(now-ui.recordAt).toISOString().slice(14,19)}`,w/2,89,10,'#fff','center');
    mono(`FRAME GAPS >100MS  ${ui.rafGapCount}`,w/2,115,9,'#f1ddd5','center');
  }
  if(ui.recordStatus)mono(clampText(ui.recordStatus,mobile?36:70),w/2,h-(mobile?73:108),mobile?8:10,'#e2efdf','center');
  if(ui.sourceError)mono(clampText(ui.sourceError,76),w/2,h*.5,11,'#ffd8cb','center');
  const progress=ui.clip?ui.frame/Math.max(1,ui.clip.frames-1):0;
  mono(`CLIP ${ui.clips.findIndex(c=>c.id===ui.clipId)+1} / ${ui.clips.length}   ·   ${ui.clip?Math.round(ui.clip.frames/ui.clip.fps):0} S`,margin,h-(mobile?111:144),9,'#e0ebe0');
  mono('DRAG TIMELINE · EDIT FRAME NUMBER',w-margin,h-(mobile?111:144),9,'#b9ccba','right');
  const barY=h-(mobile?100:132);
  rect(margin,barY,w-2*margin,7,3,'#4c655cb0');
  rect(margin,barY,Math.max(4,(w-2*margin)*progress),7,3,'#d9ecc3');
  ctx.beginPath();ctx.arc(margin+(w-2*margin)*progress,barY+3.5,5,0,Math.PI*2);ctx.fillStyle='#f1f6e7';ctx.fill();
  for(const b of buttonRects){
    const hover=ui.pointerX>=b.x&&ui.pointerX<=b.x+b.w&&ui.pointerY>=b.y&&ui.pointerY<=b.y+b.h;
    const rec=b.def.id==='record'&&ui.record?.state==='recording';
    const live=b.def.id==='live'&&ui.live;
    const unavailable=b.def.id==='live'&&Boolean(liveUnavailableReason(ui.clip));
    rect(b.x,b.y,b.w,b.h,4,rec?'#923734':live?'#416d49':hover?'#577963':'#1e332ae8',rec?'#efaaa2':live?'#d5efc4':hover?'#d5efc4':'#9bb8a066');
    let label=b.def.label;if(b.def.id==='play')label=ui.playing?'Ⅱ  PAUSE':'▶  PLAY';if(b.def.id==='live')label=unavailable?'↻  LIVE N/A':live?'↻  LIVE ON':'↻  LIVE OFF';if(rec)label='■  STOP & SAVE';
    if(b.def.node.textContent!==label){b.def.node.textContent=label;b.def.node.setAttribute('aria-label',label);}
    b.def.node.disabled=unavailable;
    b.def.node.style.background=rec?'#923734':live?'#416d49':'';
    mono(label,b.x+b.w/2,b.y+b.h/2+4,mobile?8:10,'#edf5e9','center');
  }
  mono('GENERATED BODY MOTION · APPROXIMATE RETARGET · NO FINGER/FACE ANIMATION',margin,h-12,mobile?7:9,'#b3cbb6');
  if(!mobile)mono(clampText(ui.sceneProvenance,45),w-margin,h-12,9,'#9eb5a2','right');
}

function animate(now){
  requestAnimationFrame(animate);
  const rawFrameMs=now-ui.lastTick,dt=Math.min(.1,rawFrameMs/1000);ui.lastTick=now;
  if(ui.record?.state==='recording'){
    if(ui.ready&&ui.clip&&!ui.loadingClip){ui.rafSamples.push(rawFrameMs);if(rawFrameMs>100)ui.rafGapCount++;}
    else ui.excludedLoadingFrames++;
  }
  if(ui.playing&&ui.clip&&!ui.loadingClip){
    maybeAutoContinue();
    ui.playhead+=dt;
    const frame=Math.floor(ui.playhead*ui.clip.fps);
    const end=ui.job&&ui.activeContinuation?ui.activeContinuation.prefix_frames-1:ui.clip.frames-1;
    if(frame>=end){
      showFrame(end);
      ui.playhead=end/ui.clip.fps;
      if(ui.job){ui.wantedResume=true;ui.jobStatus='At committed frame · waiting for native continuation';
        if(ui.bufferHoldStart===null)ui.bufferHoldStart=performance.now();}
      else ui.playing=false;
    }else if(frame!==ui.frame)showFrame(frame);
  }
  updateCamera(dt);
  renderer.render(scene,camera);
  ctx.setTransform(ui.dpr,0,0,ui.dpr,0,0);
  ctx.drawImage(renderer.domElement,0,0,ui.w,ui.h);
  drawHud(now);
  markRenderedContinuations();
}
requestAnimationFrame(animate);
initialize();
