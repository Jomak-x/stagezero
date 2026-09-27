/** Isolated crowd affine GPU renderer. Does not change the 1–3 actor player. */
import * as THREE from 'three';
import { NativePairGeometry, skinNativePairFrame } from '../mesh/NativePairPlayback';
import type { NativePairPart } from '../mesh/NativePairPlayback';
export interface CrowdClip { id:string; type?:string; fps:number; frames:number; offset:number; strideMeters:number; speed:number }
export interface CrowdPart { name:string; positions:number[]; indices:number[]; localBind:number[]; bones:number[]; weights:number[]; color?:string }
export interface CrowdManifest { schema?:string; upAxis:string; affineUrl:string; totalFrames:number; clips:CrowdClip[]; parts:CrowdPart[]; [key:string]:unknown }
export type Person = [number,number,number,number,number,number?];
export interface Trajectory { schema:string; dt:number; duration:number; agents:{id:string|number; group_id?:number; radius:number; gait:string; color_index:number}[]; frames:{t:number;people:Person[]}[]; [key:string]:unknown }
export const CROWD_AFFINE_GLSL=`uniform sampler2D crowdTexture; uniform float crowdTextureHeight;
attribute vec3 bind0; attribute vec3 bind1; attribute vec3 bind2; attribute vec3 bind3;
attribute vec4 crowdBones; attribute vec4 crowdWeights; attribute vec4 crowdFrame; attribute vec2 crowdBlend;
vec4 crowdRow(float bone,float row,float frame){return texture2D(crowdTexture,vec2((bone*3.0+row+0.5)/66.0,(frame+0.5)/crowdTextureHeight));}
vec4 crowdPose(float bone,float row){vec4 walk=mix(crowdRow(bone,row,crowdFrame.x),crowdRow(bone,row,crowdFrame.y),crowdFrame.z);return mix(crowdRow(bone,row,crowdFrame.w),walk,crowdBlend.x);}
vec3 crowdPoint(float bone,vec3 p){return vec3(dot(crowdPose(bone,0.0),vec4(p,1.0)),dot(crowdPose(bone,1.0),vec4(p,1.0)),dot(crowdPose(bone,2.0),vec4(p,1.0)));}
vec3 crowdNormal(float bone,vec3 n){return vec3(dot(crowdPose(bone,0.0).xyz,n),dot(crowdPose(bone,1.0).xyz,n),dot(crowdPose(bone,2.0).xyz,n));}
`;
export const CROWD_POSITION_GLSL=`vec3 transformed=crowdWeights.x*crowdPoint(crowdBones.x,bind0)+crowdWeights.y*crowdPoint(crowdBones.y,bind1)+crowdWeights.z*crowdPoint(crowdBones.z,bind2)+crowdWeights.w*crowdPoint(crowdBones.w,bind3);`;
const palette = ['#cf6443','#447686','#d4b059','#48566d','#759278','#925872','#b7b5a9','#434b45','#bb8972','#426278'];
export function geometryFor(part:CrowdPart) {
  const n=part.positions.length/3;
  if (!Number.isInteger(n)||!n||part.localBind.length!==n*12||part.weights.length!==n*4||part.bones.length!==n*4) throw new Error('Malformed native crowd mesh');
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(part.positions,3)); g.setIndex(part.indices); g.computeVertexNormals();
  for(let j=0;j<4;j++) {const a=new Float32Array(n*3);for(let v=0;v<n;v++)a.set(part.localBind.slice(v*12+j*3,v*12+j*3+3),v*3);g.setAttribute(`bind${j}`,new THREE.BufferAttribute(a,3));}
  g.setAttribute('crowdBones',new THREE.Float32BufferAttribute(part.bones,4));g.setAttribute('crowdWeights',new THREE.Float32BufferAttribute(part.weights,4));return g;
}
export class CrowdRenderer {
  group=new THREE.Group(); texture:THREE.DataTexture; meshes:THREE.InstancedMesh[]=[]; frames:THREE.InstancedBufferAttribute; blends:THREE.InstancedBufferAttribute;
  manifest:CrowdManifest; count:number; triangleCount=0; blobs:THREE.InstancedMesh; private weightTrajectory:Trajectory|undefined; private weights:number[][]=[]; private matrix=new THREE.Matrix4(); private q=new THREE.Quaternion(); private p=new THREE.Vector3(); private scale=new THREE.Vector3(); private axis=new THREE.Vector3(0,1,0);
  constructor(manifest:CrowdManifest, affine:Float32Array, count:number) {
    if(affine.length!==manifest.totalFrames*22*12 || !affine.every(Number.isFinite))throw new Error('Malformed crowd affine animation');
    if(manifest.upAxis!=='y')throw new Error('Crowd assets must be canonical Y-up, facing +Z');
    this.manifest=manifest;this.count=count;
    const blobGeo=new THREE.CircleGeometry(1,24);blobGeo.rotateX(-Math.PI/2);
    const blobMat=new THREE.ShaderMaterial({transparent:true,depthWrite:false,vertexShader:'varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*instanceMatrix*vec4(position,1.0);}',fragmentShader:'varying vec2 vUv; void main(){float r=length(vUv-.5)*2.;float a=(1.-smoothstep(.15,1.,r))*.23;gl_FragColor=vec4(.08,.09,.10,a);}'});
    this.blobs=new THREE.InstancedMesh(blobGeo,blobMat,count);this.blobs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.blobs.frustumCulled=false;this.group.add(this.blobs);
    this.texture=new THREE.DataTexture(affine,66,manifest.totalFrames,THREE.RGBAFormat,THREE.FloatType);this.texture.needsUpdate=true;
    this.frames=new THREE.InstancedBufferAttribute(new Float32Array(count*4),4).setUsage(THREE.DynamicDrawUsage);
    this.blends=new THREE.InstancedBufferAttribute(new Float32Array(count*2),2).setUsage(THREE.DynamicDrawUsage);
    for(const part of manifest.parts){
      const geo=geometryFor(part);geo.setAttribute('crowdFrame',this.frames);geo.setAttribute('crowdBlend',this.blends);
      const mat=new THREE.MeshStandardMaterial({color:part.color||'#ffffff',roughness:0.85,metalness:0.02,side:THREE.DoubleSide});
      mat.onBeforeCompile=(shader)=>{
        shader.uniforms.crowdTexture={value:this.texture};shader.uniforms.crowdTextureHeight={value:manifest.totalFrames};
        shader.vertexShader=CROWD_AFFINE_GLSL+shader.vertexShader;
        shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>',CROWD_POSITION_GLSL);
        shader.vertexShader=shader.vertexShader.replace('#include <beginnormal_vertex>',`vec3 objectNormal=normalize(crowdWeights.x*crowdNormal(crowdBones.x,normal)+crowdWeights.y*crowdNormal(crowdBones.y,normal)+crowdWeights.z*crowdNormal(crowdBones.z,normal)+crowdWeights.w*crowdNormal(crowdBones.w,normal));`);
      };
      mat.customProgramCacheKey=()=> 'crowd-affine-v1';
      const mesh=new THREE.InstancedMesh(geo,mat,count);mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.frustumCulled=false;this.triangleCount+=part.indices.length/3*count;
      for(let i=0;i<count;i++)mesh.setColorAt(i,new THREE.Color(palette[i%palette.length]));
      this.meshes.push(mesh);this.group.add(mesh);
    }
  }
  update(trajectory:Trajectory,time:number, distances:number[][]){
    if(this.weightTrajectory!==trajectory){this.weights=locomotionWeights(trajectory);this.weightTrajectory=trajectory;}
    const index=Math.min(trajectory.frames.length-2,Math.max(0,Math.floor(time/trajectory.dt))), a=trajectory.frames[index], b=trajectory.frames[index+1];
    const alpha=THREE.MathUtils.clamp((time-a.t)/trajectory.dt,0,1), walk=this.manifest.clips.filter(c=>c.type==='walk'||c.id.includes('walk')), idle=this.manifest.clips.find(c=>c.id.includes('idle'))||this.manifest.clips[0];
    for(let i=0;i<this.count;i++){
      const pa=a.people[i],pb=b.people[i],actor=trajectory.agents[i],clip=walk.find(c=>c.id.includes(actor.gait==='brisk'?'brisk':'relaxed'))||walk[actor.gait==='brisk'?Math.min(1,walk.length-1):0];
      const distance=THREE.MathUtils.lerp(distances[index][i],distances[index+1][i],alpha);
      const phase=(distance/(Math.max(0.3,clip.strideMeters)*(0.96+(i*7%9)/100))*clip.frames+i*0.61803398875*clip.frames)%clip.frames;
      this.frames.setXYZW(i,clip.offset+Math.floor(phase),clip.offset+(Math.floor(phase)+1)%clip.frames,phase%1,idle.offset+(Math.floor(time*idle.fps+i*7)%idle.frames));
      // A smooth velocity-driven transition, explicitly procedural, keeps stopped roots in a real idle pose.
      this.blends.setXY(i,THREE.MathUtils.lerp(this.weights[index][i],this.weights[index+1][i],alpha),0);
      let delta=pb[2]-pa[2];delta=Math.atan2(Math.sin(delta),Math.cos(delta));
      this.p.set(THREE.MathUtils.lerp(pa[0],pb[0],alpha),0,THREE.MathUtils.lerp(pa[1],pb[1],alpha));this.q.setFromAxisAngle(this.axis,pa[2]+delta*alpha);
      const height=0.94+(i*17%13)/100;this.scale.set(0.96+(i*7%9)/100,height,0.96+(i*7%9)/100);this.matrix.compose(this.p,this.q,this.scale);
      for(const mesh of this.meshes){mesh.setMatrixAt(i,this.matrix); if(time===0)mesh.setColorAt(i,new THREE.Color(palette[actor.color_index%palette.length]));}
      this.p.y=.012;this.scale.set(.40,1,.28);this.matrix.compose(this.p,this.q,this.scale);this.blobs.setMatrixAt(i,this.matrix);
    }
    this.frames.needsUpdate=this.blends.needsUpdate=this.blobs.instanceMatrix.needsUpdate=true;for(const mesh of this.meshes)mesh.instanceMatrix.needsUpdate=true;
  }
  dispose(){this.blobs.geometry.dispose();(this.blobs.material as THREE.Material).dispose();this.texture.dispose();for(const mesh of this.meshes){mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();}this.group.clear();}
}
/** Deterministic bounded transition rate prevents collision-correction speed spikes from snapping feet. */
export function locomotionWeights(trajectory:Trajectory){
 const weights:number[][]=[];
 for(let f=0;f<trajectory.frames.length;f++){const dt=f?trajectory.frames[f].t-trajectory.frames[f-1].t:0;weights.push(trajectory.frames[f].people.map((person,i)=>{const target=THREE.MathUtils.smoothstep(person[3],.025,.42);return f?weights[f-1][i]+THREE.MathUtils.clamp(target-weights[f-1][i],-1.8*dt,1.8*dt):target;}));}
 return weights;
}
export function accumulatedDistances(trajectory:Trajectory){return trajectory.frames.map((frame,f)=>frame.people.map((p)=>p[5]??(f?0:0))).map((_,f,all)=>{if(f)for(let i=0;i<all[f].length;i++){const a=trajectory.frames[f-1].people[i],b=trajectory.frames[f].people[i];all[f][i]=all[f-1][i]+Math.hypot(b[0]-a[0],b[1]-a[1]);}return all[f];});}
export function nativeParts(manifest:CrowdManifest,affine:Float32Array):NativePairPart[]{
 const linear=new Float32Array(manifest.totalFrames*22*9),targets=new Float32Array(manifest.totalFrames*22*3);
 for(let f=0;f<manifest.totalFrames;f++)for(let b=0;b<22;b++)for(let r=0;r<3;r++){const src=(f*22+b)*12+r*4;linear.set(affine.subarray(src,src+3),(f*22+b)*9+r*3);targets[(f*22+b)*3+r]=affine[src+3];}
 return manifest.parts.map(p=>({name:p.name,vertexCount:p.positions.length/3,localBind:new Float32Array(p.localBind),bones:new Uint16Array(p.bones),weights:new Float32Array(p.weights),linear,targets,frames:manifest.totalFrames}));
}
const stats=(v:number[])=>{const s=[...v].sort((a,b)=>a-b);return {median:s[Math.floor(s.length*.5)],p95:s[Math.floor(s.length*.95)],p99:s[Math.floor(s.length*.99)],samples:s.length};};
/** CPU microbenchmark of the unmodified shipped native functions, not a browser FPS claim. */
export async function benchmarkNative(manifest:CrowdManifest,affine:Float32Array,samples=30){
 const parts=nativeParts(manifest,affine),out:unknown[]=[];
 for(const count of [16,32,64,100]){
  const clones=Array.from({length:count},()=>manifest.parts.map((p,i)=>({part:parts[i],output:new Float32Array(parts[i].vertexCount*3),geometry:geometryFor(p)})));
  const updaters=clones.map(row=>row.map(c=>new NativePairGeometry(c.geometry))), skin:number[]=[],full:number[]=[];
  for(let f=0;f<samples+5;f++){
   let now=performance.now();for(const row of clones)for(const c of row)skinNativePairFrame(c.part,f%manifest.totalFrames,c.output);if(f>=5)skin.push(performance.now()-now);
   now=performance.now();for(let a=0;a<count;a++)for(let p=0;p<parts.length;p++)updaters[a][p].update(parts[p],{frame:f+.25,first:f%manifest.totalFrames,second:(f+1)%manifest.totalFrames,alpha:.25,revision:1,playing:true,capturing:false});if(f>=5)full.push(performance.now()-now);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  out.push({actors:count,verticesPerActor:parts.reduce((n,p)=>n+p.vertexCount,0),skinOnlyMs:stats(skin),nativeGeometryMs:stats(full)});
  for(const row of clones)for(const c of row)c.geometry.dispose();for(const row of updaters)for(const u of row)u.dispose();
 }
 return {kind:'actual-native-functions CPU microbenchmark; no rendering',userAgent:navigator.userAgent,results:out};
}
