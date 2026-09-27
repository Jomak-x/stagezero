import * as THREE from 'three';
import {CrowdRenderer, type CrowdManifest, type Trajectory} from '../crowd/CrowdRenderer';
export class DemoCrowdRenderer extends CrowdRenderer {
  private colors=['#ee754b','#2aaec2','#e9c64e','#7575c6','#5ead84','#da739b','#d7dfda','#718294','#bf8053','#668ccb'];
  private scratch = new THREE.Matrix4();
  private up=new THREE.Vector3(0,1,0);
  private position=new THREE.Vector3(); private rotation=new THREE.Quaternion(); private unit=new THREE.Vector3(1,1,1); private size=new THREE.Vector3();
  constructor(manifest:CrowdManifest, affine:Float32Array, count:number){
    super(manifest,affine,count);
    const colors=['#ee754b','#2aaec2','#e9c64e','#7575c6','#5ead84','#da739b','#d7dfda','#718294','#bf8053','#668ccb'];
    for(const mesh of this.meshes){
      (mesh.material as THREE.MeshStandardMaterial).color.set('#e8e9eb');
      (mesh.material as THREE.MeshStandardMaterial).roughness=.63;
      for(let i=0;i<count;i++)mesh.setColorAt(i,new THREE.Color(colors[i%colors.length]));
      if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;
    }
  }
  update(trajectory:Trajectory,time:number,distances:number[][]){
    super.update(trajectory,time,distances);
    if(time===0)for(const mesh of this.meshes){for(let i=0;i<this.count;i++)mesh.setColorAt(i,new THREE.Color((trajectory.agents[i] as any).stationary_work_role?'#2aaec2':this.colors[trajectory.agents[i].color_index%this.colors.length]));if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;}
    const f=Math.min(trajectory.frames.length-2,Math.max(0,Math.floor(time/trajectory.dt)));
    const a=trajectory.frames[f],b=trajectory.frames[f+1],alpha=THREE.MathUtils.clamp((time-a.t)/trajectory.dt,0,1);
    const idle=this.manifest.clips.find(c=>c.id==='idle')||this.manifest.clips[0];
    for(let i=0;i<this.count;i++){
      const pa=a.people[i] as number[],pb=b.people[i] as number[];
      const unitScale=i<6||Boolean((trajectory.agents[i] as any).unit_scale);
      const clipIndex=pa[6];
      const aPair=this.manifest.clips[pa[6]]?.type==='pair',bPair=this.manifest.clips[pb[6]]?.type==='pair';
      if(Number.isInteger(clipIndex)&&clipIndex>=0&&clipIndex<this.manifest.clips.length){
        const clip=this.manifest.clips[clipIndex],seconds=pa[7]+(pb[6]===clipIndex?(pb[7]-pa[7])*alpha:0);
        const phase=THREE.MathUtils.clamp(seconds*clip.fps,0,clip.frames-1);
        const pair=clip.id.includes('greeting-role');const neutral=pair?clip.offset+(seconds<(clip.frames-1)/clip.fps/2?0:clip.frames-1):idle.offset+(Math.floor(time*idle.fps+i*7)%idle.frames);
        this.frames.setXYZW(i,clip.offset+Math.floor(phase),clip.offset+Math.min(clip.frames-1,Math.floor(phase)+1),phase%1,neutral);
        const remaining=(clip.frames-1)/clip.fps-seconds;
        this.blends.setXY(i,THREE.MathUtils.smoothstep(Math.min(seconds,remaining),0,.25),0);
      }
      const y=THREE.MathUtils.lerp(pa[8]||0,pb[8]||0,alpha);
      if(y!==0||unitScale){for(const mesh of this.meshes){mesh.getMatrixAt(i,this.scratch);if(unitScale){this.scratch.decompose(this.position,this.rotation,this.size);if(aPair!==bPair)this.rotation.setFromAxisAngle(this.up,pa[2]);this.position.y=y;this.scratch.compose(this.position,this.rotation,this.unit);}else this.scratch.elements[13]=y;mesh.setMatrixAt(i,this.scratch);}this.blobs.getMatrixAt(i,this.scratch);this.scratch.elements[13]=((trajectory.agents[i] as any).crew==='flashmob'?0:y)+.012;this.blobs.setMatrixAt(i,this.scratch);}
    }
    this.frames.needsUpdate=this.blends.needsUpdate=true;
  }
}
