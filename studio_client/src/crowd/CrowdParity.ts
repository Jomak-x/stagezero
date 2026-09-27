/** Real WebGL2 transform-feedback parity check of the same GPU deformation source. */
import * as THREE from 'three';
import {CROWD_AFFINE_GLSL,CROWD_POSITION_GLSL,geometryFor,nativeParts} from './CrowdRenderer';
import type {CrowdManifest} from './CrowdRenderer';
import {skinNativePairFrame} from '../mesh/NativePairPlayback';
export function gpuPositionParity(renderer:THREE.WebGLRenderer,manifest:CrowdManifest,affine:Float32Array){
 const gl=renderer.getContext() as WebGL2RenderingContext, resources:WebGLBuffer[]=[];
 const diagnostics:unknown[]=[];const check=(stage:string)=>{const errors:number[]=[];let error;while((error=gl.getError())!==gl.NO_ERROR)errors.push(error);if(errors.length)diagnostics.push({stage,errors});};check('before-parity');
 const vs=gl.createShader(gl.VERTEX_SHADER)!,fs=gl.createShader(gl.FRAGMENT_SHADER)!,program=gl.createProgram()!,vao=gl.createVertexArray()!,texture=gl.createTexture()!,feedback=gl.createTransformFeedback()!;
 try{
  gl.shaderSource(vs,'#version 300 es\nprecision highp float;\n'+CROWD_AFFINE_GLSL.replaceAll('attribute ','in ').replaceAll('texture2D(','texture(')+'\nout vec3 parityPosition; void main(){'+CROWD_POSITION_GLSL+'parityPosition=transformed;gl_Position=vec4(0.,0.,0.,1.);}') ;gl.compileShader(vs);
  if(!gl.getShaderParameter(vs,gl.COMPILE_STATUS))throw new Error('Parity vertex shader: '+gl.getShaderInfoLog(vs));
  gl.shaderSource(fs,'#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1.);}');gl.compileShader(fs);gl.attachShader(program,vs);gl.attachShader(program,fs);gl.transformFeedbackVaryings(program,['parityPosition'],gl.INTERLEAVED_ATTRIBS);gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('Parity link: '+gl.getProgramInfoLog(program));
  gl.useProgram(program);gl.bindVertexArray(vao);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,66,manifest.totalFrames,0,gl.RGBA,gl.FLOAT,affine);gl.uniform1i(gl.getUniformLocation(program,'crowdTexture'),0);gl.uniform1f(gl.getUniformLocation(program,'crowdTextureHeight'),manifest.totalFrames);check('texture-and-uniforms');
  const parts=nativeParts(manifest,affine),idle=manifest.clips.find(c=>c.id.includes('idle'))!.offset;
  let maxError=0,values=0;const cases=[[0,1,.37,idle,1],[Math.floor(manifest.totalFrames/3),Math.floor(manifest.totalFrames/3)+1,.63,idle,.41],[0,1,.5,idle,0]];
  for(let p=0;p<parts.length;p++){
   const geometry=geometryFor(manifest.parts[p]),part=parts[p];
   for(const name of ['bind0','bind1','bind2','bind3','crowdBones','crowdWeights']){const attribute=geometry.getAttribute(name);const buffer=gl.createBuffer()!;resources.push(buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,attribute.array,gl.STATIC_DRAW);const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,attribute.itemSize,gl.FLOAT,false,0,0);}
   const output=gl.createBuffer()!;resources.push(output);gl.bindBuffer(gl.ARRAY_BUFFER,output);gl.bufferData(gl.ARRAY_BUFFER,part.vertexCount*3*4,gl.STREAM_READ);
   const a=new Float32Array(part.vertexCount*3),b=new Float32Array(a.length),rest=new Float32Array(a.length),actual=new Float32Array(a.length);
   for(const [first,second,alpha,idleFrame,blend] of cases){
    gl.vertexAttrib4f(gl.getAttribLocation(program,'crowdFrame'),first,second,alpha,idleFrame);gl.vertexAttrib2f(gl.getAttribLocation(program,'crowdBlend'),blend,0);gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,feedback);gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,output);gl.bindBuffer(gl.ARRAY_BUFFER,null);check('attributes-and-output');gl.enable(gl.RASTERIZER_DISCARD);gl.beginTransformFeedback(gl.POINTS);check('begin-transform-feedback');gl.drawArrays(gl.POINTS,0,part.vertexCount);check('draw-transform-feedback');gl.endTransformFeedback();gl.disable(gl.RASTERIZER_DISCARD);gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,null);gl.bindBuffer(gl.ARRAY_BUFFER,output);gl.getBufferSubData(gl.ARRAY_BUFFER,0,actual);check('read-transform-feedback');
    skinNativePairFrame(part,first,a);skinNativePairFrame(part,second,b);skinNativePairFrame(part,idleFrame,rest);
    if(p===0)diagnostics.push({case:[first,second,alpha,idleFrame,blend],actual:Array.from(actual.slice(0,9)),cpuFirst:Array.from(a.slice(0,9)),cpuSecond:Array.from(b.slice(0,9)),cpuIdle:Array.from(rest.slice(0,9))});
    for(let i=0;i<a.length;i++){const expected=rest[i]+((a[i]+(b[i]-a[i])*alpha)-rest[i])*blend;maxError=Math.max(maxError,Math.abs(actual[i]-expected));values++;}
   }geometry.dispose();
  }
  return {method:'WebGL2 transform feedback of the exact shared affine position shader vs shipped native CPU skinning',diagnostics,cases:cases.length,comparedScalarValues:values,maxAbsolutePositionErrorMeters:maxError,toleranceMeters:1e-5,passed:Number.isFinite(maxError)&&maxError<1e-5,normals:'Approximate weighted affine normals; not claimed equal to CPU triangle-normal recomputation'};
 }finally{gl.disable(gl.RASTERIZER_DISCARD);gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,null);gl.bindVertexArray(null);for(const b of resources)gl.deleteBuffer(b);gl.deleteTransformFeedback(feedback);gl.deleteTexture(texture);gl.deleteVertexArray(vao);gl.deleteProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);renderer.resetState();}
}
