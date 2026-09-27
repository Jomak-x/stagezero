import {defineConfig} from 'vite';
import {resolve} from 'node:path';
import {mkdir,writeFile} from 'node:fs/promises';
const root=resolve('..'),output=resolve('../review/crowd-demos/browser');
export default defineConfig({cacheDir:resolve('../.runtime/demo-vite-cache'),publicDir:false,plugins:[{name:'saved-crowd-demos',configureServer(server){server.middlewares.use(async(req,res,next)=>{
 if(req.url?.startsWith('/review/crowd-demos/')||req.url?.startsWith('/review/crowd-crossing/assets/')){req.url='/@fs'+root+req.url;return next();}
 if(!req.url?.startsWith('/__demo/'))return next();
 if(req.method!=='POST'||req.headers.origin!=='http://127.0.0.1:24985'){res.statusCode=403;res.end();return;}
 try{const chunks:Buffer[]=[];let size=0;for await(const c of req){size+=c.length;if(size>240*1024*1024)throw new Error('Capture exceeds240MB');chunks.push(Buffer.from(c));}
 const url=new URL(req.url,'http://127.0.0.1:24985');const name=(url.searchParams.get('name')||'demo').replace(/[^a-zA-Z0-9_-]/g,'').slice(0,90)+'-'+Date.now();const ext=url.pathname==='/__demo/video'?'.webm':url.pathname==='/__demo/still'?'.png':'.json';
 await mkdir(output,{recursive:true});await writeFile(resolve(output,name+ext),Buffer.concat(chunks));res.setHeader('Content-Type','application/json');res.end(JSON.stringify({saved:name+ext}));}catch(e){res.statusCode=400;res.end(String(e));}
 });}}],server:{host:'127.0.0.1',port:24985,strictPort:true,fs:{allow:[root]}},build:{outDir:'../.runtime/demos-client-build',rollupOptions:{input:resolve('demos.html')}}});
