import {defineConfig} from 'vite';
import {resolve} from 'node:path';
export default defineConfig({cacheDir:resolve('../.runtime/crowd-vite-cache'),publicDir:false,plugins:[{name:'crowd-evidence-assets',configureServer(server){server.middlewares.use((req,_res,next)=>{if(req.url?.startsWith('/review/crowd-crossing/'))req.url='/@fs'+resolve('..')+req.url;next();});}}],server:{host:'127.0.0.1',port:24970,strictPort:true,fs:{allow:[resolve('..')]}},build:{outDir:'../.runtime/crowd-client-build',rollupOptions:{input:resolve('crowd.html')}}});
