import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export const crowdDemoFiles = [
  ...['crossing', 'city', 'station'].flatMap(scene =>
    [16, 32, 64, 100, 112, 128].map(count => `${scene}-${count}.json`)),
  'assets/manifest.json',
  'assets/affine.bin',
];
const allowed = new Set(crowdDemoFiles);

/** An explicit public asset list: never expose arbitrary review/runtime files. */
export function crowdDemoAssets(sourceRoot = resolve('../review/crowd-demos')) {
  return {
    name: 'saved-crowd-assets',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = (req.url || '').split('?')[0];
        if (!path.startsWith('/demo-data/')) return next();
        const file = path.slice('/demo-data/'.length);
        if (!allowed.has(file)) { res.statusCode = 404; res.end(); return; }
        if (req.method !== 'GET' && req.method !== 'HEAD') {
          res.statusCode = 405; res.setHeader('Allow', 'GET, HEAD'); res.end(); return;
        }
        try {
          const data = readFileSync(resolve(sourceRoot, file));
          res.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
          res.setHeader('Content-Length', data.length);
          res.end(req.method === 'HEAD' ? undefined : data);
        } catch (error) { next(error); }
      });
    },
    generateBundle() {
      for (const file of crowdDemoFiles) {
        this.emitFile({ type: 'asset', fileName: `demo-data/${file}`, source: readFileSync(resolve(sourceRoot, file)) });
      }
    },
  };
}
