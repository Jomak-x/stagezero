import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { crowdDemoAssets, crowdDemoFiles } from '../crowdDemoAssets.mjs';

test('production and development expose the same exact saved-take payloads', () => {
  const root = mkdtempSync(join(tmpdir(), 'crowd-assets-'));
  try {
    for (const file of crowdDemoFiles) {
      mkdirSync(dirname(join(root, file)), { recursive: true });
      writeFileSync(join(root, file), Buffer.from(`payload:${file}`));
    }
    writeFileSync(join(root, 'private.json'), 'not a demo asset');
    const plugin = crowdDemoAssets(root), emitted = [];
    plugin.generateBundle.call({ emitFile: asset => emitted.push(asset) });
    assert.equal(emitted.length, 20);
    assert.equal(new Set(emitted.map(x => x.fileName)).size, 20);
    let handler;
    plugin.configureServer({ middlewares: { use: fn => { handler = fn; } } });
    for (const asset of emitted) {
      const response = { statusCode: 200, setHeader() {}, end(data) { this.data = data; } };
      handler({url: `/${asset.fileName}?v=1`, method: 'GET'}, response, () => assert.fail('unexpected fallthrough'));
      assert.deepEqual(response.data, asset.source);
    }
    for (const path of ['private.json', '../private.json', '%2e%2e/private.json', 'assets/../../private.json']) {
      const response = { statusCode: 200, end() {} };
      handler({url: `/demo-data/${path}`, method: 'GET'}, response, () => assert.fail('unexpected fallthrough'));
      assert.equal(response.statusCode, 404);
    }
    const response = { statusCode: 200, setHeader() {}, end() {} };
    handler({url: '/demo-data/city-112.json', method: 'POST'}, response, () => assert.fail('unexpected fallthrough'));
    assert.equal(response.statusCode, 405);
  } finally { rmSync(root, {recursive: true, force: true}); }
});
