import test from 'node:test';
import assert from 'node:assert/strict';
import { createMessageThrottle } from '../src/MessageThrottle.ts';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
test('different GUI controls are not lost within one throttle interval', async () => {
  const out = [];
  const queue = createMessageThrottle(message => out.push(message), 15);
  queue.send({type:'GuiUpdateMessage', uuid:'prompt', updates:{value:'walk'}});
  queue.send({type:'GuiUpdateMessage', uuid:'source', updates:{value:'Live ARDY'}});
  queue.send({type:'GuiUpdateMessage', uuid:'generate', updates:{value:true}});
  await sleep(30);
  assert.deepEqual(out.map(m => m.uuid), ['prompt','source','generate']);
  queue.dispose();
});
test('trailing seek is sent, intermediate values coalesce, dispose cancels pending work', async () => {
  const out = [];
  const queue = createMessageThrottle(message => out.push(message), 15);
  for (let i=0;i<10;i++) queue.send({type:'TimelineUpdateMessage',current_frame:i});
  await sleep(25);
  assert.deepEqual(out.map(m => m.current_frame), [0,9]);
  queue.send({type:'TimelineUpdateMessage',current_frame:10});
  queue.dispose();
  await sleep(25);
  assert.deepEqual(out.map(m => m.current_frame), [0,9]);
});
