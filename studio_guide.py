"""Short, task-oriented help for the studio inspector."""
GUIDE_HTML = '''
<style>
.sz-guide { margin: 8px 12px; color: #c5d7e3; font-size: 12px; line-height: 1.55; }
.sz-guide h3 { color: #a0ead5; font-size: 14px; margin: 8px 0 5px; }
.sz-guide p { margin: 5px 0 10px; }
.sz-guide details { border-top: 1px solid #304052; padding: 8px 0; }
.sz-guide summary { cursor: pointer; font-weight: 600; color: #e4eff6; }
.sz-guide b { color: #f0f6fa; }
</style>
<div class="sz-guide">
<h3>Make a take</h3>
<p>Choose <b>New take</b>, describe the motion, choose a length, and press <b>Generate motion</b>. Use the bottom timeline to play and inspect it. New takes keep your earlier versions and scene.</p>
<details open><summary>Edit motion</summary>
<p>Select a take, then click an action on the timeline to edit it or insert an action nearby. Change the direction and length, then press <b>Save action</b> or <b>Add action</b>. Later actions regenerate to follow the change. <b>Undo edit</b> restores the previous motion.</p>
<p>To continue from the end, press <b>Add action</b> in Motion.</p></details>
<details><summary>Make a different ending</summary>
<p>Open <b>Advanced: change ending as a new version</b> in Motion, or use <b>Change ending · new version</b> in Takes. Choose the time to keep, describe the new motion, and generate. The original take stays available.</p>
<p>In Takes, you can also duplicate, rename, remove, or save a shorter copy at the playhead. <b>Undo remove</b> restores the last removed take in this session.</p></details>
<details><summary>Choose motion length</summary>
<p><b>Auto</b> uses a duration from your direction, such as “wave for 6 seconds,” or estimates one. <b>Set new motion length</b> specifies the added action. <b>Set total scene length</b> includes motion already kept. Each generation adds at most 30 seconds.</p></details>
<details><summary>Build and view the scene</summary>
<p>Use <b>Scene</b> for props, lighting, and effects. Drag or two-finger scroll to pan; pinch to zoom. In <b>View</b>, choose Orbit or Look, or reset the camera. Click the viewport and use WASD and Q/E to move.</p></details>
<details><summary>Save your project</summary>
<p><b>Project → Save project + download</b> stores takes and the shared scene. You can reopen a saved project there. Opening another project backs up the current one first.</p></details>
</div>
'''
