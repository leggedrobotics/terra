/*!
 * Three.js 0.185.1 — The MIT License
 * Copyright © 2010-2026 three.js authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
import { TerraScene } from './scene.js';
import { validateReplay, validateFrame, terrainFacts, transitionFacts, actionName, formatReward, TYPES } from './data.js';

const $ = id => document.getElementById(id);
const number = value => new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
const text = (id, value) => { $(id).textContent = value; };
let scene, replay, index = 0, mode = 'replay', liveMode = null, imported = false, busy = false, playing = false, selected = null, eventTimer, lastTick = 0;
const embedded = $('terra-replay');
const offline = !!embedded?.textContent.trim();

function showError(error) { $('loading').hidden = true; text('error-message', error?.message || String(error)); $('error').hidden = false; }
function notify(message, duration = 3100) { clearTimeout(eventTimer); text('event', message); $('event').classList.add('visible'); eventTimer = setTimeout(() => $('event').classList.remove('visible'), duration); }
function frame() { return replay?.frames[index]; }
function manualReady() { return !!replay && mode === 'manual' && !imported && !busy && !playing && index === replay.frames.length - 1 && !frame().done; }

function updateControls() {
  const ready = manualReady(), current = frame();
  document.querySelectorAll('[data-action]').forEach(button => { button.disabled = !ready; });
  $('reset').disabled = busy || mode !== 'manual' || imported;
  $('export').disabled = !replay || busy;
  $('screenshot').disabled = !scene || !replay;
  $('open-file').disabled = busy;
  $('previous').disabled = !replay || index <= 0 || busy;
  $('next').disabled = !replay || index >= replay.frames.length - 1 || busy;
  $('play').disabled = !replay || replay.frames.length < 2 || busy;
  $('seek').disabled = !replay || replay.frames.length < 2 || busy;
  $('play').textContent = playing ? 'Ⅱ' : '▶'; $('play').setAttribute('aria-label', playing ? 'Pause replay' : 'Play replay');
  $('resume-live').hidden = !liveMode || offline || (!imported && !(mode === 'manual' && replay && index < replay.frames.length - 1));
  $('resume-live').disabled = busy;
  const historical = replay && index < replay.frames.length - 1;
  text('manual-status', busy ? 'STEPPING…' : imported || mode !== 'manual' ? 'REPLAY ONLY' : historical ? 'HISTORY' : current?.done ? 'ENDED' : playing ? 'PLAYBACK' : 'LIVE');
  text('session-mode', imported ? 'Imported replay' : mode === 'manual' ? (historical ? 'Manual · history' : 'Manual session') : 'Replay session');
}

function updateInspector() {
  if (!selected || !frame()) return;
  const { row, col } = selected, snapshot = frame(), { maps } = snapshot;
  if (row >= snapshot.grid.rows || col >= snapshot.grid.cols) { selected = null; return; }
  const inspector = $('cell-inspector'); inspector.replaceChildren();
  const eyebrow = document.createElement('span'); eyebrow.className = 'eyebrow'; eyebrow.textContent = 'CELL INSPECTOR'; inspector.append(eyebrow);
  const heading = document.createElement('div'); heading.className = 'cell-heading'; heading.textContent = `ROW ${row}  ·  COL ${col}`; inspector.append(heading);
  const rows = document.createElement('div'); rows.className = 'cell-data'; inspector.append(rows);
  const mask = (name, positive, negative) => maps[name] == null ? 'Unavailable' : maps[name][row][col] ? positive : negative;
  const target = maps.target[row][col];
  const fields = [
    ['Raw soil height', `${maps.action[row][col]} units`],
    ['Target', target < 0 ? `Dig ${-target}` : target > 0 ? `Dump ${target}` : 'Neutral'],
    ['Obstacle', maps.padding[row][col] ? 'Yes' : 'No'],
    ['Static dumping', mask('dumpability_static', 'Allowed', 'Prohibited')],
    ['Dumpable now', mask('dumpability', 'Yes', 'No')],
    ['Workspace', mask('interaction', 'Inside', 'Outside')],
    ['Traversability feature', maps.traversability == null ? 'Unavailable' : ({ '-1': 'Occupied (−1)', 0: 'Clear (0)', 1: 'Blocked (1)' })[Number(maps.traversability[row][col])]],
  ];
  for (const [name, value] of fields) { const label = document.createElement('span'), content = document.createElement('strong'); label.textContent = name; content.textContent = value; rows.append(label, content); }
}

function updateUI() {
  const snapshot = frame(), agent = snapshot.agents.find(item => item.id === snapshot.current_agent), facts = terrainFacts(snapshot);
  text('title', replay.metadata.title); text('source', replay.metadata.source);
  text('grid-spec', `${snapshot.grid.rows} × ${snapshot.grid.cols} · ${number(snapshot.grid.tile_size_m)} m / cell`);
  text('agent-count', `${snapshot.agents.length} machine${snapshot.agents.length === 1 ? '' : 's'}`);
  text('agent-id', String(agent.id + 1).padStart(2, '0')); text('machine-name', TYPES[agent.type]);
  text('embodiment', agent.action_type === 1 ? 'Wheeled' : 'Tracked');
  $('load').replaceChildren(document.createTextNode(number(agent.loaded))); const unit = document.createElement('small'); unit.textContent = ' units'; $('load').append(unit);
  text('reward', formatReward(snapshot.reward));
  text('outcome', snapshot.task_done ? 'Task complete' : snapshot.done ? 'Episode ended · task incomplete' : `Ready · machine ${agent.id + 1} acts next`); $('outcome').classList.toggle('done', snapshot.done);
  const tags = $('agent-list'); tags.replaceChildren(); tags.hidden = snapshot.agents.length <= 1;
  for (const item of snapshot.agents) { const tag = document.createElement('span'); tag.className = `agent-tag${item.id === snapshot.current_agent ? ' active' : ''}`; tag.textContent = `${String(item.id + 1).padStart(2, '0')} ${TYPES[item.type]} · ${item.loaded}`; tags.append(tag); }
  text('cut-units', `${number(facts.cut)} units`); text('fill-units', `${number(facts.fill)} units`);
  text('scene-caption', `${number(snapshot.grid.cols * snapshot.grid.tile_size_m)} × ${number(snapshot.grid.rows * snapshot.grid.tile_size_m)} m worksite · illustrative soil mounds`);
  text('step', snapshot.step); text('frame-count', `${index + 1} / ${replay.frames.length}`); text('action-label', actionName(snapshot, replay.frames[index - 1]));
  $('seek').max = String(replay.frames.length - 1); $('seek').value = String(index); $('seek').setAttribute('aria-valuetext', `Snapshot ${index + 1} of ${replay.frames.length}, step ${snapshot.step}`);
  const left = $('left-action'), right = $('right-action');
  left.dataset.action = agent.action_type === 1 ? '2' : '3'; right.dataset.action = agent.action_type === 1 ? '3' : '2';
  left.querySelector('.turn-label').textContent = agent.action_type === 1 ? 'Steer left' : 'Turn left'; right.querySelector('.turn-label').textContent = agent.action_type === 1 ? 'Steer right' : 'Turn right';
  left.title = `${agent.action_type === 1 ? 'Steer left' : 'Turn anticlockwise'} · Left or A`; right.title = `${agent.action_type === 1 ? 'Steer right' : 'Turn clockwise'} · Right or D`;
  text('work-label', agent.type === 2 ? (agent.shovel_lifted ? 'Lower shovel / dump' : 'Lift shovel') : agent.loaded > 0 ? 'Dump / transfer soil' : agent.type === 1 ? 'Dump (empty)' : 'Dig soil');
  for (const [name, map] of [['interaction', 'interaction'], ['restricted', 'dumpability_static'], ['dumpability', 'dumpability']]) { const checkbox = document.querySelector(`[data-layer="${name}"]`); checkbox.disabled = snapshot.maps[map] == null; checkbox.closest('label').title = snapshot.maps[map] == null ? 'This diagnostic layer is unavailable in the recording.' : ''; }
  updateInspector(); updateControls();
}

function showFrame(next, { animate = false, reset = false, announce = false } = {}) {
  if (!replay) return;
  const oldIndex = index; index = Math.max(0, Math.min(replay.frames.length - 1, next));
  const snapshot = frame();
  scene.setFrame(snapshot, { animate: animate && index === oldIndex + 1, reset, duration: Math.min(650, 800 / Number($('speed').value)) });
  updateUI();
  if (announce && index > 0) { const previous = replay.frames[index - 1]; if (snapshot.step > previous.step && !previous.done) notify(`${actionName(snapshot, previous)} · ${transitionFacts(previous, snapshot).message}`); else notify('Episode boundary · initial snapshot'); }
  else if (reset) { clearTimeout(eventTimer); $('event').classList.remove('visible'); }
}

function setPlaying(value) { playing = value; lastTick = performance.now(); updateControls(); }
function togglePlayback() { if (!replay || replay.frames.length < 2 || busy) return; if (!playing && index === replay.frames.length - 1) showFrame(0); setPlaying(!playing); }
function playbackTick(time) {
  if (playing && !document.hidden && time - lastTick >= 900 / Number($('speed').value)) {
    lastTick = time;
    if (index < replay.frames.length - 1) showFrame(index + 1, { animate: true, announce: true });
    if (index >= replay.frames.length - 1) setPlaying(false);
  }
  requestAnimationFrame(playbackTick);
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? { cache: 'no-store' } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const value = await response.json().catch(() => { throw new Error(`The server returned an unreadable response (${response.status}).`); });
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

function loadSession(session, { local = false } = {}) {
  validateReplay(session.replay); if (!['manual', 'replay'].includes(session.mode)) throw new Error('Unknown viewer session mode.');
  replay = session.replay; mode = session.mode; imported = local; index = 0; selected = null; playing = false;
  if (!local) liveMode = session.mode;
  $('cell-inspector').replaceChildren(); const eyebrow = document.createElement('span'); eyebrow.className = 'eyebrow'; eyebrow.textContent = 'CELL INSPECTOR'; const hint = document.createElement('span'); hint.className = 'cell-hint'; hint.textContent = 'Click the terrain to inspect a cell'; $('cell-inspector').append(eyebrow, hint);
  showFrame(mode === 'manual' && !local ? replay.frames.length - 1 : 0, { reset: true });
  $('loading').hidden = true; $('error').hidden = true;
}

async function performAction(action) {
  if (!manualReady()) return;
  busy = true; updateControls();
  try {
    const { frame: next } = await request('/api/action', { action }); validateFrame(next);
    replay.frames.push(next); showFrame(replay.frames.length - 1, { animate: true, announce: true });
  } catch (error) { showError(error); }
  finally { busy = false; updateControls(); }
}

async function resetSession() {
  if (busy || mode !== 'manual' || imported) return;
  busy = true; setPlaying(false); updateControls();
  try { loadSession(await request('/api/reset', {})); notify('Episode reset · ready to play'); }
  catch (error) { showError(error); }
  finally { busy = false; updateControls(); }
}

function download(url, filename, revoke = false) { const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove(); if (revoke) setTimeout(() => URL.revokeObjectURL(url), 1000); }
function exportReplay() { if (!replay) return; const blob = new Blob([JSON.stringify(replay)], { type: 'application/json' }); download(URL.createObjectURL(blob), 'terra-replay.json', true); notify(`Exported ${replay.frames.length} recorded snapshots`); }

function bindControls() {
  document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', event => { performAction(Number(button.dataset.action)); if (event.detail > 0) $('viewport').focus({ preventScroll: true }); }));
  $('reset').addEventListener('click', event => { resetSession(); if (event.detail > 0) $('viewport').focus({ preventScroll: true }); });
  $('previous').addEventListener('click', () => { setPlaying(false); showFrame(index - 1); });
  $('next').addEventListener('click', () => { setPlaying(false); showFrame(index + 1, { animate: true, announce: true }); });
  $('play').addEventListener('click', togglePlayback);
  $('seek').addEventListener('input', () => { setPlaying(false); showFrame(Number($('seek').value)); });
  $('speed').addEventListener('change', () => { lastTick = performance.now(); });
  $('camera-home').addEventListener('click', () => scene?.home()); $('brand-home').addEventListener('click', event => { event.preventDefault(); scene?.home(); });
  $('camera-top').addEventListener('click', () => scene?.top()); $('camera-follow').addEventListener('click', () => scene?.setFollow(!scene.follow));
  $('quality').addEventListener('click', toggleQuality); $('presentation').addEventListener('click', togglePresentation);
  $('height-scale').addEventListener('input', () => { const value = Number($('height-scale').value); text('height-value', `${number(value)}×`); scene?.setHeight(value); });
  document.querySelectorAll('[data-layer]').forEach(input => input.addEventListener('change', () => { scene?.setLayer(input.dataset.layer, input.checked); updateLayerButton(); }));
  $('layers-toggle').addEventListener('click', () => { const inputs = [...document.querySelectorAll('[data-layer]')].filter(input => !input.disabled), value = !inputs.some(input => input.checked); for (const input of inputs) { input.checked = value; scene?.setLayer(input.dataset.layer, value); } updateLayerButton(); });
  $('export').addEventListener('click', exportReplay);
  $('screenshot').addEventListener('click', () => { try { const shot = scene.capture(); download(shot.url, `terra-step-${frame().step}.png`); notify(`Scene captured · ${shot.width} × ${shot.height} PNG`); } catch (error) { showError(error); } });
  $('open-file').addEventListener('click', () => $('replay-file').click());
  $('replay-file').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    if (busy) { event.target.value = ''; notify('Wait for the current action before opening a replay.'); return; }
    busy = true; setPlaying(false); updateControls();
    try { if (file.size > 256 * 1024 * 1024) throw new Error('Please use a JSON recording smaller than 256 MB. Large recordings can be opened through Python with --replay.'); const value = JSON.parse(await file.text()); loadSession({ mode: 'replay', replay: value }, { local: true }); notify(`Opened ${file.name}`); }
    catch (error) { showError(error); }
    finally { event.target.value = ''; busy = false; updateControls(); }
  });
  $('resume-live').addEventListener('click', async () => { if (busy) return; busy = true; setPlaying(false); updateControls(); try { loadSession(await request('/api/session')); } catch (error) { showError(error); } finally { busy = false; updateControls(); } });
  $('dismiss-error').addEventListener('click', () => { $('error').hidden = true; });
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName) || event.target.isContentEditable || !$('error').hidden) return;
    const key = event.key.toLowerCase();
    if (key === 'g') { event.preventDefault(); toggleQuality(); return; } if (key === 'p') { event.preventDefault(); togglePresentation(); return; }
    if (key === 'h') { event.preventDefault(); scene?.home(); return; } if (key === 't') { event.preventDefault(); scene?.top(); return; } if (key === 'f') { event.preventDefault(); scene?.setFollow(!scene.follow); return; }
    if (!replay || busy) return;
    if (mode !== 'manual' || imported || index < replay.frames.length - 1 || playing) { if (key === ' ') { event.preventDefault(); togglePlayback(); } if (key === 'arrowleft' || key === 'arrowright') { event.preventDefault(); setPlaying(false); showFrame(index + (key === 'arrowright' ? 1 : -1)); } return; }
    if (key === 'r') { event.preventDefault(); resetSession(); return; }
    const agent = frame().agents.find(item => item.id === frame().current_agent), left = agent.action_type === 1 ? 2 : 3, right = agent.action_type === 1 ? 3 : 2;
    const actions = { arrowup: 0, w: 0, arrowdown: 1, s: 1, arrowleft: left, a: left, arrowright: right, d: right, q: 5, e: 4, ' ': 6, n: 7 };
    if (key in actions) { event.preventDefault(); performAction(actions[key]); }
  });
}
function showQuality(quality) { $('quality').setAttribute('aria-pressed', String(quality === 'high')); }
const STYLES = { studio: ['Studio', 'Studio style · earth block on a studio floor'], paper: ['Paper', 'Paper style · plain figure look'], diorama: ['Diorama', 'Diorama style · stylized island'] };
function showPresentation(value) { $('presentation').querySelector('span').textContent = STYLES[value][0]; }
function togglePresentation() { if (!scene) return; const order = Object.keys(STYLES), value = scene.setPresentation(order[(order.indexOf(scene.presentation) + 1) % order.length]); showPresentation(value); updateInspector(); notify(STYLES[value][1]); }
function toggleQuality() { if (!scene) return; const quality = scene.setQuality(scene.quality === 'high' ? 'fast' : 'high'); showQuality(quality); notify(quality === 'high' ? 'Rich lighting on · ambient occlusion and outlines' : 'Fast graphics · plain lighting'); }
function updateLayerButton() { text('layers-toggle', [...document.querySelectorAll('[data-layer]')].some(input => input.checked && !input.disabled) ? 'Hide all' : 'Show all'); }

async function start() {
  bindControls(); updateControls();
  try {
    scene = new TerraScene($('viewport'), {
      onPick: cell => { selected = cell; updateInspector(); },
      onCameraChange: ({ view, follow }) => { if (view) { $('camera-home').classList.toggle('selected', view === 'home'); $('camera-top').classList.toggle('selected', view === 'top'); } if (follow !== undefined) $('camera-follow').setAttribute('aria-pressed', String(follow)); },
      onError: showError,
      onQualityChange: quality => { showQuality(quality); notify('Switched to fast graphics for smoother motion · press G to restore'); },
    });
    showQuality(scene.quality); showPresentation(scene.presentation);
    window.terraViewer = { scene, show: (next, options) => showFrame(next, options) };
    if (offline) loadSession({ mode: 'replay', replay: JSON.parse(embedded.textContent) }, { local: true });
    else loadSession(await request('/api/session'));
    requestAnimationFrame(playbackTick);
  } catch (error) { showError(error); text('session-mode', 'Unavailable'); text('title', 'Open a Terra worksite'); text('source', 'Check the error message to continue.'); }
}
start();
