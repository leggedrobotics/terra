/* Offline video adapter: use the installed viewer, never a separate renderer. */
({
caption(frame, index, count, metric) {
  if (metric) return index ? frame.metric.event.phase : 'Initial state';
  if (frame.diagnostics?.exploring) return `Exploration outside episode budget · ${frame.task_done ? 'task complete' : 'task incomplete'}`;
  return frame.task_done ? 'Task complete' : frame.done ? 'Episode ended without success'
    : index === count - 1 ? 'Recording ended before termination' : 'In progress';
},
title(data, metric) {
  const kind = metric ? (data.metadata.variant === 'original' ? 'Original plan' : 'Postprocessed plan') : 'Native Terra replay';
  const status = data.metadata.plan_status || data.metadata.validation?.verdict || '';
  const schedule = data.metadata.schedule_status || '';
  return [data.metadata.title, kind, status.replaceAll('_', ' '), schedule.replaceAll('_', ' '),
    metric ? '' : data.metadata.source].filter(Boolean).join('\n');
},
prepare(options) {
  const view = window.timelineView || window.terraViewer, scene = view.scene;
  const metric = !!window.timelineView;
  const data = metric ? view.data : JSON.parse(document.getElementById('terra-replay').textContent);
  const count = data.frames.length + Number(metric);
  view.pause?.(); scene.renderer.setAnimationLoop(null);
  window.requestAnimationFrame = window.terraVideoRAF;
  scene.reducedMotion = true; scene.perf = null;
  scene.setQuality(options.quality); scene.setPresentation(options.presentation);
  scene.controls.enableDamping = false;
  // Keep the existing controls in the DOM: seeking updates their status text.
  // Only the shared scene and compact video labels are visible in the export.
  document.body.appendChild(scene.element);
  const style = document.createElement('style');
  style.textContent = `
    body { margin:0!important; overflow:hidden!important; }
    body > * { visibility:hidden!important; }
    #${scene.element.id} { visibility:visible!important; position:fixed!important;
      inset:0!important; width:100vw!important; height:100vh!important; }
    #terra-video-title, #terra-video-caption { visibility:visible!important;
      position:fixed; z-index:1000; left:20px; max-width:calc(100vw - 40px);
      padding:9px 13px; border-radius:6px; background:#faf9f2e8; color:#282721;
      font:14px/1.4 system-ui,sans-serif; white-space:pre-line; }
    #terra-video-title { top:16px; font-weight:600; }
    #terra-video-caption { bottom:16px; font-size:12px; }
  `;
  document.head.appendChild(style);
  const title = document.createElement('div'), caption = document.createElement('div');
  title.id = 'terra-video-title'; caption.id = 'terra-video-caption';
  title.textContent = this.title(data, metric);
  document.body.append(title, caption);
  scene.resize(); view.resize?.();
  // The interactive metric worksite camera fits the initial work area only.
  // A fixed export camera must also cover later machines and reservations.
  if (options.camera === 'top') scene.top();
  else scene.home({ instant: true });
  const cameraPosition = scene.camera.position.clone(), cameraTarget = scene.controls.target.clone();
  const check = () => {
    const error = document.getElementById('error');
    if (error && !error.hidden) throw new Error(error.textContent);
    if (scene.renderer.getContext().isContextLost()) throw new Error('WebGL context lost');
  };
  this.show = index => {
      if (!Number.isInteger(index) || index < 0 || index >= count) throw new Error('Frame outside recording');
      view.show(index, { animate: false });
      // A seek to the reset frame may reset the camera; preserve the chosen view.
      scene.tween = null; scene.camera.position.copy(cameraPosition);
      scene.controls.target.copy(cameraTarget); scene.controls.update();
      if (scene.motion) throw new Error('Endpoint capture must not interpolate motion');
      const frame = scene.frame;
      const outcome = this.caption(frame, index, count, metric);
      const qualification = metric && data.metadata.plan_status ? 'Modeled plan · native replay required' : 'Recorded states';
      caption.textContent = `Frame ${index} / ${count - 1} · ${outcome}\n${qualification} · illustrative playback timing`;
      check(); scene.render(); check();
      return { index, step: frame.step, motion: !!scene.motion };
  };
}
})
