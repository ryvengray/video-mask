console.info('[video-mask] dashboard.js loaded (manual-upload build 2026-08-19a, no token prompt)');
const tabs = document.querySelectorAll('.tab');
function parseAlgorithmArguments(value) {
  const raw = value.trim();
  if (!raw) return [];
  if (raw.startsWith('[')) {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || !parsed.every(argument => typeof argument === 'string')) {
      throw new Error('not a string array');
    }
    return parsed;
  }

  const argumentsValue = [];
  let argument = '';
  let quote = '';
  let escaping = false;
  let started = false;
  for (const character of raw) {
    if (escaping) {
      argument += character;
      escaping = false;
      started = true;
    } else if (character === '\\') {
      escaping = true;
      started = true;
    } else if (quote) {
      if (character === quote) quote = '';
      else argument += character;
    } else if (character === '"' || character === "'") {
      quote = character;
      started = true;
    } else if (/\s/.test(character)) {
      if (started) {
        argumentsValue.push(argument);
        argument = '';
        started = false;
      }
    } else {
      argument += character;
      started = true;
    }
  }
  if (escaping || quote) throw new Error('unterminated command-line argument');
  if (started) argumentsValue.push(argument);
  return argumentsValue;
}

function configureAlgorithmArgumentsInput(input) {
  if (!input) return;
  input.placeholder = '--fisheye --fisheye-device pico4 --face-size 640';
  const label = input.closest('label');
  const text = [...(label?.childNodes || [])].find(node => node.nodeType === Node.TEXT_NODE);
  if (text) text.textContent = 'Algorithm parameters (JSON array or command line)';
}

const panels = {
  tasks: document.getElementById('panel-tasks'),
  workers: document.getElementById('panel-workers'),
  statistics: document.getElementById('panel-statistics'),
  settings: document.getElementById('panel-settings'),
  cost: document.getElementById('panel-cost'),
};
const costFrame = document.getElementById('cost-monitor-frame');
function activate(name) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === name));
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle('active', key === name));
  if (name === 'cost' && costFrame && !costFrame.getAttribute('src')) costFrame.src = costFrame.dataset.src;
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
}
tabs.forEach(tab => tab.addEventListener('click', () => activate(tab.dataset.tab)));
activate(['tasks', 'workers', 'statistics', 'settings', 'cost'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'tasks');
const workerFilter = document.getElementById('worker-status-filter');
workerFilter?.addEventListener('change', () => {
  const selected = workerFilter.value;
  document.querySelectorAll('.server-card').forEach(card => {
    let visible = false;
    card.querySelectorAll('tbody tr[data-worker-status]').forEach(row => {
      const hidden = Boolean(selected) && row.dataset.workerStatus !== selected;
      row.hidden = hidden;
      if (!hidden) visible = true;
    });
    card.style.display = visible ? '' : 'none';
  });
});

const s3IngestControl = document.getElementById('s3-ingest-control');
if (s3IngestControl) {
  const configured = s3IngestControl.dataset.configured === 'true';
  let enabled = s3IngestControl.dataset.enabled === 'true';
  const toggle = document.getElementById('s3-ingest-toggle');
  const status = document.getElementById('s3-ingest-status');
  const message = document.getElementById('s3-ingest-message');

  function renderS3IngestState() {
    const label = enabled ? 'S3 ingestion enabled' : 'S3 ingestion paused';
    status.className = `s3-ingest-status-light ${enabled ? 'enabled' : 'paused'}`;
    status.setAttribute('aria-label', label);
    status.title = label;
    status.textContent = '';
    toggle.textContent = enabled ? 'Pause S3 ingestion' : 'Resume S3 ingestion';
  }

  if (configured) {
    toggle.addEventListener('click', async () => {
      const nextEnabled = !enabled;
      const action = nextEnabled ? 'resume' : 'pause';
      if (!window.confirm(`Do you want to ${action} automatic S3 task ingestion?`)) return;
      toggle.disabled = true;
      message.textContent = '';
      try {
        const response = await fetch('/api/admin/s3-ingest', {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({enabled: nextEnabled}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
        enabled = Boolean(payload.enabled);
        renderS3IngestState();
        message.textContent = `S3 task ingestion is now ${enabled ? 'enabled' : 'paused'}.`;
      } catch (error) {
        message.textContent = error instanceof Error ? error.message : 'Unable to update S3 task ingestion.';
      } finally {
        toggle.disabled = false;
      }
    });
  }
}

const manualTaskControl = document.getElementById('manual-task-control');
if (manualTaskControl?.dataset.s3Configured === 'true') {
  const manualTaskForm = document.getElementById('manual-task-form');
  const manualTaskOpen = document.getElementById('manual-task-open');
  const manualTaskFile = document.getElementById('manual-task-file');
  const manualTaskAlgorithm = document.getElementById('manual-task-algorithm');
  const manualTaskArguments = document.getElementById('manual-task-arguments');
  const manualTaskSubmit = document.getElementById('manual-task-submit');
  const manualTaskMessage = document.getElementById('manual-task-message');
  configureAlgorithmArgumentsInput(manualTaskArguments);
  const defaultAlgorithm = manualTaskAlgorithm.value;
  const defaultArguments = manualTaskArguments.value;

  function setManualTaskMessage(text, error = false) {
    manualTaskMessage.textContent = text;
    manualTaskMessage.classList.toggle('error', error);
  }

  function setManualTaskBusy(busy) {
    manualTaskForm.querySelectorAll('input, textarea, button').forEach(control => {
      control.disabled = busy;
    });
  }

  manualTaskOpen.addEventListener('click', () => {
    manualTaskControl.hidden = false;
    manualTaskFile.focus();
  });

  manualTaskForm.addEventListener('submit', event => {
    event.preventDefault();
    const source = manualTaskFile.files?.[0];
    if (!source) {
      setManualTaskMessage('Choose a video file first.', true);
      return;
    }
    let parsed;
    try {
      parsed = parseAlgorithmArguments(manualTaskArguments.value);
    } catch (_) {
      setManualTaskMessage('Algorithm parameters must be a JSON array or command-line string.', true);
      return;
    }
    const request = new XMLHttpRequest();
    setManualTaskBusy(true);
    pauseAutoRefresh();
    setManualTaskMessage(`Uploading ${source.name}… 0%`);
    request.open('PUT', '/api/dashboard/manual-tasks');
    request.setRequestHeader('Content-Type', source.type || 'application/octet-stream');
    request.setRequestHeader('X-Video-Mask-Filename', source.name);
    request.setRequestHeader('X-Video-Mask-Algorithm', manualTaskAlgorithm.value.trim());
    request.setRequestHeader('X-Video-Mask-Arguments', JSON.stringify(parsed));
    request.upload.addEventListener('progress', progress => {
      if (!progress.lengthComputable) return;
      const percent = Math.min(100, Math.round(progress.loaded / progress.total * 100));
      setManualTaskMessage(`Uploading ${source.name}… ${percent}%`);
    });
    request.addEventListener('load', () => {
      let response = {};
      try { response = JSON.parse(request.responseText || '{}'); } catch (_) { /* use fallback below */ }
      if (request.status >= 200 && request.status < 300) {
        const taskId = response.task?.task_id || 'new task';
        setManualTaskMessage(`Upload complete. ${taskId} is queued for processing.`);
        manualTaskForm.reset();
        manualTaskAlgorithm.value = defaultAlgorithm;
        manualTaskArguments.value = defaultArguments;
        window.setTimeout(() => window.location.reload(), 700);
        return;
      }
      setManualTaskBusy(false);
      scheduleAutoRefresh();
      setManualTaskMessage(response.detail || `Upload failed (${request.status}).`, true);
    });
    request.addEventListener('error', () => {
      setManualTaskBusy(false);
      scheduleAutoRefresh();
      setManualTaskMessage('The upload connection failed. The task was not queued.', true);
    });
    request.send(source);
  });
}

const playModal = document.getElementById('play-link-modal');
const playVideo = document.getElementById('play-modal-video');
const playKindBadge = document.getElementById('play-modal-kind');
const playFileName = document.getElementById('play-modal-name');
const playOpenLink = document.getElementById('play-modal-open');
const playDownloadLink = document.getElementById('play-modal-download');
const taskLogModal = document.getElementById('task-log-modal');
const taskLogNote = document.getElementById('task-log-modal-note');
const taskLogContent = document.getElementById('task-log-modal-content');
const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
const autoRefreshNote = document.getElementById('auto-refresh-note');
let autoRefreshTimer;
let autoRefreshEnabled = window.localStorage.getItem('video-mask-auto-refresh') !== 'false';
let activeFaceReview = null;
let beginModalAnnotation = () => {};
let endModalAnnotation = () => {};

function scheduleAutoRefresh() {
  window.clearTimeout(autoRefreshTimer);
  if (!autoRefreshEnabled || activeFaceReview) return;
  autoRefreshTimer = window.setTimeout(() => window.location.reload(), 60_000);
}

function pauseAutoRefresh() {
  window.clearTimeout(autoRefreshTimer);
}

function renderAutoRefreshState() {
  if (autoRefreshToggle) autoRefreshToggle.checked = autoRefreshEnabled;
  if (autoRefreshNote) {
    autoRefreshNote.textContent = `${autoRefreshEnabled ? 'Auto-refreshes every 60 seconds' : 'Auto-refresh paused'} · ${autoRefreshNote.dataset.updatedAt}`;
  }
}

if (autoRefreshNote) autoRefreshNote.dataset.updatedAt = autoRefreshNote.textContent.split(' · ').at(-1) || '';
renderAutoRefreshState();
autoRefreshToggle?.addEventListener('change', () => {
  autoRefreshEnabled = autoRefreshToggle.checked;
  window.localStorage.setItem('video-mask-auto-refresh', String(autoRefreshEnabled));
  renderAutoRefreshState();
  if (autoRefreshEnabled) scheduleAutoRefresh();
  else pauseAutoRefresh();
});

function openPlayModal(url, kind, name, taskId) {
  playKindBadge.textContent = kind === 'output' ? 'Output' : 'Input';
  playKindBadge.className = `badge ${kind === 'output' ? 'success' : 'active'}`;
  playFileName.textContent = name;
  playFileName.title = name;
  playVideo.src = url;
  playOpenLink.href = url;
  playDownloadLink.href = url;
  playDownloadLink.download = name;
  playModal.hidden = false;
  pauseAutoRefresh();
  beginModalAnnotation(taskId);
  playVideo.focus();
}

function closePlayModal() {
  if (!playModal || playModal.hidden) return;
  playVideo.pause();
  playVideo.removeAttribute('src');
  playVideo.load();
  endModalAnnotation();
  playModal.hidden = true;
  scheduleAutoRefresh();
}

async function openTaskLogModal(taskId) {
  taskLogModal.hidden = false;
  taskLogNote.textContent = 'Loading logs…';
  taskLogContent.textContent = '';
  pauseAutoRefresh();
  try {
    const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskId)}/logs?limit=1000`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    const logs = payload.logs || [];
    taskLogNote.textContent = logs.length
      ? `Showing the latest ${logs.length} Worker algorithm log lines.`
      : 'No Worker algorithm logs are available for this task.';
    taskLogContent.textContent = logs.map(log => {
      const timestamp = new Date(Number(log.created_at || 0) * 1000).toLocaleString();
      return `[${timestamp}] ${log.line}`;
    }).join('\n');
  } catch (error) {
    taskLogNote.textContent = error instanceof Error ? error.message : 'Unable to load task logs.';
  }
}

function closeTaskLogModal() {
  if (!taskLogModal || taskLogModal.hidden) return;
  taskLogModal.hidden = true;
  scheduleAutoRefresh();
}

playModal?.querySelectorAll('[data-play-close]').forEach(element =>
  element.addEventListener('click', closePlayModal));
taskLogModal?.querySelectorAll('[data-task-log-close]').forEach(element =>
  element.addEventListener('click', closeTaskLogModal));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    closePlayModal();
    closeTaskLogModal();
    closeTaskRestartModal();
  }
});
scheduleAutoRefresh();
const taskTable = document.querySelector('.task-table');

function copyTaskId(value, button) {
  const fallback = () => {
    const input = document.createElement('textarea');
    input.value = value;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.append(input);
    input.select();
    document.execCommand('copy');
    input.remove();
  };
  const copied = navigator.clipboard?.writeText
    ? navigator.clipboard.writeText(value).catch(fallback)
    : Promise.resolve().then(fallback);
  copied.then(() => {
    const original = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = original; }, 1200);
  }).catch(() => window.alert('Unable to copy the task ID.'));
}

function taskFilename(path) {
  return path.split('/').filter(Boolean).at(-1) || 'video.mp4';
}

async function downloadTaskFile(button) {
  const {taskId, file, filePath} = button.dataset;
  if (!taskId || !file) return;
  button.disabled = true;
  try {
    const response = await fetch(
      `/api/tasks/${encodeURIComponent(taskId)}/play-url?file=${encodeURIComponent(file)}&download=true`,
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    const link = document.createElement('a');
    link.href = payload.url;
    link.download = taskFilename(filePath || '');
    document.body.append(link);
    link.click();
    link.remove();
  } catch (error) {
    window.alert(error instanceof Error ? error.message : 'Unable to download this file.');
  } finally {
    button.disabled = false;
  }
}

function enhanceTaskIdentifiersAndFiles() {
  if (!taskTable) return;
  taskTable.querySelectorAll('td.task-id').forEach(cell => {
    if (cell.dataset.enhanced === 'true') return;
    const taskId = cell.textContent.trim();
    if (!taskId) return;
    cell.dataset.enhanced = 'true';
    cell.title = taskId;
    cell.replaceChildren();
    const display = document.createElement('span');
    display.className = 'task-id-display';
    display.textContent = taskId.length > 12 ? `${taskId.slice(0, 12)}…` : taskId;
    display.title = taskId;
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'task-id-copy';
    copy.title = 'Copy full task ID';
    copy.setAttribute('aria-label', 'Copy full task ID');
    copy.textContent = '⧉';
    copy.addEventListener('click', () => copyTaskId(taskId, copy));
    cell.append(display, copy);
  });
  taskTable.querySelectorAll('.file-play[data-task-id]').forEach(fileButton => {
    if (fileButton.dataset.enhanced === 'true') return;
    const filePath = fileButton.textContent.trim();
    if (!filePath || filePath === '-') return;
    fileButton.dataset.enhanced = 'true';
    fileButton.title = filePath;
    const download = document.createElement('button');
    download.type = 'button';
    download.className = 'task-file-download';
    download.dataset.taskDownload = 'true';
    download.dataset.taskId = fileButton.dataset.taskId;
    download.dataset.file = fileButton.dataset.file;
    download.dataset.filePath = filePath;
    download.title = `Download ${filePath}`;
    download.setAttribute('aria-label', `Download ${filePath}`);
    download.textContent = '⤓';
    download.addEventListener('click', () => downloadTaskFile(download));
    fileButton.after(download);
  });
}

enhanceTaskIdentifiersAndFiles();

const faceReviewControl = document.getElementById('face-review-control');
if (faceReviewControl) {
  const reviewerStorageKey = 'video-mask-face-reviewer-id';
  let reviewerId = sessionStorage.getItem(reviewerStorageKey);
  if (!reviewerId) {
    reviewerId = window.crypto?.randomUUID?.() ||
      `face-review-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem(reviewerStorageKey, reviewerId);
  }

  const claimButton = document.getElementById('face-review-claim');
  const reviewDetails = faceReviewControl;
  const activeBadge = document.getElementById('face-review-active');
  const message = document.getElementById('face-review-message');
  const player = document.getElementById('face-review-player');
  const video = document.getElementById('face-review-video');
  const reviewTabs = document.querySelectorAll('[data-face-review-tab]');
  const framesPanel = document.getElementById('face-review-frames-panel');
  const videoPanel = document.getElementById('face-review-video-panel');
  const framesMessage = document.getElementById('face-review-frames-message');
  const frames = document.getElementById('face-review-frames');
  const faceButton = document.getElementById('face-review-face');
  const noFaceButton = document.getElementById('face-review-no-face');
  const releaseButton = document.getElementById('face-review-release');
  const modalAnnotation = document.getElementById('play-modal-annotation');
  const modalAnnotationStatus = document.getElementById('play-modal-annotation-status');
  const modalHasFaceButton = document.getElementById('play-modal-has-face');
  const modalNoFaceButton = document.getElementById('play-modal-no-face');
  let heartbeatTimer;
  let framePreviewTimer;
  let modalHeartbeatTimer;
  let activeModalReview = null;
  let modalReviewToken = 0;

  async function reviewRequest(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {'Content-Type': 'application/json', ...(options.headers || {})},
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    return payload;
  }

  function setReviewMessage(value) {
    message.textContent = value;
  }

  function setReviewControls(busy = false) {
    claimButton.disabled = busy || Boolean(activeFaceReview);
    faceButton.disabled = busy;
    noFaceButton.disabled = busy;
    releaseButton.disabled = busy;
  }

  function modalAnnotationText(faceAnnotation) {
    if (faceAnnotation === 1) return 'Current label: 👍 Has face';
    if (faceAnnotation === 0) return 'Current label: 👎 No face';
    return 'Current label: unlabelled';
  }

  function setModalAnnotationControls(enabled) {
    modalHasFaceButton.disabled = !enabled;
    modalNoFaceButton.disabled = !enabled;
  }

  async function heartbeatModalReview() {
    if (!activeModalReview) return;
    try {
      await reviewRequest(`/api/face-reviews/${encodeURIComponent(activeModalReview.task_id)}/heartbeat`, {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
    } catch (error) {
      window.clearInterval(modalHeartbeatTimer);
      activeModalReview = null;
      setModalAnnotationControls(false);
      modalAnnotationStatus.textContent = error instanceof Error ? error.message : 'Manual label lease ended.';
    }
  }

  async function saveModalAnnotation(hasFace) {
    if (!activeModalReview) return;
    setModalAnnotationControls(false);
    try {
      const task = await reviewRequest(
        `/api/face-reviews/${encodeURIComponent(activeModalReview.task_id)}/annotation`,
        {method: 'PUT', body: JSON.stringify({reviewer_id: reviewerId, has_face: hasFace})},
      );
      const reopened = await reviewRequest(
        `/api/face-reviews/${encodeURIComponent(activeModalReview.task_id)}/open`,
        {method: 'POST', body: JSON.stringify({reviewer_id: reviewerId})},
      );
      modalAnnotationStatus.textContent = modalAnnotationText(task.face_annotation);
      activeModalReview = {task_id: reopened.task.task_id};
      refreshFaceReviewStatuses();
    } catch (error) {
      modalAnnotationStatus.textContent = error instanceof Error ? error.message : 'Unable to save manual label.';
    } finally {
      setModalAnnotationControls(Boolean(activeModalReview));
    }
  }

  beginModalAnnotation = async taskId => {
    const token = ++modalReviewToken;
    window.clearInterval(modalHeartbeatTimer);
    activeModalReview = null;
    modalAnnotation.hidden = false;
    modalAnnotationStatus.textContent = 'Opening manual label…';
    setModalAnnotationControls(false);
    try {
      const payload = await reviewRequest(`/api/face-reviews/${encodeURIComponent(taskId)}/open`, {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
      if (token !== modalReviewToken || playModal.hidden) {
        fetch(`/api/face-reviews/${encodeURIComponent(taskId)}/release`, {
          method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({reviewer_id: reviewerId}),
        });
        return;
      }
      activeModalReview = {task_id: taskId};
      modalAnnotationStatus.textContent = modalAnnotationText(payload.task.face_annotation);
      setModalAnnotationControls(true);
      modalHeartbeatTimer = window.setInterval(heartbeatModalReview, 30_000);
    } catch (error) {
      modalAnnotationStatus.textContent = error instanceof Error
        ? `${error.message}.`
        : 'Manual labels are available for completed videos only.';
    }
  };

  endModalAnnotation = () => {
    modalReviewToken += 1;
    window.clearInterval(modalHeartbeatTimer);
    modalHeartbeatTimer = undefined;
    const review = activeModalReview;
    activeModalReview = null;
    modalAnnotation.hidden = true;
    setModalAnnotationControls(false);
    if (!review) return;
    fetch(`/api/face-reviews/${encodeURIComponent(review.task_id)}/release`, {
      method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer_id: reviewerId}),
    });
  };

  function stopActiveReview({keepMessage = false} = {}) {
    window.clearInterval(heartbeatTimer);
    window.clearTimeout(framePreviewTimer);
    heartbeatTimer = undefined;
    framePreviewTimer = undefined;
    video.pause();
    video.removeAttribute('src');
    video.load();
    frames.replaceChildren();
    framesMessage.textContent = 'Preparing low-resolution frame previews…';
    player.hidden = true;
    reviewDetails.hidden = true;
    activeFaceReview = null;
    activeBadge.className = 'badge muted';
    activeBadge.textContent = 'No active review';
    setReviewControls();
    if (!keepMessage) setReviewMessage('Video released. You can claim another unlabelled completed video.');
    scheduleAutoRefresh();
  }

  function renderFramePreviews(previews) {
    frames.replaceChildren();
    previews.forEach(preview => {
      const item = document.createElement('figure');
      const image = document.createElement('img');
      image.src = preview.url;
      image.loading = 'lazy';
      image.alt = `Frame at ${Number(preview.timestamp_seconds || 0).toFixed(0)} seconds`;
      const caption = document.createElement('figcaption');
      const seconds = Number(preview.timestamp_seconds || 0);
      caption.textContent = `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
      item.append(image, caption);
      frames.append(item);
    });
  }

  async function loadFramePreviews() {
    if (!activeFaceReview) return;
    const taskId = activeFaceReview.task_id;
    try {
      const payload = await reviewRequest(
        `/api/tasks/${encodeURIComponent(taskId)}/frame-previews`
      );
      if (!activeFaceReview || activeFaceReview.task_id !== taskId) return;
      renderFramePreviews(payload.frames || []);
      if (payload.state === 'ready') {
        framesMessage.textContent = `${(payload.frames || []).length} low-resolution frames ready. Open Video only when you need more detail.`;
      } else if (payload.state === 'error') {
        framesMessage.textContent = payload.error || 'Unable to generate frame previews. You can still use the Video tab.';
      } else {
        framesMessage.textContent = payload.frames?.length
          ? `Preparing frames… ${payload.frames.length} available so far.`
          : 'Preparing low-resolution frame previews from S3…';
        framePreviewTimer = window.setTimeout(loadFramePreviews, 2_000);
      }
    } catch (error) {
      framesMessage.textContent = error instanceof Error
        ? `${error.message}. You can still use the Video tab.`
        : 'Unable to generate frame previews. You can still use the Video tab.';
    }
  }

  function showFaceReviewTab(name) {
    const showFrames = name === 'frames';
    reviewTabs.forEach(tab => {
      const active = tab.dataset.faceReviewTab === name;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    });
    framesPanel.hidden = !showFrames;
    videoPanel.hidden = showFrames;
    if (showFrames) {
      window.clearTimeout(framePreviewTimer);
      loadFramePreviews();
    } else if (activeFaceReview && !video.src) {
      video.src = activeFaceReview.playback_url;
    }
  }

  async function heartbeatFaceReview() {
    if (!activeFaceReview) return;
    try {
      await reviewRequest(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/heartbeat`, {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
    } catch (error) {
      stopActiveReview({keepMessage: true});
      setReviewMessage(error instanceof Error ? `${error.message}. This video is available for another reviewer.` : 'Review lease ended.');
      refreshFaceReviewStatuses();
    }
  }

  async function claimFaceReview() {
    setReviewControls(true);
    setReviewMessage('Finding an unlabelled completed video…');
    try {
      const payload = await reviewRequest('/api/face-reviews/claim', {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
      if (!payload.task) {
        window.alert('No unlabelled completed videos are available right now.');
        setReviewControls();
        return;
      }
      activeFaceReview = {...payload.task, playback_url: payload.playback_url};
      reviewDetails.hidden = false;
      player.hidden = false;
      activeBadge.className = 'badge active';
      activeBadge.textContent = 'Review in progress';
      const filename = payload.playback_file === 'output'
        ? (payload.task.output_object_key || 'output video')
        : (payload.task.source_object_key || 'input video');
      setReviewMessage(`Reserved for this browser. Playing ${payload.playback_file} video: ${filename}`);
      setReviewControls();
      pauseAutoRefresh();
      heartbeatTimer = window.setInterval(heartbeatFaceReview, 30_000);
      showFaceReviewTab('video');
      refreshFaceReviewStatuses();
    } catch (error) {
      window.alert(error instanceof Error ? error.message : 'Unable to claim a video.');
      setReviewControls();
    }
  }

  async function annotateFaceReview(hasFace) {
    if (!activeFaceReview) return;
    setReviewControls(true);
    try {
      await reviewRequest(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/annotation`, {
        method: 'PUT', body: JSON.stringify({reviewer_id: reviewerId, has_face: hasFace}),
      });
      stopActiveReview({keepMessage: true});
      setReviewMessage(hasFace ? 'Saved: face detected.' : 'Saved: no face detected.');
      refreshFaceReviewStatuses();
    } catch (error) {
      setReviewMessage(error instanceof Error ? error.message : 'Unable to save the label.');
      setReviewControls();
    }
  }

  async function releaseFaceReview() {
    if (!activeFaceReview) return;
    setReviewControls(true);
    try {
      await reviewRequest(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/release`, {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
      stopActiveReview();
      refreshFaceReviewStatuses();
    } catch (error) {
      setReviewMessage(error instanceof Error ? error.message : 'Unable to release this video.');
      setReviewControls();
    }
  }

  function ensureFaceAnnotationColumn() {
    if (!taskTable) return [];
    const table = taskTable.querySelector('table');
    const header = table?.querySelector('thead tr');
    if (!table || !header) return [];
    if (!header.querySelector('.face-annotation-heading')) {
      const cell = document.createElement('th');
      cell.className = 'face-annotation-heading';
      cell.textContent = 'Manual label';
      header.insertBefore(cell, header.children[2] || null);
    }
    const rows = [...table.querySelectorAll('tbody tr')];
    rows.forEach(row => {
      const playButton = row.querySelector('.file-play[data-task-id]');
      if (!playButton) {
        row.querySelector('td')?.setAttribute('colspan', String(header.children.length));
        return;
      }
      if (!row.querySelector('.face-annotation-cell')) {
        const cell = document.createElement('td');
        cell.className = 'face-annotation-cell';
        cell.dataset.taskId = playButton.dataset.taskId;
        cell.textContent = 'Loading…';
        row.insertBefore(cell, row.children[2] || null);
      }
    });
    return [...table.querySelectorAll('.face-annotation-cell')];
  }

  function renderFaceReviewStatuses(reviews) {
    const byTaskId = new Map(reviews.map(review => [review.task_id, review]));
    ensureFaceAnnotationColumn().forEach(cell => {
      const review = byTaskId.get(cell.dataset.taskId);
      cell.className = 'face-annotation-cell';
      if (!review?.reviewable) {
        cell.textContent = '–';
      } else if (review.has_face === true) {
        cell.textContent = '👍 Face';
        cell.classList.add('labelled');
      } else if (review.has_face === false) {
        cell.textContent = '👎 No face';
        cell.classList.add('labelled');
      } else if (review.reviewing) {
        cell.textContent = '👀 Reviewing';
        cell.classList.add('reviewing');
      } else {
        cell.textContent = 'Unlabelled';
      }
    });
  }

  async function refreshFaceReviewStatuses() {
    const cells = ensureFaceAnnotationColumn();
    const taskIds = [...new Set(cells.map(cell => cell.dataset.taskId).filter(Boolean))];
    if (!taskIds.length) return;
    try {
      const payload = await reviewRequest(`/api/face-reviews/status?task_ids=${encodeURIComponent(taskIds.join(','))}`);
      renderFaceReviewStatuses(payload.reviews || []);
    } catch (error) {
      console.warn('[video-mask] unable to refresh face-review status', error);
    }
  }

  claimButton.addEventListener('click', claimFaceReview);
  reviewTabs.forEach(tab => tab.addEventListener('click', () => showFaceReviewTab(tab.dataset.faceReviewTab)));
  faceButton.addEventListener('click', () => annotateFaceReview(true));
  noFaceButton.addEventListener('click', () => annotateFaceReview(false));
  releaseButton.addEventListener('click', releaseFaceReview);
  modalHasFaceButton.addEventListener('click', () => saveModalAnnotation(true));
  modalNoFaceButton.addEventListener('click', () => saveModalAnnotation(false));
  window.addEventListener('pagehide', () => {
    if (!activeFaceReview) return;
    fetch(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/release`, {
      method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer_id: reviewerId}),
    });
  });
  refreshFaceReviewStatuses();
  window.setInterval(refreshFaceReviewStatuses, 10_000);
}

function ensureTaskActionsColumn() {
  if (!taskTable) return;
  const table = taskTable.querySelector('table');
  const header = table?.querySelector('thead tr');
  if (!table || !header) return;
  if (!header.querySelector('.task-actions-heading')) {
    const cell = document.createElement('th');
    cell.className = 'task-actions-heading';
    cell.textContent = 'Actions';
    header.append(cell);
  }
  table.querySelectorAll('tbody tr').forEach(row => {
    if (row.querySelector('.task-actions-cell')) return;
    const playButton = row.querySelector('.file-play[data-task-id]');
    if (!playButton) {
      row.querySelector('td')?.setAttribute('colspan', String(header.children.length));
      return;
    }
    const status = row.querySelector('.badge')?.textContent?.trim();
    const cell = document.createElement('td');
    cell.className = 'task-actions-cell';
    if (['completed', 'failed'].includes(status) || ['assigned', 'downloading', 'processing', 'uploading'].includes(status)) {
      const action = ['completed', 'failed'].includes(status) ? 'restart' : 'cancel';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `task-action-button task-action-${action}`;
      button.dataset.taskAction = action;
      button.dataset.taskId = playButton.dataset.taskId;
      button.textContent = action === 'restart' ? 'Restart' : 'Cancel';
      cell.append(button);
    }
    const logsButton = document.createElement('button');
    logsButton.type = 'button';
    logsButton.className = 'task-action-button task-action-logs';
    logsButton.dataset.taskAction = 'logs';
    logsButton.dataset.taskId = playButton.dataset.taskId;
    logsButton.textContent = 'View logs';
    cell.append(logsButton);
    row.append(cell);
  });
}

ensureTaskActionsColumn();
let taskRestartModal;
let taskRestartForm;
let taskRestartId;
let taskRestartAlgorithm;
let taskRestartArguments;
let taskRestartMessage;

function ensureTaskRestartModal() {
  if (taskRestartModal) return;
  document.body.insertAdjacentHTML('beforeend', `
    <div id="task-restart-modal" class="play-modal" hidden>
      <div class="play-modal-backdrop" data-task-restart-close></div>
      <div class="play-modal-card" role="dialog" aria-modal="true" aria-labelledby="task-restart-modal-title">
        <div class="play-modal-head"><h3 id="task-restart-modal-title">Restart task</h3><button class="play-modal-close" type="button" data-task-restart-close aria-label="Close">&times;</button></div>
        <p class="play-modal-note">The existing output will be cleared; face annotation will be kept. Edit the algorithm settings for this run if needed.</p>
        <form id="task-restart-form" style="display:grid;gap:12px">
          <input id="task-restart-id" type="hidden">
          <label style="display:grid;gap:5px;color:var(--muted);font-size:12px">Algorithm file name<input id="task-restart-algorithm" type="text" required style="width:100%;border:1px solid var(--line);border-radius:4px;padding:8px;font:13px Consolas,monospace"></label>
          <label style="display:grid;gap:5px;color:var(--muted);font-size:12px">Algorithm parameters (JSON array)<textarea id="task-restart-arguments" required style="width:100%;min-height:92px;resize:vertical;border:1px solid var(--line);border-radius:4px;padding:8px;font:13px Consolas,monospace"></textarea></label>
          <p id="task-restart-message" style="min-height:18px;margin:0;color:var(--red);font-size:12px" aria-live="polite"></p>
          <div class="play-modal-actions"><button class="play-btn play-btn-open" type="submit">Restart task</button><button class="play-btn" type="button" data-task-restart-close>Cancel</button></div>
        </form>
      </div>
    </div>`);
  taskRestartModal = document.getElementById('task-restart-modal');
  taskRestartForm = document.getElementById('task-restart-form');
  taskRestartId = document.getElementById('task-restart-id');
  taskRestartAlgorithm = document.getElementById('task-restart-algorithm');
  taskRestartArguments = document.getElementById('task-restart-arguments');
  taskRestartMessage = document.getElementById('task-restart-message');
  taskRestartModal.querySelectorAll('[data-task-restart-close]').forEach(element =>
    element.addEventListener('click', closeTaskRestartModal));
  taskRestartForm.addEventListener('submit', submitTaskRestart);
}

function closeTaskRestartModal() {
  if (!taskRestartModal || taskRestartModal.hidden) return;
  taskRestartModal.hidden = true;
  scheduleAutoRefresh();
}

async function openTaskRestartModal(taskId) {
  ensureTaskRestartModal();
  taskRestartId.value = taskId;
  taskRestartMessage.textContent = 'Loading current algorithm settings…';
  taskRestartForm.querySelectorAll('input, textarea, button[type="submit"]').forEach(control => {
    control.disabled = true;
  });
  taskRestartModal.hidden = false;
  pauseAutoRefresh();
  try {
    const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskId)}/restart-config`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    taskRestartAlgorithm.value = payload.algorithm || '';
    taskRestartArguments.value = JSON.stringify(payload.arguments || [], null, 2);
    taskRestartMessage.textContent = '';
    taskRestartAlgorithm.focus();
  } catch (error) {
    taskRestartMessage.textContent = error instanceof Error ? error.message : 'Unable to load this task.';
  } finally {
    taskRestartForm.querySelectorAll('input, textarea, button[type="submit"]').forEach(control => {
      control.disabled = false;
    });
  }
}

async function submitTaskRestart(event) {
  event.preventDefault();
  let argumentsValue;
  try {
    argumentsValue = JSON.parse(taskRestartArguments.value);
    if (!Array.isArray(argumentsValue) || !argumentsValue.every(value => typeof value === 'string')) {
      throw new Error('not a string array');
    }
  } catch (_) {
    taskRestartMessage.textContent = 'Algorithm parameters must be a JSON array of strings.';
    return;
  }
  const algorithm = taskRestartAlgorithm.value.trim();
  if (!algorithm) {
    taskRestartMessage.textContent = 'Algorithm file name is required.';
    return;
  }
  const controls = taskRestartForm.querySelectorAll('input, textarea, button');
  controls.forEach(control => { control.disabled = true; });
  taskRestartMessage.textContent = 'Restarting…';
  try {
    const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskRestartId.value)}/restart`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({algorithm, arguments: argumentsValue}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    window.location.reload();
  } catch (error) {
    taskRestartMessage.textContent = error instanceof Error ? error.message : 'Unable to restart this task.';
    controls.forEach(control => { control.disabled = false; });
  }
}
taskTable?.addEventListener('click', async event => {
  const button = event.target.closest('[data-task-action]');
  if (!button) return;
  const {taskAction, taskId} = button.dataset;
  if (taskAction === 'logs') {
    openTaskLogModal(taskId);
    return;
  }
  if (taskAction === 'restart') {
    button.disabled = true;
    try {
      await openTaskRestartModal(taskId);
    } finally {
      button.disabled = false;
    }
    return;
  }
  const message = 'Cancel this active task? The Worker will stop it as soon as possible.';
  if (!window.confirm(message)) return;
  button.disabled = true;
  try {
    const response = await fetch(
      `/api/dashboard/tasks/${encodeURIComponent(taskId)}/${encodeURIComponent(taskAction)}`,
      {method: 'POST'},
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    window.location.reload();
  } catch (error) {
    window.alert(error instanceof Error ? error.message : 'Unable to update this task.');
    button.disabled = false;
  }
});

console.info('[video-mask] task-table found:', Boolean(taskTable),
  '| play buttons:', document.querySelectorAll('.file-play').length,
  '| modal:', Boolean(document.getElementById('play-link-modal')));
taskTable?.addEventListener('click', async (event) => {
  console.info('[video-mask] click on task table:', event.target);
  const button = event.target.closest('.file-play');
  if (!button) {
    console.info('[video-mask] click target is not a .file-play button - ignored');
    return;
  }
  const { taskId, file } = button.dataset;
  console.info('[video-mask] play-link requested:', { taskId, file });
  if (!taskId) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/play-url?file=${encodeURIComponent(file)}`);
    const payload = await response.json();
    console.info('[video-mask] play-url response:', response.status, payload);
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    openPlayModal(payload.url, file, button.textContent.trim(), taskId);
  } catch (error) {
    console.error('[video-mask] play-link failed:', error);
    window.alert(error instanceof Error ? error.message : 'Unable to get the S3 link.');
  } finally {
    button.disabled = false;
  }
});

const statisticsPayload = document.getElementById('processing-statistics');
const statisticsFilter = document.getElementById('statistics-filter');

function hours(value) {
  return `${(Number(value || 0) / 3600).toFixed(1)} h`;
}

function ratio(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}×` : '–';
}

function workerHoursPerVideoHour(value) {
  if (!Number.isFinite(value)) return '–';
  const minutes = value * 60;
  return minutes < 60 ? `${minutes.toFixed(0)} min` : `${value.toFixed(2)} h`;
}

function svgEscaped(value) {
  return String(value).replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
}

function chartFrame(svg, values, title, draw) {
  const width = 720;
  const height = 240;
  const left = 46;
  const right = 14;
  const top = 18;
  const bottom = 36;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maximum = Math.max(1, ...values);
  const y = value => top + plotHeight - (value / maximum) * plotHeight;
  const grid = Array.from({length: 5}, (_, index) => {
    const value = maximum * index / 4;
    const point = y(value);
    return `<line x1="${left}" y1="${point}" x2="${width - right}" y2="${point}" class="chart-grid"/><text x="${left - 7}" y="${point + 4}" text-anchor="end" class="chart-axis">${value.toFixed(value < 2 ? 1 : 0)}</text>`;
  }).join('');
  const pointCount = draw.hours?.length || values.length;
  const labels = pointCount ? [0, Math.floor((pointCount - 1) / 2), pointCount - 1].map(index => {
    const date = new Date((draw.hours?.[index]?.start_at || 0) * 1000);
    return `<text x="${left + (pointCount === 1 ? plotWidth / 2 : index * plotWidth / (pointCount - 1))}" y="${height - 10}" text-anchor="middle" class="chart-axis">${svgEscaped(date.toLocaleString([], {month: 'numeric', day: 'numeric', hour: '2-digit'}))}</text>`;
  }).join('') : '';
  svg.innerHTML = `<title>${svgEscaped(title)}</title>${grid}<line x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}" class="chart-axis-line"/>${draw.content({left, top, plotWidth, plotHeight, y, maximum})}${labels}`;
}

function renderStatisticsCharts(statistics) {
  const hourly = statistics.hourly || [];
  const throughput = document.getElementById('statistics-throughput-chart');
  const concurrency = document.getElementById('statistics-concurrency-chart');
  if (!hourly.length) {
    throughput.textContent = 'No data for this period';
    concurrency.textContent = 'No data for this period';
    return;
  }
  const videoHours = hourly.map(point => Number(point.completed_video_seconds || 0) / 3600);
  chartFrame(throughput, videoHours, 'Completed video hours by hour', {
    hours: hourly,
    content: ({left, top, plotWidth, plotHeight, y}) => {
      const step = plotWidth / hourly.length;
      return hourly.map((point, index) => {
        const value = videoHours[index];
        const barWidth = Math.max(1, step - 1);
        const x = left + index * step + (step - barWidth) / 2;
        const barY = y(value);
        const failure = Number(point.failed_tasks || 0);
        const marker = failure ? `<circle cx="${x + barWidth / 2}" cy="${Math.max(12, barY - 5)}" r="3" class="chart-failure"><title>${failure} failed task(s)</title></circle>` : '';
        return `<rect x="${x}" y="${barY}" width="${barWidth}" height="${Math.max(0, top + plotHeight - barY)}" class="chart-bar"><title>${value.toFixed(2)} video hours · ${point.completed_tasks} completed</title></rect>${marker}`;
      }).join('');
    },
  });
  const concurrencyValues = hourly.flatMap(point => [Number(point.average_concurrency || 0), Number(point.peak_concurrency || 0)]);
  chartFrame(concurrency, concurrencyValues, 'Average and peak in-flight task concurrency by hour', {
    hours: hourly,
    content: ({left, plotWidth, y}) => {
      const x = index => left + (hourly.length === 1 ? plotWidth / 2 : index * plotWidth / (hourly.length - 1));
      const line = field => hourly.map((point, index) => `${index ? 'L' : 'M'}${x(index).toFixed(2)},${y(Number(point[field] || 0)).toFixed(2)}`).join('');
      return `<path d="${line('average_concurrency')}" class="chart-line average"/><path d="${line('peak_concurrency')}" class="chart-line peak"/>`;
    },
  });
}

function renderWorkerStatistics(rows) {
  const body = document.getElementById('statistics-worker-rows');
  body.replaceChildren();
  if (!rows.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 9;
    cell.textContent = 'No completed or failed tasks for this period';
    row.append(cell);
    body.append(row);
    return;
  }
  rows.forEach(worker => {
    const row = document.createElement('tr');
    [worker.worker, worker.slot_count || '–', worker.completed_tasks, worker.failed_tasks,
      hours(worker.video_seconds), hours(worker.processing_seconds), ratio(worker.algorithm_realtime),
      workerHoursPerVideoHour(worker.worker_hours_per_video_hour), ratio(worker.end_to_end_realtime)]
      .forEach(value => {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        row.append(cell);
      });
    body.append(row);
  });
}

function renderStatistics(statistics) {
  const hourly = statistics.hourly || [];
  const averageConcurrency = hourly.length
    ? hourly.reduce((sum, point) => sum + Number(point.in_flight_worker_seconds || 0), 0) /
      (hourly.length * Number(statistics.bucket_seconds || 3600))
    : 0;
  const peakConcurrency = hourly.reduce((maximum, point) => Math.max(maximum, Number(point.peak_concurrency || 0)), 0);
  document.getElementById('stat-completed').textContent = statistics.completed_tasks ?? 0;
  document.getElementById('stat-success-note').textContent = `${statistics.failed_tasks || 0} failed · ${Number.isFinite(statistics.success_rate) ? (statistics.success_rate * 100).toFixed(1) : '–'}% success`;
  document.getElementById('stat-video-hours').textContent = hours(statistics.video_seconds);
  document.getElementById('stat-throughput-note').textContent = `${ratio(statistics.calendar_realtime)} video hours per calendar hour`;
  document.getElementById('stat-algorithm-speed').textContent = ratio(statistics.algorithm_realtime);
  document.getElementById('stat-worker-hours-note').textContent = `${workerHoursPerVideoHour(statistics.worker_hours_per_video_hour)} Worker time / video hour`;
  document.getElementById('stat-concurrency').textContent = `${averageConcurrency.toFixed(2)} / ${peakConcurrency}`;
  document.getElementById('stat-concurrency-note').textContent = 'Average / peak in-flight Worker slots';
  renderStatisticsCharts(statistics);
  renderWorkerStatistics(statistics.workers || []);
}

if (statisticsPayload) {
  try {
    renderStatistics(JSON.parse(statisticsPayload.textContent));
  } catch (error) {
    console.error('Unable to render processing statistics', error);
  }
}

function localDateTimeValue(date) {
  const pad = value => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

const algorithmDefaultsForm = document.getElementById('algorithm-defaults-form');
if (algorithmDefaultsForm) {
  const algorithmDefaultsName = document.getElementById('algorithm-defaults-name');
  const algorithmDefaultsArgs = document.getElementById('algorithm-defaults-args');
  const algorithmDefaultsMessage = document.getElementById('algorithm-defaults-message');
  configureAlgorithmArgumentsInput(algorithmDefaultsArgs);

  algorithmDefaultsForm.addEventListener('submit', async event => {
    event.preventDefault();
    let parsedArgs;
    try {
      parsedArgs = parseAlgorithmArguments(algorithmDefaultsArgs.value);
    } catch (_) {
      algorithmDefaultsMessage.textContent = 'Algorithm parameters must be a JSON array or command-line string.';
      algorithmDefaultsMessage.classList.add('error');
      return;
    }
    algorithmDefaultsMessage.textContent = 'Saving...';
    algorithmDefaultsMessage.classList.remove('error');
    try {
      const response = await fetch('/api/admin/algorithm-defaults', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({algorithm: algorithmDefaultsName.value.trim(), arguments: parsedArgs}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      algorithmDefaultsMessage.textContent = 'Saved successfully.';
      algorithmDefaultsMessage.classList.remove('error');
    } catch (error) {
      algorithmDefaultsMessage.textContent = error instanceof Error ? error.message : 'Unable to save algorithm defaults.';
      algorithmDefaultsMessage.classList.add('error');
    }
  });
}

document.getElementById('statistics-preset')?.addEventListener('change', event => {
  const hoursBack = Number(event.target.value);
  if (!hoursBack) return;
  const end = new Date();
  const start = new Date(end.getTime() - hoursBack * 3600 * 1000);
  document.getElementById('statistics-start').value = localDateTimeValue(start);
  document.getElementById('statistics-end').value = localDateTimeValue(end);
  statisticsFilter.requestSubmit();
});
