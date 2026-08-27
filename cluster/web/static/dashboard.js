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

const taskDispatchControl = document.getElementById('task-dispatch-control');
if (taskDispatchControl) {
  let enabled = taskDispatchControl.dataset.enabled === 'true';
  const toggle = document.getElementById('task-dispatch-toggle');
  const status = document.getElementById('task-dispatch-status');
  const message = document.getElementById('task-dispatch-message');

  function renderTaskDispatchState() {
    const label = enabled ? 'Task dispatch enabled' : 'Task dispatch paused';
    status.className = `s3-ingest-status-light ${enabled ? 'enabled' : 'paused'}`;
    status.setAttribute('aria-label', label);
    status.title = label;
    toggle.textContent = enabled ? 'Pause task dispatch' : 'Resume task dispatch';
  }

  toggle.addEventListener('click', async () => {
    const nextEnabled = !enabled;
    const action = nextEnabled ? 'resume' : 'pause';
    const detail = nextEnabled
      ? 'Pending tasks will be available to Workers again.'
      : 'Running tasks continue, but Workers will not receive any new pending tasks.';
    if (!window.confirm(`Do you want to ${action} task dispatch? ${detail}`)) return;
    toggle.disabled = true;
    message.textContent = '';
    try {
      const response = await fetch('/api/admin/task-dispatch', {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: nextEnabled}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      enabled = Boolean(payload.enabled);
      renderTaskDispatchState();
      message.textContent = '';
    } catch (error) {
      message.textContent = error instanceof Error ? error.message : 'Unable to update task dispatch.';
    } finally {
      toggle.disabled = false;
    }
  });
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
    const status = fileButton.closest('tr')?.querySelector('.badge')?.textContent?.trim();
    if (fileButton.dataset.file === 'output' && status !== 'completed') return;
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

let taskShareModal;

function selectedTaskIdsForShare() {
  return [...document.querySelectorAll('[data-share-task]:checked')]
    .map(input => input.dataset.shareTask).filter(Boolean);
}

function updateTaskShareButton() {
  const button = document.getElementById('task-share-selected');
  if (!button) return;
  const count = selectedTaskIdsForShare().length;
  button.disabled = count === 0;
  button.textContent = count ? `Share selected (${count})` : 'Share selected';
}

function ensureTaskShareSelection() {
  if (!taskTable) return;
  const table = taskTable.querySelector('table');
  const header = table?.querySelector('thead tr');
  if (!table || !header) return;
  if (!header.querySelector('.task-share-heading')) {
    const cell = document.createElement('th');
    cell.className = 'task-share-heading';
    const selectAll = document.createElement('input');
    selectAll.type = 'checkbox';
    selectAll.title = 'Select all tasks on this page';
    selectAll.setAttribute('aria-label', 'Select all tasks on this page');
    selectAll.addEventListener('change', () => {
      table.querySelectorAll('[data-share-task]').forEach(input => { input.checked = selectAll.checked; });
      updateTaskShareButton();
    });
    cell.append(selectAll);
    header.insertBefore(cell, header.firstElementChild);
  }
  table.querySelectorAll('tbody tr').forEach(row => {
    if (row.querySelector('.task-share-cell')) return;
    const playButton = row.querySelector('.file-play[data-task-id]');
    if (!playButton) {
      row.querySelector('td')?.setAttribute('colspan', String(header.children.length));
      return;
    }
    const cell = document.createElement('td');
    cell.className = 'task-share-cell';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.dataset.shareTask = playButton.dataset.taskId;
    input.setAttribute('aria-label', `Select task ${playButton.dataset.taskId} for customer sharing`);
    input.addEventListener('change', updateTaskShareButton);
    cell.append(input);
    row.insertBefore(cell, row.firstElementChild);
  });
  if (!document.getElementById('task-share-selected')) {
    const button = document.createElement('button');
    button.id = 'task-share-selected';
    button.type = 'button';
    button.className = 'task-search-button task-share-button';
    button.addEventListener('click', openTaskShareModal);
    const exportButton = document.querySelector('.task-export-button');
    (exportButton?.parentElement || document.querySelector('.section-head'))?.append(button);
  }
  updateTaskShareButton();
}

function closeTaskShareModal() {
  if (!taskShareModal || taskShareModal.hidden) return;
  taskShareModal.hidden = true;
  scheduleAutoRefresh();
}

function copyShareLink(value, button) {
  const fallback = () => {
    const input = document.createElement('textarea');
    input.value = value;
    input.style.cssText = 'position:fixed;opacity:0';
    document.body.append(input);
    input.select();
    document.execCommand('copy');
    input.remove();
  };
  const copied = navigator.clipboard?.writeText ? navigator.clipboard.writeText(value).catch(fallback) : Promise.resolve().then(fallback);
  copied.then(() => { button.textContent = 'Copied'; }).catch(() => window.alert('Unable to copy the share link.'));
}

function ensureTaskShareModal() {
  if (taskShareModal) return;
  document.body.insertAdjacentHTML('beforeend', `
    <div id="task-share-modal" class="play-modal" hidden>
      <div class="play-modal-backdrop" data-task-share-close></div>
      <div class="play-modal-card" role="dialog" aria-modal="true" aria-labelledby="task-share-modal-title">
        <div class="play-modal-head"><div><h3 id="task-share-modal-title">Share videos with customer</h3><p id="task-share-summary" class="play-modal-sub">Selected tasks</p></div><button class="play-modal-close" type="button" data-task-share-close aria-label="Close">&times;</button></div>
        <form id="task-share-form" class="task-share-form">
          <label class="task-share-option"><input name="share-file" type="checkbox" value="output" checked> Processed output</label>
          <label class="task-share-option"><input name="share-file" type="checkbox" value="input"> Source video</label>
          <label class="task-share-expiry">Link expires in <input id="task-share-expiry" type="number" min="1" max="30" value="7" required> days</label>
          <p id="task-share-message" class="play-modal-note">The customer will not need a Controller account. The link works only for the selected files.</p>
          <div id="task-share-link-area" hidden><input id="task-share-link" class="task-share-link" type="text" readonly aria-label="Customer share link"><button id="task-share-copy" class="play-btn play-btn-open" type="button">Copy link</button></div>
          <div class="play-modal-actions"><button id="task-share-submit" class="play-btn play-btn-open" type="submit">Create share link</button><button class="play-btn" type="button" data-task-share-close>Cancel</button></div>
        </form>
      </div>
    </div>`);
  taskShareModal = document.getElementById('task-share-modal');
  taskShareModal.querySelectorAll('[data-task-share-close]').forEach(element => element.addEventListener('click', closeTaskShareModal));
  taskShareModal.querySelector('#task-share-form').addEventListener('submit', submitTaskShare);
  taskShareModal.querySelector('#task-share-copy').addEventListener('click', () => {
    const input = taskShareModal.querySelector('#task-share-link');
    copyShareLink(input.value, taskShareModal.querySelector('#task-share-copy'));
  });
}

function openTaskShareModal() {
  const taskIds = selectedTaskIdsForShare();
  if (!taskIds.length) return;
  ensureTaskShareModal();
  taskShareModal.querySelector('#task-share-summary').textContent = `${taskIds.length} task${taskIds.length === 1 ? '' : 's'} selected`;
  taskShareModal.querySelector('#task-share-message').textContent = 'The customer will not need a Controller account. The link works only for the selected files.';
  taskShareModal.querySelector('#task-share-link-area').hidden = true;
  taskShareModal.querySelector('#task-share-copy').textContent = 'Copy link';
  taskShareModal.hidden = false;
  pauseAutoRefresh();
}

async function submitTaskShare(event) {
  event.preventDefault();
  const taskIds = selectedTaskIdsForShare();
  const files = [...taskShareModal.querySelectorAll('[name="share-file"]:checked')].map(input => input.value);
  const expiry = Number(taskShareModal.querySelector('#task-share-expiry').value);
  const message = taskShareModal.querySelector('#task-share-message');
  const submit = taskShareModal.querySelector('#task-share-submit');
  if (!files.length) { message.textContent = 'Select at least one file type.'; return; }
  if (!Number.isInteger(expiry) || expiry < 1 || expiry > 30) { message.textContent = 'Expiry must be between 1 and 30 days.'; return; }
  submit.disabled = true;
  message.textContent = 'Creating customer link…';
  try {
    const response = await fetch('/api/dashboard/shares', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_ids: taskIds, files, expires_in_days: expiry}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    const link = new URL(payload.url, window.location.origin).href;
    taskShareModal.querySelector('#task-share-link').value = link;
    taskShareModal.querySelector('#task-share-link-area').hidden = false;
    message.textContent = `Link created for ${payload.item_count} video file${payload.item_count === 1 ? '' : 's'}. It expires in ${expiry} days.`;
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : 'Unable to create a share link.';
  } finally {
    submit.disabled = false;
  }
}

ensureTaskShareSelection();
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeTaskShareModal(); });

let contentCategoryState = {categories: [], assignments: {}, shareId: '', shareFile: 'input', shareLimit: 10};
let contentCategoryModal;

function contentCategoryTaskIds() {
  return [...new Set([...document.querySelectorAll('.file-play[data-task-id]')]
    .map(button => button.dataset.taskId).filter(Boolean))];
}

function renderContentCategoryCells() {
  document.querySelectorAll('.content-category-cell').forEach(cell => {
    const taskId = cell.dataset.taskId;
    const selected = contentCategoryState.assignments[taskId];
    const select = document.createElement('select');
    select.className = 'content-category-select';
    select.setAttribute('aria-label', `Set content category for task ${taskId}`);
    select.append(new Option('Unclassified', ''));
    contentCategoryState.categories.forEach(category => {
      select.append(new Option(category.name, String(category.category_id), false,
        Number(selected) === Number(category.category_id)));
    });
    select.addEventListener('change', async () => {
      select.disabled = true;
      try {
        const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskId)}/content-category`, {
          method: 'PUT', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({category_id: select.value ? Number(select.value) : null}),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
        contentCategoryState.assignments[taskId] = payload.content_category_id ?? null;
      } catch (error) {
        window.alert(error instanceof Error ? error.message : 'Unable to set content category.');
        select.value = selected ? String(selected) : '';
      } finally {
        select.disabled = false;
      }
    });
    cell.replaceChildren(select);
  });
}

function ensureContentCategoryColumn() {
  if (!taskTable) return;
  const table = taskTable.querySelector('table');
  const header = table?.querySelector('thead tr');
  if (!table || !header) return;
  if (!header.querySelector('.content-category-heading')) {
    const cell = document.createElement('th');
    cell.className = 'content-category-heading';
    cell.textContent = 'Content category';
    header.insertBefore(cell, header.children[2] || null);
  }
  table.querySelectorAll('tbody tr').forEach(row => {
    if (row.querySelector('.content-category-cell')) return;
    const playButton = row.querySelector('.file-play[data-task-id]');
    if (!playButton) {
      row.querySelector('td')?.setAttribute('colspan', String(header.children.length));
      return;
    }
    const cell = document.createElement('td');
    cell.className = 'content-category-cell';
    cell.dataset.taskId = playButton.dataset.taskId;
    row.insertBefore(cell, row.children[2] || null);
  });
}

async function loadContentCategories() {
  ensureContentCategoryColumn();
  const taskIds = contentCategoryTaskIds();
  try {
    const response = await fetch(`/api/dashboard/content-categories?task_ids=${encodeURIComponent(taskIds.join(','))}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    contentCategoryState = {
      categories: Array.isArray(payload.categories) ? payload.categories : [],
      assignments: payload.assignments || {}, shareId: payload.share_id || '',
      shareFile: payload.share_file === 'output' ? 'output' : 'input',
      shareLimit: Number(payload.share_max_videos_per_category) || 10,
    };
    renderContentCategoryCells();
    renderContentCategoryManager();
  } catch (error) {
    console.warn('[video-mask] unable to load content categories', error);
  }
}

function ensureContentCategoryManager() {
  if (document.getElementById('content-category-manage')) return;
  const button = document.createElement('button');
  button.id = 'content-category-manage';
  button.type = 'button';
  button.className = 'task-search-button content-category-manage-button';
  button.textContent = 'Manage content categories';
  button.addEventListener('click', openContentCategoryModal);
  const exportButton = document.querySelector('.task-export-button');
  (exportButton?.parentElement || document.querySelector('.section-head'))?.append(button);
}

function closeContentCategoryModal() {
  if (!contentCategoryModal || contentCategoryModal.hidden) return;
  contentCategoryModal.hidden = true;
  scheduleAutoRefresh();
}

function renderContentCategoryManager() {
  if (!contentCategoryModal) return;
  const list = contentCategoryModal.querySelector('#content-category-list');
  list.replaceChildren();
  if (!contentCategoryState.categories.length) {
    list.textContent = 'No categories yet.';
  } else {
    contentCategoryState.categories.forEach(category => {
      const row = document.createElement('div');
      row.className = 'content-category-row';
      const label = document.createElement('span');
      label.textContent = `${category.name} (${category.task_count} task${category.task_count === 1 ? '' : 's'})`;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'play-btn'; remove.textContent = 'Hide';
      remove.title = category.task_count ? `Hide category while retaining its ${category.task_count} task assignment(s)` : 'Hide category';
      remove.addEventListener('click', () => deleteContentCategory(category));
      row.append(label, remove); list.append(row);
    });
  }
  const shareInput = contentCategoryModal.querySelector('#content-category-share-id');
  if (document.activeElement !== shareInput) shareInput.value = contentCategoryState.shareId;
  const shareFile = contentCategoryModal.querySelector('#content-category-share-file');
  if (document.activeElement !== shareFile) shareFile.value = contentCategoryState.shareFile;
  const shareLimit = contentCategoryModal.querySelector('#content-category-share-limit');
  if (document.activeElement !== shareLimit) shareLimit.value = String(contentCategoryState.shareLimit || 10);
}

function ensureContentCategoryModal() {
  if (contentCategoryModal) return;
  document.body.insertAdjacentHTML('beforeend', `
    <div id="content-category-modal" class="play-modal" hidden>
      <div class="play-modal-backdrop" data-content-category-close></div>
      <div class="play-modal-card content-category-modal-card" role="dialog" aria-modal="true" aria-labelledby="content-category-modal-title">
        <div class="play-modal-head"><div><h3 id="content-category-modal-title">Content categories</h3><p class="play-modal-sub">One category can be selected for each task.</p></div><button class="play-modal-close" type="button" data-content-category-close aria-label="Close">&times;</button></div>
        <form id="content-category-add-form" class="content-category-add-form"><input id="content-category-name" type="text" maxlength="64" placeholder="e.g. Laundry" required><button class="play-btn play-btn-open" type="submit">Add category</button></form>
        <div id="content-category-list" class="content-category-list"></div>
        <section class="content-category-public"><h4>Public category page</h4><p class="play-modal-note">This page is public. Use a hard-to-guess ID (16–64 letters, numbers, hyphens, or underscores).</p><label class="content-category-share-file" for="content-category-share-file">Share video<select id="content-category-share-file"><option value="input">Source video</option><option value="output">Processed video</option></select></label><label class="content-category-share-file" for="content-category-share-limit">Videos per category<input id="content-category-share-limit" type="number" min="1" max="100" value="10" required></label><div class="content-category-share-entry"><input id="content-category-share-id" type="text" maxlength="64" placeholder="special share ID"><button id="content-category-share-generate" class="play-btn" type="button">Generate</button><button id="content-category-share-save" class="play-btn play-btn-open" type="button">Save</button></div><div id="content-category-share-link-area" hidden><input id="content-category-share-link" class="task-share-link" type="text" readonly aria-label="Public category page link"><button id="content-category-share-copy" class="play-btn play-btn-open" type="button">Copy link</button></div><p id="content-category-message" class="play-modal-note"></p></section>
        <div class="play-modal-actions"><button class="play-btn" type="button" data-content-category-close>Close</button></div>
      </div>
    </div>`);
  contentCategoryModal = document.getElementById('content-category-modal');
  contentCategoryModal.querySelectorAll('[data-content-category-close]').forEach(item => item.addEventListener('click', closeContentCategoryModal));
  contentCategoryModal.querySelector('#content-category-add-form').addEventListener('submit', createContentCategory);
  contentCategoryModal.querySelector('#content-category-share-generate').addEventListener('click', () => {
    const value = crypto.randomUUID ? crypto.randomUUID().replaceAll('-', '') : Math.random().toString(36).slice(2).padEnd(24, '0');
    contentCategoryModal.querySelector('#content-category-share-id').value = value;
  });
  contentCategoryModal.querySelector('#content-category-share-save').addEventListener('click', saveContentCategoryShare);
  contentCategoryModal.querySelector('#content-category-share-copy').addEventListener('click', () => {
    const input = contentCategoryModal.querySelector('#content-category-share-link');
    copyShareLink(input.value, contentCategoryModal.querySelector('#content-category-share-copy'));
  });
}

function openContentCategoryModal() {
  ensureContentCategoryModal();
  renderContentCategoryManager();
  contentCategoryModal.querySelector('#content-category-share-link-area').hidden = true;
  contentCategoryModal.querySelector('#content-category-message').textContent = '';
  contentCategoryModal.hidden = false;
  pauseAutoRefresh();
}

async function createContentCategory(event) {
  event.preventDefault();
  const input = contentCategoryModal.querySelector('#content-category-name');
  const message = contentCategoryModal.querySelector('#content-category-message');
  try {
    const response = await fetch('/api/dashboard/content-categories', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: input.value}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    input.value = ''; message.textContent = 'Category added.'; await loadContentCategories();
  } catch (error) { message.textContent = error instanceof Error ? error.message : 'Unable to add category.'; }
}

async function deleteContentCategory(category) {
  const categoryId = category.category_id;
  const message = contentCategoryModal.querySelector('#content-category-message');
  const detail = category.task_count ? ` Its ${category.task_count} task assignment(s) will be retained.` : '';
  if (!window.confirm(`Hide category “${category.name}”?${detail}`)) return;
  try {
    const response = await fetch(`/api/dashboard/content-categories/${encodeURIComponent(categoryId)}`, {method: 'DELETE'});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    const retained = Number(payload.retained_tasks) || 0;
    message.textContent = retained ? `Category hidden; ${retained} task assignment(s) retained.` : 'Category hidden.';
    await loadContentCategories();
  } catch (error) { message.textContent = error instanceof Error ? error.message : 'Unable to delete category.'; }
}

async function saveContentCategoryShare() {
  const input = contentCategoryModal.querySelector('#content-category-share-id');
  const file = contentCategoryModal.querySelector('#content-category-share-file');
  const limit = contentCategoryModal.querySelector('#content-category-share-limit');
  const message = contentCategoryModal.querySelector('#content-category-message');
  try {
    const response = await fetch('/api/dashboard/content-category-share', {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({share_id: input.value.trim(), file: file.value,
        max_videos_per_category: Number(limit.value)}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    contentCategoryState.shareId = payload.share_id;
    contentCategoryState.shareFile = payload.share_file === 'output' ? 'output' : 'input';
    contentCategoryState.shareLimit = Number(payload.max_videos_per_category) || 10;
    contentCategoryModal.querySelector('#content-category-share-link').value = new URL(payload.url, window.location.origin).href;
    contentCategoryModal.querySelector('#content-category-share-link-area').hidden = false;
    message.textContent = 'Public category page saved.';
  } catch (error) { message.textContent = error instanceof Error ? error.message : 'Unable to save public page.'; }
}

ensureContentCategoryManager();
loadContentCategories();
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeContentCategoryModal(); });

function configureManualLabelFilters() {
  const form = document.getElementById('task-status-filter');
  const oldFilter = form?.querySelector('.annotation-filter');
  if (!form || !oldFilter) return;
  const faceOptions = [...oldFilter.querySelectorAll('input[name="face_annotation"]')].map(input => ({
    value: input.value, label: input.parentElement?.textContent?.trim() || input.value, selected: input.checked,
  }));
  const faceLabel = document.createElement('label');
  faceLabel.className = 'subtle';
  faceLabel.htmlFor = 'face-annotation-filter-select';
  faceLabel.textContent = 'Manual label';
  const faceSelect = document.createElement('select');
  faceSelect.id = 'face-annotation-filter-select';
  faceSelect.className = 'fleet-filter';
  faceSelect.name = 'face_annotation';
  faceSelect.setAttribute('aria-label', 'Filter tasks by manual face label');
  const allFaces = new Option('All labels', '');
  faceSelect.add(allFaces);
  faceOptions.forEach(option => faceSelect.add(new Option(option.label, option.value, false, option.selected)));

  const selectedTags = new Set(new URLSearchParams(window.location.search).getAll('content_tag'));
  const selectedCategories = new Set(new URLSearchParams(window.location.search).getAll('content_category'));
  const tagFilter = document.createElement('details');
  tagFilter.className = 'content-tag-filter';
  tagFilter.open = selectedTags.size > 0;
  const summary = document.createElement('summary');
  summary.textContent = selectedTags.size ? `Content tags (${selectedTags.size})` : 'Content tags';
  const options = document.createElement('div');
  options.className = 'content-tag-filter-options';
  options.textContent = 'Loading tags…';
  tagFilter.append(summary, options);
  const categoryFilter = document.createElement('details');
  categoryFilter.className = 'content-tag-filter';
  categoryFilter.open = selectedCategories.size > 0;
  const categorySummary = document.createElement('summary');
  categorySummary.textContent = selectedCategories.size ? `Content categories (${selectedCategories.size})` : 'Content categories';
  const categoryOptions = document.createElement('div');
  categoryOptions.className = 'content-tag-filter-options';
  categoryOptions.textContent = 'Loading categories…';
  categoryFilter.append(categorySummary, categoryOptions);
  oldFilter.replaceWith(faceLabel, faceSelect, tagFilter, categoryFilter);

  fetch('/api/content-tags').then(async response => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Unable to load content tags');
    return Array.isArray(payload.tags) ? payload.tags : [];
  }).then(tags => {
    options.replaceChildren();
    if (!tags.length) {
      options.textContent = 'No content tags yet';
      return;
    }
    tags.forEach(tag => {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.name = 'content_tag';
      checkbox.value = tag;
      checkbox.checked = selectedTags.has(tag);
      label.append(checkbox, document.createTextNode(tag));
      options.append(label);
    });
  }).catch(() => {
    options.textContent = 'Unable to load content tags';
  });

  fetch('/api/dashboard/content-categories').then(async response => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Unable to load content categories');
    return Array.isArray(payload.categories) ? payload.categories : [];
  }).then(categories => {
    categoryOptions.replaceChildren();
    if (!categories.length) {
      categoryOptions.textContent = 'No content categories yet';
      return;
    }
    categories.forEach(category => {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.name = 'content_category';
      checkbox.value = String(category.category_id);
      checkbox.checked = selectedCategories.has(checkbox.value);
      label.append(checkbox, document.createTextNode(`${category.name} (${category.task_count})`));
      categoryOptions.append(label);
    });
  }).catch(() => {
    categoryOptions.textContent = 'Unable to load content categories';
  });
}

configureManualLabelFilters();

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
  const contentTagInput = document.getElementById('content-tag-input');
  const contentTagAddButton = document.getElementById('content-tag-add');
  const contentTagsSaveButton = document.getElementById('content-tags-save');
  const contentTagSelected = document.getElementById('content-tag-selected');
  const contentTagOptions = document.getElementById('content-tag-options');
  const contentTagSuggestions = document.getElementById('content-tag-suggestions');
  const contentTagMessage = document.getElementById('content-tag-message');
  const reviewContentCategory = document.getElementById('face-review-content-category');
  const reviewContentCategoryMessage = document.getElementById('face-review-content-category-message');
  const modalAnnotation = document.getElementById('play-modal-annotation');
  const modalAnnotationStatus = document.getElementById('play-modal-annotation-status');
  const modalHasFaceButton = document.getElementById('play-modal-has-face');
  const modalNoFaceButton = document.getElementById('play-modal-no-face');
  const modalContentTags = document.getElementById('play-modal-content-tags');
  const modalContentTagInput = document.getElementById('play-modal-content-tag-input');
  const modalContentTagAddButton = document.getElementById('play-modal-content-tag-add');
  const modalContentTagsSaveButton = document.getElementById('play-modal-content-tags-save');
  const modalContentTagSelected = document.getElementById('play-modal-content-tag-selected');
  const modalContentTagOptions = document.getElementById('play-modal-content-tag-options');
  const modalContentTagSuggestions = document.getElementById('play-modal-content-tag-suggestions');
  const modalContentTagMessage = document.getElementById('play-modal-content-tag-message');
  const modalCategoryControl = document.getElementById('play-modal-category-control');
  const modalContentCategory = document.getElementById('play-modal-content-category');
  const modalContentCategoryMessage = document.getElementById('play-modal-content-category-message');
  let heartbeatTimer;
  let framePreviewTimer;
  let modalHeartbeatTimer;
  let activeModalReview = null;
  let modalReviewToken = 0;
  let selectedContentTags = [];
  let selectedModalContentTags = [];
  let knownContentTags = [];

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
    faceButton.disabled = busy || !activeFaceReview;
    noFaceButton.disabled = busy || !activeFaceReview;
    releaseButton.disabled = busy || !activeFaceReview;
    contentTagInput.disabled = busy || !activeFaceReview;
    contentTagAddButton.disabled = busy || !activeFaceReview;
    contentTagsSaveButton.disabled = busy || !activeFaceReview || !selectedContentTags.length;
    reviewContentCategory.disabled = busy || !activeFaceReview;
  }

  function normaliseContentTag(value) {
    return String(value || '').trim().replace(/\s+/g, ' ');
  }

  function renderContentTags() {
    contentTagSelected.replaceChildren();
    selectedContentTags.forEach(tag => {
      const chip = document.createElement('span');
      chip.className = 'content-tag-chip';
      chip.textContent = tag;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${tag}`);
      remove.textContent = '×';
      remove.addEventListener('click', () => {
        selectedContentTags = selectedContentTags.filter(value => value !== tag);
        renderContentTags();
        setReviewControls();
      });
      chip.append(remove);
      contentTagSelected.append(chip);
    });
    contentTagOptions.replaceChildren();
    contentTagSuggestions.replaceChildren();
    const selected = new Set(selectedContentTags.map(tag => tag.toLocaleLowerCase()));
    knownContentTags.forEach(tag => {
      const option = document.createElement('option');
      option.value = tag;
      contentTagOptions.append(option);
      if (selected.has(tag.toLocaleLowerCase())) return;
      const suggestion = document.createElement('button');
      suggestion.type = 'button';
      suggestion.className = 'content-tag-suggestion';
      suggestion.textContent = `+ ${tag}`;
      suggestion.addEventListener('click', () => addContentTag(tag));
      contentTagSuggestions.append(suggestion);
    });
  }

  function setSelectedContentTags(tags) {
    const seen = new Set();
    selectedContentTags = (Array.isArray(tags) ? tags : []).map(normaliseContentTag).filter(tag => {
      const key = tag.toLocaleLowerCase();
      if (!tag || seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 20);
    renderContentTags();
  }

  function addContentTag(value = contentTagInput.value) {
    const candidates = String(value || '').split(/[,，;；\n]/).map(normaliseContentTag).filter(Boolean);
    let added = false;
    candidates.forEach(tag => {
      if (selectedContentTags.length >= 20 || selectedContentTags.some(value => value.toLocaleLowerCase() === tag.toLocaleLowerCase())) return;
      selectedContentTags.push(tag);
      added = true;
    });
    contentTagInput.value = '';
    if (added) {
      contentTagMessage.textContent = 'Tags ready to save.';
      renderContentTags();
      setReviewControls();
    }
  }

  async function loadKnownContentTags() {
    try {
      const payload = await reviewRequest('/api/content-tags');
      knownContentTags = Array.isArray(payload.tags) ? payload.tags.map(normaliseContentTag).filter(Boolean) : [];
      renderContentTags();
      renderModalContentTags();
    } catch (error) {
      console.warn('[video-mask] unable to load content-tag suggestions', error);
    }
  }

  function modalAnnotationText(faceAnnotation) {
    if (faceAnnotation === 1) return 'Current label: 👍 Has face';
    if (faceAnnotation === 0) return 'Current label: 👎 No face';
    return 'Current label: unlabelled';
  }

  function setModalAnnotationControls(enabled) {
    modalHasFaceButton.disabled = !enabled;
    modalNoFaceButton.disabled = !enabled;
    modalContentTagInput.disabled = !enabled;
    modalContentTagAddButton.disabled = !enabled;
    modalContentTagsSaveButton.disabled = !enabled || !selectedModalContentTags.length;
    modalContentCategory.disabled = !enabled;
  }

  function renderReviewContentCategoryOptions(select, selectedCategoryId) {
    if (!select) return;
    const selected = selectedCategoryId == null ? '' : String(selectedCategoryId);
    select.replaceChildren(new Option('未分类', ''));
    contentCategoryState.categories.forEach(category => {
      select.append(new Option(category.name, String(category.category_id)));
    });
    select.value = selected;
  }

  async function loadReviewContentCategory(task, select, message, refreshControls) {
    if (!task?.task_id) return;
    const taskId = task.task_id;
    select.disabled = true;
    message.textContent = '加载分类中…';
    try {
      const response = await fetch(
        `/api/dashboard/content-categories?task_ids=${encodeURIComponent(taskId)}`,
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      contentCategoryState.categories = Array.isArray(payload.categories) ? payload.categories : [];
      Object.assign(contentCategoryState.assignments, payload.assignments || {});
      contentCategoryState.shareId = payload.share_id || contentCategoryState.shareId;
      contentCategoryState.shareFile = payload.share_file === 'output' ? 'output' : 'input';
      task.content_category_id = payload.assignments?.[taskId] ?? null;
      renderContentCategoryCells();
      renderReviewContentCategoryOptions(select, task.content_category_id);
      message.textContent = contentCategoryState.categories.length
        ? '' : '请先在任务列表中创建内容分类。';
    } catch (error) {
      message.textContent = error instanceof Error ? error.message : '无法加载内容分类。';
      renderReviewContentCategoryOptions(select, task.content_category_id);
    } finally {
      refreshControls();
    }
  }

  async function saveReviewContentCategory(task, select, message, refreshControls) {
    if (!task?.task_id) return;
    const taskId = task.task_id;
    const previous = task.content_category_id ?? null;
    select.disabled = true;
    message.textContent = '保存分类中…';
    try {
      const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskId)}/content-category`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({category_id: select.value ? Number(select.value) : null}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      task.content_category_id = payload.content_category_id ?? null;
      contentCategoryState.assignments[taskId] = task.content_category_id;
      renderContentCategoryCells();
      renderReviewContentCategoryOptions(select, task.content_category_id);
      message.textContent = '内容分类已保存。';
    } catch (error) {
      renderReviewContentCategoryOptions(select, previous);
      message.textContent = error instanceof Error ? error.message : '无法保存内容分类。';
    } finally {
      refreshControls();
    }
  }

  function renderModalContentTags() {
    modalContentTagSelected.replaceChildren();
    selectedModalContentTags.forEach(tag => {
      const chip = document.createElement('span');
      chip.className = 'content-tag-chip';
      chip.textContent = tag;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${tag}`);
      remove.textContent = '×';
      remove.addEventListener('click', () => {
        selectedModalContentTags = selectedModalContentTags.filter(value => value !== tag);
        renderModalContentTags();
        setModalAnnotationControls(Boolean(activeModalReview));
      });
      chip.append(remove);
      modalContentTagSelected.append(chip);
    });
    modalContentTagOptions.replaceChildren();
    modalContentTagSuggestions.replaceChildren();
    const selected = new Set(selectedModalContentTags.map(tag => tag.toLocaleLowerCase()));
    knownContentTags.forEach(tag => {
      const option = document.createElement('option');
      option.value = tag;
      modalContentTagOptions.append(option);
      if (selected.has(tag.toLocaleLowerCase())) return;
      const suggestion = document.createElement('button');
      suggestion.type = 'button';
      suggestion.className = 'content-tag-suggestion';
      suggestion.textContent = `+ ${tag}`;
      suggestion.addEventListener('click', () => addModalContentTag(tag));
      modalContentTagSuggestions.append(suggestion);
    });
  }

  function setSelectedModalContentTags(tags) {
    const seen = new Set();
    selectedModalContentTags = (Array.isArray(tags) ? tags : []).map(normaliseContentTag).filter(tag => {
      const key = tag.toLocaleLowerCase();
      if (!tag || seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 20);
    renderModalContentTags();
  }

  function addModalContentTag(value = modalContentTagInput.value) {
    const candidates = String(value || '').split(/[,，;；\n]/).map(normaliseContentTag).filter(Boolean);
    let added = false;
    candidates.forEach(tag => {
      if (selectedModalContentTags.length >= 20 || selectedModalContentTags.some(value => value.toLocaleLowerCase() === tag.toLocaleLowerCase())) return;
      selectedModalContentTags.push(tag);
      added = true;
    });
    modalContentTagInput.value = '';
    if (added) {
      modalContentTagMessage.textContent = 'Tags ready to save.';
      renderModalContentTags();
      setModalAnnotationControls(Boolean(activeModalReview));
    }
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
      modalContentTagMessage.textContent = error instanceof Error ? error.message : 'Manual label lease ended.';
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
      activeModalReview = {...activeModalReview, ...reopened.task};
      setSelectedModalContentTags(reopened.task.content_tags || []);
      refreshFaceReviewStatuses();
    } catch (error) {
      modalAnnotationStatus.textContent = error instanceof Error ? error.message : 'Unable to save manual label.';
    } finally {
      setModalAnnotationControls(Boolean(activeModalReview));
    }
  }

  async function saveModalContentTags() {
    if (!activeModalReview) return;
    addModalContentTag();
    if (!selectedModalContentTags.length) {
      modalContentTagMessage.textContent = 'Add at least one content tag before saving.';
      return;
    }
    setModalAnnotationControls(false);
    modalContentTagMessage.textContent = 'Saving tags…';
    try {
      const task = await reviewRequest(
        `/api/face-reviews/${encodeURIComponent(activeModalReview.task_id)}/content-tags`,
        {method: 'PUT', body: JSON.stringify({reviewer_id: reviewerId, tags: selectedModalContentTags})},
      );
      activeModalReview = {...activeModalReview, ...task};
      setSelectedModalContentTags(task.content_tags || selectedModalContentTags);
      modalContentTagMessage.textContent = 'Content tags saved.';
      await loadKnownContentTags();
    } catch (error) {
      modalContentTagMessage.textContent = error instanceof Error ? error.message : 'Unable to save content tags.';
    } finally {
      setModalAnnotationControls(Boolean(activeModalReview));
    }
  }

  beginModalAnnotation = async taskId => {
    const token = ++modalReviewToken;
    window.clearInterval(modalHeartbeatTimer);
    activeModalReview = null;
    modalAnnotation.hidden = false;
    modalContentTags.hidden = false;
    modalCategoryControl.hidden = false;
    modalAnnotationStatus.textContent = 'Opening manual label…';
    modalContentTagMessage.textContent = 'Opening content labels…';
    modalContentCategoryMessage.textContent = 'Opening content category…';
    setSelectedModalContentTags([]);
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
      activeModalReview = {...payload.task};
      modalAnnotationStatus.textContent = modalAnnotationText(payload.task.face_annotation);
      setSelectedModalContentTags(payload.task.content_tags || []);
      modalContentTagMessage.textContent = 'Choose an existing tag or enter a new one. Multiple tags are supported.';
      setModalAnnotationControls(true);
      await loadReviewContentCategory(
        activeModalReview, modalContentCategory, modalContentCategoryMessage,
        () => setModalAnnotationControls(Boolean(activeModalReview)),
      );
      if (token !== modalReviewToken || playModal.hidden || !activeModalReview) return;
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
    modalContentTags.hidden = true;
    modalCategoryControl.hidden = true;
    setSelectedModalContentTags([]);
    renderReviewContentCategoryOptions(modalContentCategory, null);
    modalContentCategoryMessage.textContent = '';
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
    setSelectedContentTags([]);
    renderReviewContentCategoryOptions(reviewContentCategory, null);
    reviewContentCategoryMessage.textContent = '';
    activeBadge.className = 'badge muted';
    activeBadge.textContent = 'No active review';
    setReviewControls();
    if (!keepMessage) setReviewMessage('Video released. You can claim another video needing a manual label.');
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
    setReviewMessage('Finding a random source video needing content tags…');
    try {
      const payload = await reviewRequest('/api/face-reviews/claim', {
        method: 'POST', body: JSON.stringify({reviewer_id: reviewerId}),
      });
      if (!payload.task) {
        window.alert('No source videos need content tags right now.');
        setReviewControls();
        return;
      }
      activeFaceReview = {...payload.task, playback_url: payload.playback_url};
      setSelectedContentTags(payload.task.content_tags || []);
      reviewDetails.hidden = false;
      player.hidden = false;
      activeBadge.className = 'badge active';
      activeBadge.textContent = 'Review in progress';
      const filename = payload.playback_file === 'output'
        ? (payload.task.output_object_key || 'output video')
        : (payload.task.source_object_key || 'input video');
      setReviewMessage(`Reserved for this browser. Playing ${payload.playback_file} video: ${filename}`);
      setReviewControls();
      await loadReviewContentCategory(
        activeFaceReview, reviewContentCategory, reviewContentCategoryMessage,
        () => setReviewControls(),
      );
      if (!activeFaceReview || activeFaceReview.task_id !== payload.task.task_id) return;
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

  async function saveContentTags() {
    if (!activeFaceReview) return;
    addContentTag();
    if (!selectedContentTags.length) {
      contentTagMessage.textContent = 'Add at least one content tag before saving.';
      return;
    }
    setReviewControls(true);
    contentTagMessage.textContent = 'Saving tags…';
    try {
      const task = await reviewRequest(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/content-tags`, {
        method: 'PUT', body: JSON.stringify({reviewer_id: reviewerId, tags: selectedContentTags}),
      });
      activeFaceReview = {...activeFaceReview, ...task};
      setSelectedContentTags(task.content_tags || selectedContentTags);
      contentTagMessage.textContent = 'Content tags saved.';
      await loadKnownContentTags();
    } catch (error) {
      contentTagMessage.textContent = error instanceof Error ? error.message : 'Unable to save content tags.';
    } finally {
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
      cell.textContent = 'Manual labels';
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
      } else {
        const labels = [];
        if (review.has_face === true) labels.push('👍 Face');
        else if (review.has_face === false) labels.push('👎 No face');
        else if (review.reviewing) labels.push('👀 Reviewing');
        else labels.push('Face: unlabelled');
        const tags = Array.isArray(review.content_tags) ? review.content_tags : [];
        if (tags.length) labels.push(`🏷 ${tags.join(' · ')}`);
        else if (review.has_face !== null) labels.push('🏷 Content: unlabelled');
        cell.textContent = labels.join('\n');
        if (review.has_face !== null || tags.length) cell.classList.add('labelled');
        else if (review.reviewing) cell.classList.add('reviewing');
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
  contentTagAddButton.addEventListener('click', () => addContentTag());
  contentTagInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      addContentTag();
    }
  });
  contentTagsSaveButton.addEventListener('click', saveContentTags);
  reviewContentCategory.addEventListener('change', () => saveReviewContentCategory(
    activeFaceReview, reviewContentCategory, reviewContentCategoryMessage, () => setReviewControls(),
  ));
  releaseButton.addEventListener('click', releaseFaceReview);
  modalHasFaceButton.addEventListener('click', () => saveModalAnnotation(true));
  modalNoFaceButton.addEventListener('click', () => saveModalAnnotation(false));
  modalContentTagAddButton.addEventListener('click', () => addModalContentTag());
  modalContentTagInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      addModalContentTag();
    }
  });
  modalContentTagsSaveButton.addEventListener('click', saveModalContentTags);
  modalContentCategory.addEventListener('change', () => saveReviewContentCategory(
    activeModalReview, modalContentCategory, modalContentCategoryMessage,
    () => setModalAnnotationControls(Boolean(activeModalReview)),
  ));
  window.addEventListener('pagehide', () => {
    if (!activeFaceReview) return;
    fetch(`/api/face-reviews/${encodeURIComponent(activeFaceReview.task_id)}/release`, {
      method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer_id: reviewerId}),
    });
  });
  refreshFaceReviewStatuses();
  loadKnownContentTags();
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
    if (['completed', 'failed', 'cancelled'].includes(status) || ['assigned', 'downloading', 'processing', 'uploading'].includes(status)) {
      const action = ['completed', 'failed', 'cancelled'].includes(status) ? 'restart' : 'cancel';
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
          <label style="display:grid;gap:5px;color:var(--muted);font-size:12px">Algorithm parameters (JSON array or command line)<textarea id="task-restart-arguments" required placeholder="--fisheye --fisheye-device pico4 --face-size 640" style="width:100%;min-height:92px;resize:vertical;border:1px solid var(--line);border-radius:4px;padding:8px;font:13px Consolas,monospace"></textarea></label>
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
  taskRestartMessage.textContent = 'Loading current Settings defaults…';
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
    argumentsValue = parseAlgorithmArguments(taskRestartArguments.value);
  } catch (_) {
    taskRestartMessage.textContent = 'Algorithm parameters must be a JSON array or command-line string.';
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
