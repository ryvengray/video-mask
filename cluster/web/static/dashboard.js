const tabs = document.querySelectorAll('.tab');
const panels = {tasks: document.getElementById('panel-tasks'), workers: document.getElementById('panel-workers')};
function activate(name) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === name));
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle('active', key === name));
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
}
tabs.forEach(tab => tab.addEventListener('click', () => activate(tab.dataset.tab)));
activate(location.hash === '#workers' ? 'workers' : 'tasks');
const workerFilter = document.getElementById('worker-status-filter');
workerFilter.addEventListener('change', () => {
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
  const note = document.getElementById('s3-ingest-note');
  const message = document.getElementById('s3-ingest-message');

  function renderS3IngestState() {
    status.className = `badge ${enabled ? 'success' : 'muted'}`;
    status.textContent = enabled ? 'Enabled' : 'Paused';
    toggle.textContent = enabled ? 'Pause S3 ingestion' : 'Resume S3 ingestion';
    note.textContent = enabled
      ? 'New source videos are discovered automatically'
      : 'Scanning is paused; existing queued tasks continue normally';
  }

  if (configured) {
    toggle.addEventListener('click', async () => {
      const nextEnabled = !enabled;
      const action = nextEnabled ? 'resume' : 'pause';
      if (!window.confirm(`Do you want to ${action} automatic S3 task ingestion?`)) return;
      const token = window.prompt('Enter the Controller admin token to continue:');
      if (!token) return;
      toggle.disabled = true;
      message.textContent = '';
      try {
        const response = await fetch('/api/admin/s3-ingest', {
          method: 'PUT',
          headers: {
            Authorization: `Bearer ${token}`,
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
