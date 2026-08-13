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
