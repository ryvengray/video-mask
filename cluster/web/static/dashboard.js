const tabs = document.querySelectorAll('.tab');
const panels = {
  tasks: document.getElementById('panel-tasks'),
  workers: document.getElementById('panel-workers'),
  statistics: document.getElementById('panel-statistics'),
};
function activate(name) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === name));
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle('active', key === name));
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
}
tabs.forEach(tab => tab.addEventListener('click', () => activate(tab.dataset.tab)));
activate(['tasks', 'workers', 'statistics'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'tasks');
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
  const labels = values.length ? [0, Math.floor((values.length - 1) / 2), values.length - 1].map(index => {
    const date = new Date(draw.hours[index].start_at * 1000);
    return `<text x="${left + (values.length === 1 ? plotWidth / 2 : index * plotWidth / (values.length - 1))}" y="${height - 10}" text-anchor="middle" class="chart-axis">${svgEscaped(date.toLocaleString([], {month: 'numeric', day: 'numeric', hour: '2-digit'}))}</text>`;
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

document.getElementById('statistics-preset')?.addEventListener('change', event => {
  const hoursBack = Number(event.target.value);
  if (!hoursBack) return;
  const end = new Date();
  const start = new Date(end.getTime() - hoursBack * 3600 * 1000);
  document.getElementById('statistics-start').value = localDateTimeValue(start);
  document.getElementById('statistics-end').value = localDateTimeValue(end);
  statisticsFilter.requestSubmit();
});
