const FIELD_W = 422;
const FIELD_H = 288;
let selectedJobId = null;
let fieldImg = null;
let refreshInterval = null;

(function loadFieldImage() {
  const img = new Image();
  img.onload = () => { fieldImg = img; };
  img.onerror = () => {};
  img.src = '/static/football_field.png';
})();

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('nav a').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      const tab = el.dataset.tab;
      document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
      el.classList.add('active');
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
    });
  });

  document.getElementById('upload-form').addEventListener('submit', async e => {
    e.preventDefault();
    const file = document.getElementById('video-file').files[0];
    if (!file) return;
    const resultDiv = document.getElementById('batch-progress');
    resultDiv.innerHTML = 'Uploading...';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('track_enabled', document.getElementById('track-enable').checked ? 'true' : 'false');
    try {
      const res = await fetch('/batch/upload', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }
      const data = await res.json();
      resultDiv.innerHTML = '<span style="color:#fbbf24">✓ Uploaded — job: ' + data.job_id + '</span>';
      if (data.job_id) pollJob(data.job_id);
    } catch (err) {
      resultDiv.innerHTML = '<span style="color:#f87171">✗ Error: ' + err.message + '</span>';
    }
  });

  document.getElementById('batch-detail-close').onclick = () => {
    document.getElementById('batch-player-detail').style.display = 'none';
  };

  checkHealth();
  refreshJobList();
  refreshInterval = setInterval(refreshJobList, 4000);
});

async function checkHealth() {
  const el = document.getElementById('health-status');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    el.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
  } catch (err) {
    el.innerHTML = `<span style="color:#f87171">Error: ${err.message}</span>`;
  }
}

function showPlayerDetail(s, prefix) {
  document.getElementById(prefix + '-id').textContent = s.tracking_id;
  document.getElementById(prefix + '-label').textContent = s.label;
  document.getElementById(prefix + '-team').textContent = s.team === 0 ? 'Team 1' : s.team === 1 ? 'Team 2' : '—';
  document.getElementById(prefix + '-distance').textContent = s.total_distance.toFixed(1);
  document.getElementById(prefix + '-avg-speed').textContent = s.avg_speed.toFixed(2);
  document.getElementById(prefix + '-top-speed').textContent = s.top_speed.toFixed(2);
  document.getElementById(prefix + '-touches').textContent = s.touches;
  const canvas = document.getElementById(prefix + '-heatmap-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (fieldImg && fieldImg.complete && fieldImg.naturalWidth > 0) {
      ctx.drawImage(fieldImg, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#1e3a2f';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (s.heatmap_positions && s.heatmap_positions.length > 0) {
      const w = 211, h = 144;
      const max = Math.min(s.heatmap_positions.length, 1000);
      s.heatmap_positions.slice(-max).forEach(p => {
        const x = (p[0] / FIELD_W) * w;
        const y = (p[1] / FIELD_H) * h;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,200,50,0.6)';
        ctx.fill();
      });
    }
  }
  document.getElementById('batch-player-detail').style.display = 'block';
}

// --- Batch ---

function closeBatchDetail() {
  document.getElementById('batch-detail-section').style.display = 'none';
  document.getElementById('batch-result-video').pause();
  selectedJobId = null;
}

async function refreshJobList() {
  const tbody = document.getElementById('job-history-body');
  try {
    const res = await fetch('/batch/jobs');
    const jobs = await res.json();
    if (!jobs || jobs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6">No jobs yet — upload a video above</td></tr>';
      return;
    }
    tbody.innerHTML = jobs.map(j => {
      let statusHtml, actionsHtml;
      if (j.status === 'pending') {
        statusHtml = '<span style="color:#fbbf24">● Pending</span>';
        actionsHtml = '<button class="btn" disabled style="opacity:0.5;font-size:0.75rem;padding:0.25rem 0.5rem">Pending</button>';
      } else if (j.status === 'processing') {
        const pct = Math.round(j.progress * 100);
        statusHtml = `<span style="color:#fbbf24">● ${pct}%</span>`;
        actionsHtml = '<button class="btn" disabled style="opacity:0.5;font-size:0.75rem;padding:0.25rem 0.5rem">Processing</button>';
      } else if (j.status === 'done') {
        statusHtml = '<span style="color:#fbbf24">✓ Done</span>';
        actionsHtml = `
          <button onclick="viewJob('${j.job_id}')" class="btn" style="font-size:0.75rem;padding:0.25rem 0.5rem">View</button>
          <button onclick="deleteJob('${j.job_id}')" class="btn" style="font-size:0.75rem;padding:0.25rem 0.5rem;border-color:#f87171;color:#f87171">Delete</button>
        `;
      } else if (j.status === 'failed') {
        statusHtml = `<span style="color:#f87171">✗ Failed</span>`;
        actionsHtml = `
          <button onclick="deleteJob('${j.job_id}')" class="btn" style="font-size:0.75rem;padding:0.25rem 0.5rem;border-color:#f87171;color:#f87171">Delete</button>
        `;
      }
      const frameInfo = (j.status === 'processing' && j.total_frames > 0)
        ? `<div style="font-size:0.75rem;color:#94a3b8;margin-top:0.25rem">${j.current_frame} / ${j.total_frames} frames</div>`
        : '';
      const bar = j.status === 'processing'
        ? `<div class="progress-bar" style="height:4px"><div class="progress-fill" style="width:${Math.round(j.progress*100)}%"></div></div>${frameInfo}`
        : '';
      return `<tr class="${selectedJobId === j.job_id ? 'selected-row' : ''}" onclick="if (event.target.tagName!=='BUTTON') viewJob('${j.job_id}')" style="cursor:pointer">
        <td>${j.filename || j.job_id.slice(0,8)}</td>
        <td>${statusHtml}</td>
        <td>${bar || (j.status === 'done' ? '100%' : j.status === 'failed' ? '—' : '0%')}</td>
        <td>${j.duration_sec ? j.duration_sec.toFixed(1) + 's' : '—'}</td>
        <td>${j.tracked_players || '—'}</td>
        <td onclick="event.stopPropagation()">${actionsHtml}</td>
      </tr>`;
    }).join('');
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:#f87171">Error loading jobs: ' + err.message + '</td></tr>';
  }
}

async function viewJob(jobId) {
  selectedJobId = jobId;
  const section = document.getElementById('batch-detail-section');
  section.style.display = 'block';
  document.getElementById('batch-detail-title').textContent = 'Job: ' + jobId.slice(0, 8) + '…';
  const video = document.getElementById('batch-result-video');
  video.style.display = 'none';
  video.src = '';
  video.load();

  const statusRes = await fetch('/batch/status/' + jobId);
  const statusData = await statusRes.json();

  if (statusData.status === 'done') {
    const videoUrl = '/batch/video/' + jobId;
    document.getElementById('video-loading').style.display = 'flex';
    video.oncanplay = () => document.getElementById('video-loading').style.display = 'none';
    video.onerror = () => document.getElementById('video-loading').style.display = 'none';
    video.src = videoUrl;
    video.style.display = 'block';
    video.load();
  }

  await loadBatchStats(jobId);
  await refreshJobList();
}

async function deleteJob(jobId) {
  if (!confirm('Delete this job and all its results?')) return;
  try {
    const res = await fetch('/batch/jobs/' + jobId, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    if (selectedJobId === jobId) closeBatchDetail();
    await refreshJobList();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

async function pollJob(jobId) {
  if (refreshInterval) clearInterval(refreshInterval);
  const progressDiv = document.getElementById('batch-progress');
  const poll = async () => {
    const res = await fetch('/batch/status/' + jobId);
    const data = await res.json();
    if (data.status === 'pending' || data.status === 'processing') {
      const pct = data.progress ? (data.progress * 100).toFixed(0) + '%' : '...';
      const frames = (data.current_frame != null && data.total_frames > 0)
        ? `<span style="color:#94a3b8;font-size:0.8125rem;margin-left:0.75rem">frame ${data.current_frame} / ${data.total_frames}</span>`
        : '';
      progressDiv.innerHTML = `<span style="color:#fbbf24">● Processing: ${pct}${frames}</span><div class="progress-bar"><div class="progress-fill" style="width:${data.progress ? (data.progress*100).toFixed(0) : 0}%"></div></div>`;
      setTimeout(poll, 2000);
    } else if (data.status === 'done') {
      progressDiv.innerHTML = '<span style="color:#fbbf24;font-size:1.1rem">✓ Processing Complete!</span>';
      await refreshJobList();
      await viewJob(jobId);
      refreshInterval = setInterval(refreshJobList, 4000);
    } else if (data.status === 'failed') {
      progressDiv.innerHTML = '<span style="color:#f87171">✗ Failed: ' + (data.error || 'Unknown error') + '</span>';
      await refreshJobList();
      refreshInterval = setInterval(refreshJobList, 4000);
    }
  };
  await refreshJobList();
  setTimeout(poll, 2000);
}

async function loadBatchStats(jobId) {
  try {
    const res = await fetch('/batch/stats/' + jobId);
    if (!res.ok) throw new Error('Failed to load stats');
    const stats = await res.json();
    document.getElementById('stat-duration').textContent = stats.duration_sec.toFixed(1) + 's';
    document.getElementById('stat-resolution').textContent = stats.frame_width + 'x' + stats.frame_height;
    document.getElementById('stat-frames').textContent = stats.total_frames;
    document.getElementById('stat-players').textContent = stats.tracked_players;
    document.getElementById('stat-distance').textContent = stats.total_distance.toFixed(1) + 'm';
    document.getElementById('stat-speed').textContent = stats.avg_speed_all.toFixed(2) + ' m/s';
    document.getElementById('stat-touches').textContent = stats.total_touches;
    const poss = stats.team_possession || {};
    document.getElementById('stat-possession').textContent = Object.entries(poss).map(([k, v]) => 'Team ' + (parseInt(k)+1) + ': ' + v + '%').join(' | ') || '—';
    const tbody = document.getElementById('batch-player-body');
    const batchStats = stats.player_stats || [];
    tbody.innerHTML = batchStats.map((s, i) =>
      `<tr class="batch-player-row" data-idx="${i}"><td>${s.tracking_id}</td><td>${s.label}</td><td>${s.team === 0 ? 'Team 1' : s.team === 1 ? 'Team 2' : '—'}</td><td>${s.total_distance.toFixed(1)}</td><td>${s.avg_speed.toFixed(2)}</td><td>${s.top_speed.toFixed(2)}</td><td>${s.touches}</td></tr>`
    ).join('');
    tbody.querySelectorAll('.batch-player-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.idx);
        showPlayerDetail(batchStats[idx], 'batch-detail');
      });
      row.style.cursor = 'pointer';
    });
  } catch (err) {
    console.error('Error loading batch stats:', err);
  }
}
