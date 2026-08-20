const API = {
    async get(path) {
        const res = await fetch(path);
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
    async post(path, data) {
        const res = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
};

let currentProject = null;
let currentTask = null;

function showView(name) {
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    const view = document.getElementById(`view-${name}`);
    if (view) view.classList.add("active");
    document.querySelectorAll(".nav-btn").forEach(btn => {
        if (btn.dataset.view === name) btn.classList.add("active");
    });
}

document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
});

async function loadProjects() {
    const projects = await API.get("/api/projects");
    const container = document.getElementById("projects-list");
    container.innerHTML = projects.map(p => `
        <div class="card" data-project-id="${p.id}">
            <h3>${escapeHtml(p.name)}</h3>
            <p>创建时间: ${new Date(p.created_at * 1000).toLocaleString()}</p>
            <div class="actions">
                <button class="btn-primary" onclick="openProject(${p.id})">打开项目</button>
            </div>
        </div>
    `).join("");
}

async function openProject(projectId) {
    currentProject = projectId;
    document.getElementById("view-projects").classList.add("hidden");
    document.getElementById("tasks-section").classList.remove("hidden");
    await loadTasks(projectId);
}

document.getElementById("back-to-projects").addEventListener("click", () => {
    document.getElementById("tasks-section").classList.add("hidden");
    document.getElementById("view-projects").classList.remove("hidden");
    currentProject = null;
});

async function loadTasks(projectId) {
    const tasks = await API.get(`/api/projects/${projectId}/tasks`);
    const container = document.getElementById("tasks-list");
    container.innerHTML = tasks.map(t => `
        <div class="card">
            <h3>任务: ${escapeHtml(t.task_id)}</h3>
            <p>状态: <span class="status-badge status-${t.status}">${t.status}</span></p>
            <div class="actions">
                ${t.status === "pending" ? `<button class="btn-primary" onclick="syncTask(${t.id})">同步视频</button>` : ""}
                ${t.status === "ready" ? `<button class="btn-primary" onclick="openAudit(${t.id})">审核</button>` : ""}
            </div>
        </div>
    `).join("");
}

async function syncTask(taskId) {
    await API.post(`/api/tasks/${taskId}/sync`, {});
    await loadTasks(currentProject);
    alert("视频同步完成");
}

document.getElementById("create-project-btn").addEventListener("click", () => {
    document.getElementById("create-project-modal").classList.remove("hidden");
});

document.getElementById("cancel-create-project").addEventListener("click", () => {
    document.getElementById("create-project-modal").classList.add("hidden");
});

document.getElementById("confirm-create-project").addEventListener("click", async () => {
    const name = document.getElementById("project-name-input").value.trim();
    if (!name) return alert("请输入项目名称");
    await API.post("/api/projects", { name });
    document.getElementById("create-project-modal").classList.add("hidden");
    document.getElementById("project-name-input").value = "";
    await loadProjects();
});

document.getElementById("add-tasks-btn").addEventListener("click", async () => {
    const input = document.getElementById("task-ids-input").value.trim();
    if (!input) return alert("请输入任务ID");
    const taskIds = input.split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
    if (taskIds.length === 0) return alert("请输入有效的任务ID");
    await API.post(`/api/projects/${currentProject}/tasks`, { task_ids: taskIds });
    document.getElementById("task-ids-input").value = "";
    await loadTasks(currentProject);
});

async function openAudit(taskId) {
    currentTask = taskId;
    const task = await API.get(`/api/tasks/${taskId}`);
    document.getElementById("audit-task-title").textContent = `审核任务: ${task.task_id}`;
    document.getElementById("audit-video").src = `/api/tasks/${taskId}/video`;
    showView("audit");
    await loadReviews(taskId);
}

document.getElementById("back-to-tasks").addEventListener("click", () => {
    showView("projects");
    document.getElementById("tasks-section").classList.remove("hidden");
});

document.getElementById("mark-missed").addEventListener("click", () => markReview("missed"));
document.getElementById("mark-false").addEventListener("click", () => markReview("false_positive"));

async function markReview(type) {
    const video = document.getElementById("audit-video");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    const formData = new FormData();
    formData.append("screenshot", blob, "screenshot.png");

    const screenshotRes = await fetch(`/api/tasks/${currentTask}/screenshot`, {
        method: "POST",
        body: formData,
    });
    const screenshotData = await screenshotRes.json();

    await API.post(`/api/tasks/${currentTask}/reviews`, {
        review_type: type,
        frame_number: Math.floor(video.currentTime * 30),
        timestamp: video.currentTime,
        note: `${type === "missed" ? "漏检" : "误检"} - ${video.currentTime.toFixed(2)}s`,
        screenshot_path: screenshotData.path,
    });

    await loadReviews(currentTask);
}

async function loadReviews(taskId) {
    const reviews = await API.get(`/api/tasks/${taskId}/reviews`);
    const container = document.getElementById("reviews-list");
    container.innerHTML = reviews.map(r => `
        <div class="review-item">
            <div>
                <div class="review-type ${r.review_type}">
                    ${r.review_type === "missed" ? "漏检" : "误检"}
                </div>
                <div class="review-time">${r.note || ""} @ ${r.timestamp.toFixed(2)}s</div>
                ${r.screenshot_path ? `<img class="review-screenshot" src="${r.screenshot_path}" alt="截图">` : ""}
            </div>
        </div>
    `).join("");
}

async function loadStats() {
    const stats = await API.get("/api/stats");
    document.getElementById("stat-total").textContent = stats.total;
    document.getElementById("stat-missed").textContent = stats.missed;
    document.getElementById("stat-false").textContent = stats.false_positive;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener("DOMContentLoaded", () => {
    loadProjects();
    loadStats();
});
