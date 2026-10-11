/** @typedef {"healthy"|"degraded"|"unavailable"|"not_configured"|"configured"|"disabled"|"enabled"|"unknown"} ServiceStatus */
/** @typedef {{name: string, status: ServiceStatus, detail?: string}} DashboardService */
/** @typedef {{bot_id: string, meeting_url?: string|null, created_at?: string|null, bot_state?: string|null, transcription_state?: string|null, status?: string|null, task_count: number, error?: string|null}} RecentMeeting */
/** @typedef {{event_id: string, event_title: string, event_start: string, event_end?: string|null, meeting_url: string, join_status: string, bot_id?: string|null, error?: string|null}} ScheduledCalendarJoin */
/** @typedef {{enabled: boolean, event_title_filter?: string|null, time_zone?: string, upcoming_events: ScheduledCalendarJoin[]}} CalendarDashboardData */
/** @typedef {{status: "healthy"|"degraded", updated_at: string, services: DashboardService[], recent_meetings: RecentMeeting[], calendar: CalendarDashboardData}} DashboardOverview */
/** @typedef {{key: string, label: string, group: string, type: "secret"|"url"|"integer"|"number"|"boolean"|"json"|"text", help?: string}} ConfigField */
/** @typedef {{fields: ConfigField[], values: Record<string, string>, secret_configured: Record<string, boolean>}} DashboardConfig */
/** @typedef {{saved: boolean, restart_required: boolean, api_token_changed: boolean}} ConfigSaveResult */
/** @typedef {{values: Record<string, string|boolean>, clear: string[]}} ConfigUpdate */

const TOKEN_KEY = "meeting-agent-dashboard-token";
let authToken = sessionStorage.getItem(TOKEN_KEY) || "";
/** @type {ConfigField[]} */
let configFields = [];
/** @type {number|null} */
let refreshTimer = null;

/** @param {string} id @returns {HTMLElement} */
const element = (id) => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Dashboard element not found: ${id}`);
  return node;
};

/** @param {string} message @param {boolean} [isWarning] */
function showNotice(message, isWarning = false) {
  const notice = element("notice");
  notice.textContent = message;
  notice.classList.toggle("warning", isWarning);
  notice.classList.remove("hidden");
  window.setTimeout(() => notice.classList.add("hidden"), 7000);
}

/** @param {boolean} locked */
function setLocked(locked) {
  element("auth-gate").classList.toggle("hidden", !locked);
  if (locked) {
    window.clearInterval(refreshTimer);
    authToken = "";
    sessionStorage.removeItem(TOKEN_KEY);
    element("auth-token").value = "";
    element("auth-token").focus();
  }
}

/**
 * @template T
 * @param {string} path
 * @param {RequestInit} [options]
 * @returns {Promise<T>}
 */
async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${authToken}`);
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  if (response.status === 401) {
    setLocked(true);
    throw new Error("Session expired. Enter APP_API_TOKEN to continue.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

/** @param {string|undefined|null} status @returns {string} */
function statusClass(status) {
  return String(status || "unknown").toLowerCase().replaceAll(" ", "_");
}

/** @param {DashboardService[]|undefined} services */
function renderServices(services) {
  const grid = element("service-grid");
  grid.replaceChildren();
  for (const service of services || []) {
    const card = document.createElement("article");
    card.className = "service-card";
    const head = document.createElement("div");
    head.className = "service-card-head";
    const title = document.createElement("h3");
    title.textContent = service.name || "Service";
    const state = document.createElement("span");
    state.className = `state state-${statusClass(service.status)}`;
    state.textContent = String(service.status || "unknown").replaceAll("_", " ");
    head.append(title, state);
    const description = document.createElement("p");
    description.textContent = service.detail || "";
    card.append(head, description);
    grid.append(card);
  }
}

/** @param {CalendarDashboardData|undefined|null} calendar */
function renderCalendarEvents(calendar) {
  const body = element("calendar-body");
  body.replaceChildren();
  const events = calendar?.upcoming_events || [];
  element("calendar-summary").textContent = calendar?.enabled
    ? `${calendar.event_title_filter || "All eligible events"} · ${calendar.time_zone || "calendar time zone"} · next 14 days`
    : "Calendar auto-join is disabled";
  if (!calendar?.enabled || events.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "empty";
    cell.textContent = calendar?.enabled
      ? "No eligible calendar events are currently scheduled."
      : "Enable Google Calendar auto-join in Configuration to view scheduled joins.";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const event of events) {
    const row = document.createElement("tr");
    const title = document.createElement("td");
    title.textContent = event.event_title || "Google Meet event";
    const start = document.createElement("td");
    start.textContent = event.event_start
      ? new Date(event.event_start).toLocaleString()
      : "—";
    const status = document.createElement("td");
    status.textContent = String(event.join_status || "scheduled").replaceAll("_", " ");
    const agent = document.createElement("td");
    agent.textContent = event.bot_id || "Not requested yet";
    const link = document.createElement("td");
    if (typeof event.meeting_url === "string") {
      const parsed = new URL(event.meeting_url);
      if (
        parsed.protocol === "https:" &&
        ["meet.google.com", "www.meet.google.com"].includes(parsed.hostname)
      ) {
        const anchor = document.createElement("a");
        anchor.href = parsed.href;
        anchor.textContent = "Open Meet";
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        link.append(anchor);
      }
    }
    if (!link.firstChild) link.textContent = "—";
    row.append(title, start, status, agent, link);
    body.append(row);
  }
}

/** @param {RecentMeeting[]|undefined} meetings */
function renderMeetings(meetings) {
  const body = element("meetings-body");
  body.replaceChildren();
  element("meeting-count").textContent = `${(meetings || []).length} shown`;
  if (!meetings || meetings.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty";
    cell.textContent = "No meetings recorded yet.";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const meeting of meetings) {
    const row = document.createElement("tr");
    const identity = document.createElement("td");
    const id = document.createElement("div");
    id.className = "meeting-id";
    id.textContent = meeting.bot_id || "Unknown bot";
    identity.append(id);
    if (meeting.meeting_url) {
      const url = document.createElement("span");
      url.className = "meeting-url";
      url.textContent = meeting.meeting_url;
      identity.append(url);
    }
    const botState = document.createElement("td");
    botState.textContent = meeting.bot_state || "unknown";
    const pipeline = document.createElement("td");
    pipeline.textContent = meeting.status || "unknown";
    const tasks = document.createElement("td");
    tasks.textContent = String(meeting.task_count ?? 0);
    const created = document.createElement("td");
    created.textContent = meeting.created_at
      ? new Date(meeting.created_at).toLocaleString()
      : "—";
    const error = document.createElement("td");
    error.textContent = meeting.error ? String(meeting.error).slice(0, 180) : "—";
    row.append(identity, botState, pipeline, tasks, created, error);
    body.append(row);
  }
}

/** @returns {Promise<void>} */
async function loadOverview() {
  const data = /** @type {DashboardOverview} */ (await request("/api/overview"));
  const status = data.status || "unknown";
  const pill = element("overall-status");
  pill.textContent = status;
  pill.className = `status-pill status-${statusClass(status)}`;
  element("updated-at").textContent = data.updated_at
    ? `Updated ${new Date(data.updated_at).toLocaleTimeString()}`
    : "Status unavailable";
  renderServices(data.services);
  renderCalendarEvents(data.calendar);
  renderMeetings(data.recent_meetings);
}

/**
 * @param {ConfigField} field
 * @param {Record<string, string>} values
 * @param {Record<string, boolean>} secretState
 * @returns {HTMLElement}
 */
function createField(field, values, secretState) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.htmlFor = `config-${field.key}`;
  label.append(document.createTextNode(field.label || field.key));
  if (field.type === "secret") {
    const status = document.createElement("span");
    status.className = "secret-status";
    status.textContent = secretState[field.key] ? "(configured)" : "(not set)";
    label.append(status);
  }
  wrapper.append(label);

  let input;
  if (field.type === "json") {
    input = document.createElement("textarea");
    input.spellcheck = false;
  } else {
    input = document.createElement("input");
    input.type = field.type === "secret" ? "password" : field.type === "boolean" ? "checkbox" : field.type === "integer" || field.type === "number" ? "number" : "text";
    if (field.type === "integer") input.step = "1";
    if (field.type === "number") input.step = "any";
  }
  input.id = `config-${field.key}`;
  input.name = field.key;
  input.dataset.type = field.type;
  input.autocomplete = field.type === "secret" ? "new-password" : "off";
  if (field.type === "boolean") {
    input.checked = String(values[field.key] || "false").toLowerCase() === "true";
  } else if (field.type !== "secret") {
    input.value = values[field.key] || "";
  } else {
    input.placeholder = "Leave blank to keep current value";
  }
  wrapper.append(input);

  if (field.help) {
    const help = document.createElement("p");
    help.className = "field-help";
    help.textContent = field.help;
    wrapper.append(help);
  }
  if (field.type === "secret" && secretState[field.key] && field.key !== "APP_API_TOKEN") {
    const clearLabel = document.createElement("label");
    clearLabel.className = "clear-secret";
    const clear = document.createElement("input");
    clear.type = "checkbox";
    clear.name = `${field.key}__clear`;
    clear.dataset.clearKey = field.key;
    clearLabel.append(clear, document.createTextNode("Clear this secret"));
    wrapper.append(clearLabel);
  }
  return wrapper;
}

/** @returns {Promise<void>} */
async function loadConfig() {
  const config = /** @type {DashboardConfig} */ (await request("/api/config"));
  configFields = config.fields || [];
  const container = element("config-fields");
  container.replaceChildren();
  const groups = new Map();
  for (const field of configFields) {
    const groupName = field.group || "Other";
    if (!groups.has(groupName)) {
      const section = document.createElement("section");
      section.className = "config-group";
      const heading = document.createElement("h3");
      heading.textContent = groupName;
      const grid = document.createElement("div");
      grid.className = "config-grid";
      section.append(heading, grid);
      container.append(section);
      groups.set(groupName, grid);
    }
    groups.get(groupName).append(
      createField(field, config.values || {}, config.secret_configured || {})
    );
  }
}

/** @param {SubmitEvent} event @returns {Promise<void>} */
async function saveConfig(event) {
  event.preventDefault();
  const saveButton = element("save-config");
  const result = element("config-result");
  saveButton.disabled = true;
  result.textContent = "Validating and saving…";
  const values = {};
  const clear = [];
  for (const field of configFields) {
    const input = element(`config-${field.key}`);
    if (field.type === "secret") {
      if (input.value) values[field.key] = input.value;
      const clearInput = document.querySelector(`[data-clear-key="${field.key}"]`);
      if (clearInput?.checked) clear.push(field.key);
    } else if (field.type === "boolean") {
      values[field.key] = input.checked;
    } else {
      values[field.key] = input.value;
    }
  }
  try {
    const update = /** @type {ConfigUpdate} */ ({ values, clear });
    const data = /** @type {ConfigSaveResult} */ (
      await request("/api/config", {
        method: "PUT",
        body: JSON.stringify(update),
      })
    );
    if (values.APP_API_TOKEN) {
      authToken = String(values.APP_API_TOKEN);
      sessionStorage.setItem(TOKEN_KEY, authToken);
    }
    element("restart-notice").classList.toggle("hidden", !data.restart_required);
    result.textContent = "Saved. Restart required.";
    await loadConfig();
  } catch (error) {
    result.textContent = error.message;
    result.className = "error";
  } finally {
    saveButton.disabled = false;
  }
}

/** @param {string} token @returns {Promise<void>} */
async function unlock(token) {
  authToken = token;
  await loadConfig();
  sessionStorage.setItem(TOKEN_KEY, token);
  setLocked(false);
  loadOverview().catch((error) => showNotice(error.message, true));
  window.clearInterval(refreshTimer);
  refreshTimer = window.setInterval(() => {
    loadOverview().catch((error) => showNotice(error.message, true));
  }, 30000);
}

element("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorNode = element("auth-error");
  errorNode.classList.add("hidden");
  try {
    await unlock(element("auth-token").value);
  } catch (error) {
    errorNode.textContent = error.message;
    errorNode.classList.remove("hidden");
  }
});

element("refresh-button").addEventListener("click", () => {
  loadOverview().catch((error) => showNotice(error.message, true));
});
element("config-form").addEventListener("submit", saveConfig);
element("lock-button").addEventListener("click", () => setLocked(true));
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) other.classList.remove("active");
    tab.classList.add("active");
    for (const panel of document.querySelectorAll(".panel")) panel.classList.add("hidden");
    element(tab.dataset.panel).classList.remove("hidden");
  });
}

if (authToken) {
  unlock(authToken).catch(() => setLocked(true));
} else {
  setLocked(true);
}
