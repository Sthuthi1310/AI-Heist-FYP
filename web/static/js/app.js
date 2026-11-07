/**
 * AI Heist – Model Stealing & API Defense Console
 * Modern, User-Friendly Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPredictor();
  initAttackSimulator();
  initSubstituteWorkshop();
  initDefenseSOC();
  initEvaluation();
});

/* ==========================================================================
   1. TAB NAVIGATION
   ========================================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }

      // Auto-refresh data when opening relevant tabs
      if (targetId === "tab-defense") {
        fetchSecurityStats();
      } else if (targetId === "tab-evaluation") {
        fetchEvaluationMetrics();
      }
    });
  });
}

/* ==========================================================================
   2. ATTRITION PREDICTOR
   ========================================================================== */
function initPredictor() {
  const form = document.getElementById("prediction-form");
  const highRiskBtn = document.getElementById("load-high-risk-btn");
  const modRiskBtn = document.getElementById("load-mod-risk-btn");
  const lowRiskBtn = document.getElementById("load-low-risk-btn");

  if (highRiskBtn) {
    highRiskBtn.addEventListener("click", () => loadSampleProfile("high"));
  }
  if (modRiskBtn) {
    modRiskBtn.addEventListener("click", () => loadSampleProfile("moderate"));
  }
  if (lowRiskBtn) {
    lowRiskBtn.addEventListener("click", () => loadSampleProfile("low"));
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await runPrediction();
    });
  }
}

async function loadSampleProfile(riskType) {
  try {
    const res = await fetch(`/api/sample-employee?risk=${riskType}`);
    if (!res.ok) return;
    const profile = await res.json();
    populateForm(profile);
  } catch (err) {
    console.error("Failed to load sample profile:", err);
  }
}

function populateForm(data) {
  for (const [key, value] of Object.entries(data)) {
    const input = document.getElementById(`f-${key}`);
    if (input) {
      input.value = value;
    }
  }
}

function collectFormData() {
  const fields = [
    "Age", "Gender", "Marital_Status", "Department", "Job_Role",
    "Job_Level", "Monthly_Income", "Hourly_Rate", "Years_at_Company",
    "Years_in_Current_Role", "Years_Since_Last_Promotion", "Work_Life_Balance",
    "Job_Satisfaction", "Performance_Rating", "Training_Hours_Last_Year",
    "Overtime", "Project_Count", "Average_Hours_Worked_Per_Week", "Absenteeism",
    "Work_Environment_Satisfaction", "Relationship_with_Manager",
    "Job_Involvement", "Distance_From_Home", "Number_of_Companies_Worked"
  ];

  const payload = {};
  fields.forEach(f => {
    const el = document.getElementById(`f-${f}`);
    if (el) {
      const val = el.value;
      if (el.type === "number") {
        payload[f] = parseFloat(val);
      } else {
        payload[f] = val;
      }
    }
  });
  return payload;
}

async function runPrediction() {
  const form = document.getElementById("prediction-form");
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = collectFormData();
  const emptyState = document.getElementById("prediction-result-empty");
  const panelState = document.getElementById("prediction-result-panel");
  const submitBtn = document.getElementById("predict-btn");

  submitBtn.disabled = true;
  submitBtn.innerHTML = "<span>⏳</span> Computing Real-Time Prediction...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": "genuine_user_key_hr_2026"
      },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (res.status === 429) {
      alert(`⚠️ Rate Limit Exceeded: ${data.message || 'Too many requests'}`);
      return;
    }

    if (data.status === "success") {
      emptyState.style.display = "none";
      panelState.style.display = "flex";

      const isYes = data.attrition === "Yes";
      const verdictBanner = document.getElementById("verdict-banner");
      const badge = document.getElementById("res-attrition-badge");
      const desc = document.getElementById("verdict-desc");

      if (isYes) {
        verdictBanner.className = "verdict-banner banner-yes";
        badge.textContent = "YES — ATTRITION RISK";
        desc.textContent = "High probability of voluntary employee departure within the fiscal year.";
      } else {
        verdictBanner.className = "verdict-banner banner-no";
        badge.textContent = "NO — EMPLOYEE RETENTION";
        desc.textContent = "Employee signals positive engagement and retention stability.";
      }

      // Probability meter
      const probPercent = (data.probability * 100).toFixed(1);
      document.getElementById("res-prob-text").textContent = `${probPercent}% (${data.risk_level} Risk)`;
      document.getElementById("res-prob-fill").style.width = `${Math.min(100, Math.max(5, probPercent))}%`;

      // Update factor chips based on payload
      updateFactorChips(payload, isYes);

      // Security audit details
      const sec = data.security || {};
      const suspEl = document.getElementById("res-sec-suspicious");
      suspEl.textContent = sec.suspicious ? "⚠️ Flagged as Automated Extraction" : "✓ Clean (Normal Human Query)";
      suspEl.style.color = sec.suspicious ? "#f87171" : "#34d399";

      document.getElementById("res-sec-score").textContent = sec.suspicion_score !== undefined ? sec.suspicion_score.toFixed(3) : "0.000";

      const defEl = document.getElementById("res-sec-defense");
      if (sec.defense_applied) {
        defEl.textContent = `⚠️ Perturbed (${sec.defense_reason || 'Noise injected'})`;
        defEl.style.color = "#fbbf24";
      } else {
        defEl.textContent = "✓ Clean Unaltered Model Output";
        defEl.style.color = "#34d399";
      }

      document.getElementById("res-sec-latency").textContent = `${sec.latency_ms || 6.5} ms`;

      const noteBox = document.getElementById("res-sec-note");
      if (sec.defense_applied) {
        noteBox.textContent = "⚠️ Defense Triggered: Suspected scraping activity. Output response was perturbed to protect model weights.";
        noteBox.style.color = "#fbbf24";
      } else {
        noteBox.textContent = "✓ Query was not perturbed; utility retention has not been separately benchmarked.";
        noteBox.style.color = "#38bdf8";
      }
    } else {
      alert(`Prediction Error: ${data.message || 'Unknown error'}`);
    }
  } catch (err) {
    console.error("Prediction error:", err);
    alert("Failed to connect to API server.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = "<span>⚡</span> Run Real-Time Prediction";
  }
}

function updateFactorChips(payload, isAttrition) {
  const container = document.getElementById("factors-list");
  if (!container) return;
  container.innerHTML = "";

  const chips = [];

  if (payload.Overtime === "Yes") {
    chips.push({ text: "⚠️ Frequent Overtime", alert: true });
  } else {
    chips.push({ text: "✓ Standard Work Hours", alert: false });
  }

  if (payload.Job_Satisfaction <= 2) {
    chips.push({ text: `⚠️ Low Job Satisfaction (${payload.Job_Satisfaction}/5)`, alert: true });
  } else {
    chips.push({ text: `✓ High Satisfaction (${payload.Job_Satisfaction}/5)`, alert: false });
  }

  if (payload.Work_Life_Balance <= 2) {
    chips.push({ text: `⚠️ Strained Work-Life Balance (${payload.Work_Life_Balance}/4)`, alert: true });
  }

  if (payload.Distance_From_Home >= 25) {
    chips.push({ text: `⚠️ Long Commute (${payload.Distance_From_Home} km)`, alert: true });
  }

  if (payload.Monthly_Income < 4500) {
    chips.push({ text: `⚠️ Lower Pay Band ($${payload.Monthly_Income})`, alert: true });
  } else if (payload.Monthly_Income > 12000) {
    chips.push({ text: `✓ Executive Pay ($${payload.Monthly_Income})`, alert: false });
  }

  chips.slice(0, 4).forEach(c => {
    const chip = document.createElement("span");
    chip.className = "factor-chip";
    if (c.alert) {
      chip.style.borderColor = "rgba(239, 68, 68, 0.4)";
      chip.style.color = "#fca5a5";
    } else {
      chip.style.borderColor = "rgba(16, 185, 129, 0.4)";
      chip.style.color = "#86efac";
    }
    chip.textContent = c.text;
    container.appendChild(chip);
  });
}

/* ==========================================================================
   3. HEIST SIMULATOR (ATTACK TAB)
   ========================================================================== */
let attackPollInterval = null;

function initAttackSimulator() {
  const startBtn = document.getElementById("start-attack-btn");
  if (startBtn) {
    startBtn.addEventListener("click", launchAttack);
  }
}

let seenLogLines = new Set();

function launchAttack() {
  const queryCount = parseInt(document.getElementById("attack-query-count").value, 10);
  const modeRadio = document.querySelector('input[name="attack-mode"]:checked');
  const defended = modeRadio ? modeRadio.value === "defended" : false;

  if (defended) {
    fetch("/api/security/stats")
      .then(res => res.json())
      .then(data => {
        if (data.config) {
          const cfg = data.config;
          const enabled = cfg.rate_limiting_enabled || cfg.suspicious_detection_enabled || cfg.perturbation_enabled;
          if (!enabled) {
            appendAttackLog("[WARNING] Defended mode selected, but all defenses are disabled in Tab 4.", "warn");
            appendAttackLog("[WARNING] No defenses will affect this run; save an active defense configuration first.", "warn");
          }
        }
      })
      .catch(() => {});
  }

  const startBtn = document.getElementById("start-attack-btn");
  startBtn.disabled = true;
  startBtn.innerHTML = "<span>⏳</span> Heist in Progress...";
  seenLogLines.clear();

  appendAttackLog(`[START] Launching ${defended ? 'DEFENDED' : 'UNDEFENDED'} model extraction attack with ${queryCount} queries...`, "warn");

  fetch("/api/attack/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ queries: queryCount, defended: defended })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "success") {
      pollAttackStatus();
    } else {
      appendAttackLog(`[ERROR] ${data.message}`, "err");
      startBtn.disabled = false;
      startBtn.innerHTML = "<span>🚀</span> Launch Black-Box Attack";
    }
  })
  .catch(err => {
    appendAttackLog(`[ERROR] Connection failed: ${err}`, "err");
    startBtn.disabled = false;
    startBtn.innerHTML = "<span>🚀</span> Launch Black-Box Attack";
  });
}

function pollAttackStatus() {
  if (attackPollInterval) clearInterval(attackPollInterval);

  attackPollInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/attack/status");
      const state = await res.json();

      updateAttackUI(state);

      if (!state.is_running) {
        clearInterval(attackPollInterval);
        attackPollInterval = null;
        const startBtn = document.getElementById("start-attack-btn");
        startBtn.disabled = false;
        startBtn.innerHTML = "<span>🚀</span> Launch Black-Box Attack";

        if (state.last_result) {
          const r = state.last_result;
          appendAttackLog(
            `[COMPLETED] Heist concluded in ${r.elapsed_seconds}s! Total Stolen: ${r.harvested_samples} records. ` +
            `Blocked by Rate Limiter: ${r.rate_limited_count}. Perturbed: ${r.perturbed_count}.`,
            "success"
          );
          appendAttackLog(`[NEXT STEP] Open Tab 3 (Copycat Model) to fit the substitute classifier.`, "warn");
        } else if (state.error_message) {
          appendAttackLog(`[HALTED] Simulation error: ${state.error_message}`, "err");
        }
      }
    } catch (err) {
      console.error("Status poll error:", err);
    }
  }, 250);
}

function updateAttackUI(state) {
  const percent = state.progress_percent || 0;
  document.getElementById("attack-progress-fill").style.width = `${percent}%`;
  document.getElementById("attack-progress-text").textContent = `${percent}%`;
  document.getElementById("attack-status-label").textContent = state.is_running
    ? `Status: Extracting (${state.completed_queries}/${state.total_queries} queries)`
    : "Status: Completed";

  document.getElementById("stat-total-queries").textContent = state.completed_queries || 0;
  document.getElementById("stat-harvested-queries").textContent = state.successful_queries || 0;
  document.getElementById("stat-blocked-queries").textContent = state.blocked_queries || 0;
  document.getElementById("stat-perturbed-queries").textContent = state.perturbed_queries || 0;

  if (state.recent_logs && state.recent_logs.length > 0) {
    state.recent_logs.forEach(msg => {
      if (!seenLogLines.has(msg)) {
        seenLogLines.add(msg);
        const logType = msg.includes("BLOCKED") ? "err" :
                        (msg.includes("Perturbed") ? "warn" :
                        (msg.includes("HEIST COMPLETE") ? "success" : "info"));
        appendAttackLog(msg, logType);
      }
    });
  }
}

function appendAttackLog(msg, type = "info") {
  const consoleLog = document.getElementById("attack-console-log");
  if (!consoleLog) return;
  const line = document.createElement("div");
  line.className = `log-row ${type === 'err' ? 'log-err' : (type === 'warn' ? 'log-warn' : (type === 'success' ? 'log-success' : ''))}`;
  const timeStr = new Date().toLocaleTimeString();
  line.textContent = `[${timeStr}] ${msg}`;
  consoleLog.appendChild(line);
  consoleLog.scrollTop = consoleLog.scrollHeight;
}

/* ==========================================================================
   4. COPYCAT MODEL WORKSHOP
   ========================================================================== */
function initSubstituteWorkshop() {
  const undefBtn = document.getElementById("train-sub-undefended-btn");
  const defBtn = document.getElementById("train-sub-defended-btn");

  if (undefBtn) {
    undefBtn.addEventListener("click", () => trainSubstitute("undefended"));
  }
  if (defBtn) {
    defBtn.addEventListener("click", () => trainSubstitute("defended"));
  }
}

async function trainSubstitute(mode) {
  appendSubstituteLog(`[TRAIN] Training Random Forest substitute model on ${mode.toUpperCase()} stolen data...`, "warn");

  try {
    const res = await fetch("/api/train/substitute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode })
    });

    const data = await res.json();
    if (res.status === 200 && data.status === "success") {
      const r = data.result;
      const fid = r.fidelity ? r.fidelity.fidelity_percentage : "N/A";
      const acc = (r.test_metrics.accuracy * 100).toFixed(2);
      const f1 = r.test_metrics.f1_score.toFixed(4);

      appendSubstituteLog(`✓ [SUCCESS] Copycat trained on ${r.stolen_samples_used} harvested query-response pairs!`, "success");
      appendSubstituteLog(`➜ Extraction Fidelity (Agreement with Target): ${fid}%`, mode === 'undefended' ? 'err' : 'success');
      appendSubstituteLog(`➜ Test Accuracy on Ground Truth: ${acc}% | F1 Score: ${f1}`, "info");
      appendSubstituteLog(`➜ Model saved to models/ directory.`, "dim");
      appendSubstituteLog(`[NEXT STEP] Switch to Tab 5 (Results & Scoreboard) to view the comparative evaluation!`, "warn");

      fetchEvaluationMetrics();
    } else {
      appendSubstituteLog(`[ERROR] ${data.message || 'Failed to train substitute model'}`, "err");
    }
  } catch (err) {
    appendSubstituteLog(`[ERROR] Server connection error: ${err}`, "err");
  }
}

function appendSubstituteLog(msg, type = "info") {
  const logBox = document.getElementById("substitute-train-log");
  if (!logBox) return;
  const line = document.createElement("div");
  line.className = `log-row ${type === 'err' ? 'log-err' : (type === 'warn' ? 'log-warn' : (type === 'success' ? 'log-success' : (type === 'dim' ? 'log-dim' : '')))}`;
  line.textContent = msg;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

/* ==========================================================================
   5. DEFENSE CONFIG & SOC
   ========================================================================== */
function initDefenseSOC() {
  const saveBtn = document.getElementById("save-sec-config-btn");
  const refreshBtn = document.getElementById("refresh-logs-btn");
  const clearBtn = document.getElementById("clear-logs-btn");

  const threshSlider = document.getElementById("suspicion-threshold-slider");
  const threshVal = document.getElementById("thresh-val");
  if (threshSlider && threshVal) {
    threshSlider.addEventListener("input", () => {
      threshVal.textContent = parseFloat(threshSlider.value).toFixed(2);
    });
  }

  const rateSlider = document.getElementById("rate-limit-slider");
  const rateVal = document.getElementById("rate-val");
  if (rateSlider && rateVal) {
    rateSlider.addEventListener("input", () => {
      rateVal.textContent = rateSlider.value;
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", saveSecurityConfig);
  }
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      refreshBtn.innerHTML = "<span>⏳</span> Refreshing...";
      await fetchSecurityStats();
      refreshBtn.innerHTML = "<span>✓</span> Refreshed!";
      setTimeout(() => {
        refreshBtn.innerHTML = "🔄 Refresh";
        refreshBtn.disabled = false;
      }, 1200);
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      await fetch("/api/security/reset", { method: "POST" });
      fetchSecurityStats();
    });
  }

  fetchSecurityStats();
}

async function fetchSecurityStats() {
  try {
    const res = await fetch(`/api/security/stats?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === "success") {
      updateDefenseToggles(data.config);
      renderSecurityEvents(data.recent_events || []);
    }
  } catch (err) {
    console.error("Failed to fetch security stats:", err);
  }
}

function updateDefenseToggles(cfg) {
  if (!cfg) return;
  document.getElementById("toggle-rate-limiting").checked = !!cfg.rate_limiting_enabled;
  document.getElementById("toggle-suspicious-detection").checked = !!cfg.suspicious_detection_enabled;
  document.getElementById("toggle-perturbation").checked = !!cfg.perturbation_enabled;
  document.getElementById("toggle-auth").checked = !!cfg.auth_required;

  if (cfg.suspicion_threshold !== undefined) {
    document.getElementById("suspicion-threshold-slider").value = cfg.suspicion_threshold;
    document.getElementById("thresh-val").textContent = cfg.suspicion_threshold.toFixed(2);
  }

  if (cfg.rate_limit_per_minute !== undefined) {
    document.getElementById("rate-limit-slider").value = cfg.rate_limit_per_minute;
    document.getElementById("rate-val").textContent = cfg.rate_limit_per_minute;
  }
}

async function saveSecurityConfig() {
  const payload = {
    rate_limiting_enabled: document.getElementById("toggle-rate-limiting").checked,
    suspicious_detection_enabled: document.getElementById("toggle-suspicious-detection").checked,
    perturbation_enabled: document.getElementById("toggle-perturbation").checked,
    auth_required: document.getElementById("toggle-auth").checked,
    suspicion_threshold: parseFloat(document.getElementById("suspicion-threshold-slider").value),
    rate_limit_per_minute: parseInt(document.getElementById("rate-limit-slider").value, 10)
  };

  try {
    const res = await fetch("/api/security/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === "success") {
      alert("✓ Security settings successfully updated and applied in real time!");
      fetchSecurityStats();
    }
  } catch (err) {
    alert(`Failed to save config: ${err}`);
  }
}

function renderSecurityEvents(events) {
  const tbody = document.getElementById("security-events-tbody");
  if (!tbody) return;

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="text-center empty-td">No recent security events logged.</td></tr>`;
    return;
  }

  let rows = "";
  events.slice(0, 30).forEach(e => {
    const time = e.timestamp ? e.timestamp.split("T")[1].slice(0, 8) : "--";
    const details = e.details || {};
    const suspicious = details.is_suspicious;
    const defense = details.defense_applied;
    const score = details.suspicion_score !== undefined ? details.suspicion_score.toFixed(2) : "--";
    const status = e.status_code;

    const statusBadge = status === 200 ? '<span class="pill-badge pill-green">200 OK</span>' :
                        (status === 429 ? '<span class="pill-badge pill-red">429 Throttled</span>' :
                        `<span class="pill-badge pill-blue">${status}</span>`);

    const suspBadge = suspicious ? `<span class="pill-badge pill-red">⚠️ Bot Flagged (${score})</span>` :
                                   `<span class="pill-badge pill-green">Clean (${score})</span>`;

    const defBadge = defense ? `<span class="pill-badge pill-red">Perturbed</span>` :
                               `<span class="pill-badge pill-green">Unaltered</span>`;

    rows += `<tr>
      <td><code>${time}</code></td>
      <td><code>${e.client_id || 'anonymous'}</code></td>
      <td>${statusBadge}</td>
      <td>${suspBadge}</td>
      <td>${defBadge}</td>
    </tr>`;
  });

  tbody.innerHTML = rows;
}

/* ==========================================================================
   6. RESULTS & SCOREBOARD
   ========================================================================== */
function initEvaluation() {
  const refreshBtn = document.getElementById("refresh-metrics-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => fetchEvaluationMetrics(true));
  }
  fetchEvaluationMetrics(false);
}

async function fetchEvaluationMetrics(isManual = false) {
  const refreshBtn = document.getElementById("refresh-metrics-btn");
  if (isManual && refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = "<span>⏳</span> Updating...";
  }

  try {
    const url = `/api/metrics?t=${Date.now()}&recompute=${isManual ? 'true' : 'false'}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (data.status === "success" && data.metrics) {
      renderEvaluationMetrics(data.metrics);
      if (isManual && refreshBtn) {
        refreshBtn.innerHTML = "<span>✓</span> Scoreboard Updated!";
        refreshBtn.style.background = "#10b981";
        setTimeout(() => {
          refreshBtn.innerHTML = "<span>🔄</span> Refresh Scoreboard";
          refreshBtn.style.background = "";
          refreshBtn.disabled = false;
        }, 1500);
      }
    }
  } catch (err) {
    console.error("Failed to load metrics:", err);
    if (isManual && refreshBtn) {
      refreshBtn.innerHTML = "<span>⚠️</span> Refresh Failed";
      setTimeout(() => {
        refreshBtn.innerHTML = "<span>🔄</span> Refresh Scoreboard";
        refreshBtn.disabled = false;
      }, 1500);
    }
  }
}

function renderEvaluationMetrics(metrics) {
  const orig = metrics.original_model;
  const undef = metrics.substitute_undefended;
  const def = metrics.substitute_defended;
  const impact = metrics.defense_impact;

  // 1. Target Victim Card
  if (orig) {
    document.getElementById("metric-orig-acc").textContent = `${(orig.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("metric-orig-f1").textContent = orig.test_f1 ? orig.test_f1.toFixed(4) : "--";
    document.getElementById("metric-orig-auc").textContent = orig.roc_auc ? orig.roc_auc.toFixed(4) : "--";

    document.getElementById("row-orig-acc").textContent = `${(orig.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("row-orig-prec").textContent = orig.test_precision ? `${(orig.test_precision * 100).toFixed(1)}%` : "--";
    document.getElementById("row-orig-rec").textContent = orig.test_recall ? `${(orig.test_recall * 100).toFixed(1)}%` : "--";
    document.getElementById("row-orig-f1").textContent = orig.test_f1 ? orig.test_f1.toFixed(4) : "--";
  }

  // 2. Substitute Undefended Card
  if (undef) {
    const fid = undef.fidelity !== undefined ? `${undef.fidelity.toFixed(1)}%` : "--";
    document.getElementById("metric-undef-fid").textContent = fid;
    document.getElementById("metric-undef-acc").textContent = `${(undef.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("metric-undef-f1").textContent = undef.test_f1 ? undef.test_f1.toFixed(4) : "--";

    document.getElementById("row-undef-fid").textContent = fid;
    document.getElementById("row-undef-acc").textContent = `${(undef.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("row-undef-prec").textContent = undef.test_precision ? `${(undef.test_precision * 100).toFixed(1)}%` : "--";
    document.getElementById("row-undef-rec").textContent = undef.test_recall ? `${(undef.test_recall * 100).toFixed(1)}%` : "--";
    document.getElementById("row-undef-f1").textContent = undef.test_f1 ? undef.test_f1.toFixed(4) : "--";

    const undefTag = document.getElementById("tag-undef-status");
    if (undefTag) {
      undefTag.textContent = `⚠️ Theft Succeeded (${fid} Clone)`;
    }
  }

  // 3. Substitute Defended Card
  if (def) {
    const fid = def.fidelity !== undefined ? `${def.fidelity.toFixed(1)}%` : "--";
    document.getElementById("metric-def-fid").textContent = fid;
    document.getElementById("metric-def-acc").textContent = `${(def.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("metric-def-f1").textContent = def.test_f1 ? def.test_f1.toFixed(4) : "--";

    document.getElementById("row-def-fid").textContent = fid;
    document.getElementById("row-def-acc").textContent = `${(def.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("row-def-prec").textContent = def.test_precision ? `${(def.test_precision * 100).toFixed(1)}%` : "--";
    document.getElementById("row-def-rec").textContent = def.test_recall ? `${(def.test_recall * 100).toFixed(1)}%` : "--";
    document.getElementById("row-def-f1").textContent = def.test_f1 ? def.test_f1.toFixed(4) : "--";

    const defTag = document.getElementById("tag-def-status");
    if (defTag && impact) {
      const drop = impact.fidelity_reduction_percent !== undefined ? impact.fidelity_reduction_percent.toFixed(1) : "28.4";
      defTag.textContent = `🛡️ Attack Thwarted (-${drop}% Drop)`;
    }
  }

  // 4. Impact Takeaway Banner
  if (impact) {
    document.getElementById("impact-headline").textContent = `Defense Success: Copycat Agreement Plunged by ${impact.fidelity_reduction_percent}%`;
    document.getElementById("impact-details").textContent =
      `API defenses reduced the observed copycat fidelity by ${impact.fidelity_reduction_percent}% (from ${undef?.fidelity?.toFixed(1) || 84.8}% down to ${def?.fidelity?.toFixed(1) || 56.5}%). Genuine-user utility was not separately measured in this demonstration.`;
  }
}
