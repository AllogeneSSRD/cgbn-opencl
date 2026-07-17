"use strict";

const state = {
  rows: [],
  columns: [],
  sortKey: "start_ts",
  sortDir: 1,
  filters: {
    worker: new Set(),
    exponent: new Set(),
    curve: new Set(),
    status: new Set(),
  },
};

const NUMERIC = new Set([
  "exponent", "curve", "b1", "b2", "worth", "avail_mem", "using_mem",
  "s1_time", "s2_init_time", "s2_time", "s2_gcd_time", "s1_fft", "s2_fft",
]);

let ganttChart = null;

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);

function tsToMs(ts) {
  if (!ts) return null;
  return new Date(ts.replace(" ", "T")).getTime();
}

function toast(msg, isErr) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  setTimeout(() => t.classList.add("hidden"), 3000);
}

function fmt(val, key) {
  if (val === null || val === undefined || val === "") return "";
  if (NUMERIC.has(key) && typeof val === "number") {
    return val.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return val;
}

// ---------- upload / parse ----------
$("logfile").addEventListener("change", (e) => {
  const f = e.target.files[0];
  $("filename").textContent = f ? f.name : "未选择文件";
  $("parseBtn").disabled = !f;
});

$("parseBtn").addEventListener("click", async () => {
  const file = $("logfile").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("logfile", file);
  $("parseBtn").disabled = true;
  $("parseBtn").textContent = "解析中…";
  try {
    const res = await fetch("/parse", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "解析失败");
    state.rows = data.rows;
    state.columns = data.columns;
    onData();
    toast(`解析完成：${data.rows.length} 条运行`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    $("parseBtn").disabled = false;
    $("parseBtn").textContent = "解析";
  }
});

// ---------- data ready ----------
function onData() {
  $("empty").classList.add("hidden");
  $("app").classList.remove("hidden");
  buildChips();
  buildHead();
  render();
  window.addEventListener("resize", () => ganttChart && ganttChart.resize());
}

function uniqSorted(key) {
  const vals = [...new Set(state.rows.map((r) => r[key]))];
  vals.sort((a, b) => (typeof a === "number" ? a - b : String(a).localeCompare(String(b))));
  return vals;
}

function buildChips() {
  makeChipGroup("fWorker", uniqSorted("worker"), "worker", (v) => "#" + v);
  makeChipGroup("fExp", uniqSorted("exponent"), "exponent", (v) => "M" + v);
  makeChipGroup("fCurve", uniqSorted("curve"), "curve", (v) => "#" + v);
  makeChipGroup("fStatus", ["complete", "stage1-only", "interrupted"], "status", (v) => v, true);
}

function makeChipGroup(containerId, values, filterKey, label, isStatus) {
  const c = $(containerId);
  c.innerHTML = "";
  values.forEach((v) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (isStatus ? " status-" + v : "");
    chip.textContent = label(v);
    chip.addEventListener("click", () => {
      const set = state.filters[filterKey];
      if (set.has(v)) { set.delete(v); chip.classList.remove("active"); }
      else { set.add(v); chip.classList.add("active"); }
      render();
    });
    c.appendChild(chip);
  });
}

// range + date inputs
["b1Min", "b1Max", "availMin", "availMax", "usingMin", "usingMax", "dateMin", "dateMax"]
  .forEach((id) => $(id).addEventListener("input", render));

$("resetFilters").addEventListener("click", () => {
  Object.values(state.filters).forEach((s) => s.clear());
  document.querySelectorAll(".chip.active").forEach((c) => c.classList.remove("active"));
  ["b1Min", "b1Max", "availMin", "availMax", "usingMin", "usingMax", "dateMin", "dateMax"]
    .forEach((id) => ($(id).value = ""));
  render();
});

// ---------- filtering ----------
function numOr(id, def) {
  const v = $(id).value;
  return v === "" ? def : Number(v);
}

function passRange(val, min, max) {
  if (val === null || val === undefined) return min === -Infinity && max === Infinity;
  return val >= min && val <= max;
}

function getFiltered() {
  const b1Min = numOr("b1Min", -Infinity), b1Max = numOr("b1Max", Infinity);
  const avMin = numOr("availMin", -Infinity), avMax = numOr("availMax", Infinity);
  const usMin = numOr("usingMin", -Infinity), usMax = numOr("usingMax", Infinity);
  const dMin = $("dateMin").value ? new Date($("dateMin").value).getTime() : -Infinity;
  const dMax = $("dateMax").value ? new Date($("dateMax").value).getTime() : Infinity;
  const f = state.filters;

  return state.rows.filter((r) => {
    if (f.worker.size && !f.worker.has(r.worker)) return false;
    if (f.exponent.size && !f.exponent.has(r.exponent)) return false;
    if (f.curve.size && !f.curve.has(r.curve)) return false;
    if (f.status.size && !f.status.has(r.status)) return false;
    if (!passRange(r.b1, b1Min, b1Max)) return false;
    if (!passRange(r.avail_mem, avMin, avMax)) return false;
    if (!passRange(r.using_mem, usMin, usMax)) return false;
    const sMs = tsToMs(r.start_ts);
    if (sMs < dMin || sMs > dMax) return false;
    return true;
  });
}

// ---------- render ----------
function render() {
  const rows = getFiltered();
  rows.sort((a, b) => {
    let x = a[state.sortKey], y = b[state.sortKey];
    if (x === null || x === undefined) x = NUMERIC.has(state.sortKey) ? -Infinity : "";
    if (y === null || y === undefined) y = NUMERIC.has(state.sortKey) ? -Infinity : "";
    if (x < y) return -state.sortDir;
    if (x > y) return state.sortDir;
    return 0;
  });
  renderCards(rows);
  renderTable(rows);
  renderGantt(rows);
  $("rowCount").textContent = `(${rows.length} / ${state.rows.length})`;
}

function renderCards(rows) {
  const complete = rows.filter((r) => r.status === "complete");
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const s1 = avg(complete.map((r) => r.s1_time).filter((v) => v != null));
  const s2 = avg(complete.map((r) => r.s2_time).filter((v) => v != null));
  const rate = rows.length ? (100 * complete.length / rows.length) : 0;
  const cards = [
    { val: rows.length, lbl: "运行数", cls: "" },
    { val: complete.length, lbl: "完成", cls: "green" },
    { val: rate.toFixed(0) + "%", lbl: "完成率", cls: "" },
    { val: rows.filter((r) => r.status === "interrupted").length, lbl: "中断", cls: "red" },
    { val: s1.toFixed(1) + "s", lbl: "平均 Stage 1", cls: "" },
    { val: s2.toFixed(1) + "s", lbl: "平均 Stage 2", cls: "orange" },
  ];
  $("statCards").innerHTML = cards.map((c) =>
    `<div class="card ${c.cls}"><div class="val">${c.val}</div><div class="lbl">${c.lbl}</div></div>`
  ).join("");
}

function buildHead() {
  const tr = $("theadRow");
  tr.innerHTML = "";
  state.columns.forEach(([key, header]) => {
    const th = document.createElement("th");
    th.textContent = header;
    th.dataset.key = key;
    th.addEventListener("click", () => {
      if (state.sortKey === key) state.sortDir *= -1;
      else { state.sortKey = key; state.sortDir = 1; }
      buildHead();
      render();
    });
    if (state.sortKey === key) {
      const a = document.createElement("span");
      a.className = "arrow";
      a.textContent = state.sortDir > 0 ? "▲" : "▼";
      th.appendChild(a);
    }
    tr.appendChild(th);
  });
}

function renderTable(rows) {
  const tb = $("tbody");
  const frag = document.createDocumentFragment();
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    state.columns.forEach(([key]) => {
      const td = document.createElement("td");
      if (key === "status") {
        td.innerHTML = `<span class="badge ${r.status}">${r.status}</span>`;
      } else {
        td.textContent = fmt(r[key], key);
        if (NUMERIC.has(key)) td.className = "num";
      }
      tr.appendChild(td);
    });
    frag.appendChild(tr);
  });
  tb.innerHTML = "";
  tb.appendChild(frag);
}

// ---------- gantt ----------
function renderGantt(rows) {
  if (!ganttChart) ganttChart = echarts.init($("gantt"), "dark");
  const workers = [...new Set(rows.map((r) => r.worker))].sort((a, b) => a - b);
  const catIndex = new Map(workers.map((w, i) => [w, i]));
  const data = [];

  rows.forEach((r) => {
    const idx = catIndex.get(r.worker);
    const start = tsToMs(r.start_ts);
    const s1End = tsToMs(r.s1_end_ts);
    const s2Start = tsToMs(r.s2_start_ts);
    const s2End = tsToMs(r.s2_end_ts);
    if (s1End) {
      data.push({ value: [idx, start, s1End, "S1", r], itemStyle: { color: "#4c8dff" } });
    } else {
      data.push({ value: [idx, start, tsToMs(r.end_ts), "run", r], itemStyle: { color: "#556080" } });
    }
    if (s2Start && s2End) {
      data.push({ value: [idx, s2Start, s2End, "S2", r], itemStyle: { color: "#ff9f43" } });
    }
  });

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      formatter: (p) => {
        const r = p.value[4];
        return `<b>${r.worker_label}</b> — M${r.exponent} #${r.curve}<br/>`
          + `${p.value[3]}<br/>`
          + `${new Date(p.value[1]).toLocaleString()} → ${new Date(p.value[2]).toLocaleString()}`;
      },
    },
    grid: { left: 90, right: 30, top: 20, bottom: 40 },
    xAxis: { type: "time", axisLabel: { color: "#8b97b4" } },
    yAxis: {
      type: "category",
      data: workers.map((w) => "Worker #" + w),
      axisLabel: { color: "#8b97b4" },
    },
    dataZoom: [
      { type: "slider", filterMode: "weakFilter", height: 16, bottom: 8 },
      { type: "inside", filterMode: "weakFilter" },
    ],
    series: [{
      type: "custom",
      renderItem: (params, api) => {
        const idx = api.value(0);
        const start = api.coord([api.value(1), idx]);
        const end = api.coord([api.value(2), idx]);
        const height = api.size([0, 1])[1] * 0.5;
        const width = Math.max(end[0] - start[0], 2);
        return {
          type: "rect",
          shape: { x: start[0], y: start[1] - height / 2, width, height, r: 2 },
          style: api.style(),
        };
      },
      encode: { x: [1, 2], y: 0 },
      data,
    }],
  };
  ganttChart.setOption(option, true);
  ganttChart.resize();
}

// ---------- export ----------
document.querySelectorAll("[data-fmt]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const fmt = btn.dataset.fmt;
    const scope = $("exportScope").value;
    const rows = scope === "all" ? state.rows : getFilteredSorted();
    try {
      const res = await fetch("/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows, format: fmt }),
      });
      if (!res.ok) throw new Error("导出失败");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "ecm_runs." + fmt;
      a.click();
      URL.revokeObjectURL(url);
      toast(`已导出 ${rows.length} 行 (${fmt.toUpperCase()})`);
    } catch (err) {
      toast(err.message, true);
    }
  });
});

function getFilteredSorted() {
  const rows = getFiltered();
  rows.sort((a, b) => {
    let x = a[state.sortKey], y = b[state.sortKey];
    if (x === null || x === undefined) x = NUMERIC.has(state.sortKey) ? -Infinity : "";
    if (y === null || y === undefined) y = NUMERIC.has(state.sortKey) ? -Infinity : "";
    if (x < y) return -state.sortDir;
    if (x > y) return state.sortDir;
    return 0;
  });
  return rows;
}
