const LANES = [
  { value: "top", label: "TOP" },
  { value: "jungle", label: "JUNGLE" },
  { value: "mid", label: "MID" },
  { value: "adc", label: "ADC" },
  { value: "support", label: "SUPPORT" },
];

let ALL_CHAMPIONS = [];
const selections = {}; // e.g. selections["counters-enemy"] = "Ahri"
const laneSelections = {}; // e.g. laneSelections["counters-lane"] = "top"

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `İstek başarısız (${res.status})`);
  }
  return res.json();
}

/* ---------------- Tabs ---------------- */
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

/* ---------------- Lane picker ---------------- */
function buildLanePicker(container, targetKey) {
  container.innerHTML = "";
  LANES.forEach((lane, i) => {
    const pill = document.createElement("button");
    pill.className = "lane-pill" + (i === 0 ? " active" : "");
    pill.textContent = lane.label;
    pill.addEventListener("click", () => {
      container.querySelectorAll(".lane-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      laneSelections[targetKey] = lane.value;
    });
    container.appendChild(pill);
  });
  laneSelections[targetKey] = LANES[0].value;
}

/* ---------------- Champion picker (autocomplete) ---------------- */
function buildChampionPicker(container, targetKey) {
  container.innerHTML = `
    <div class="selected-chip"><img src="" alt=""><span></span></div>
    <input type="text" placeholder="Şampiyon ara..." autocomplete="off">
    <div class="dropdown"></div>
  `;
  const chip = container.querySelector(".selected-chip");
  const chipImg = chip.querySelector("img");
  const chipName = chip.querySelector("span");
  const input = container.querySelector("input");
  const dropdown = container.querySelector(".dropdown");

  function renderList(filter) {
    const term = filter.trim().toLowerCase();
    const matches = ALL_CHAMPIONS.filter((c) => c.name.toLowerCase().includes(term)).slice(0, 40);
    dropdown.innerHTML = "";
    if (matches.length === 0) {
      dropdown.innerHTML = `<div class="dropdown-item" style="opacity:0.5;cursor:default;">Sonuç yok</div>`;
    }
    matches.forEach((c) => {
      const item = document.createElement("div");
      item.className = "dropdown-item";
      item.innerHTML = `<img src="${c.icon_url || ""}" onerror="this.style.visibility='hidden'"><span>${c.name}</span>`;
      item.addEventListener("click", () => select(c));
      dropdown.appendChild(item);
    });
    dropdown.classList.add("open");
  }

  function select(champion) {
    selections[targetKey] = champion.name;
    chipImg.src = champion.icon_url || "";
    chipName.textContent = champion.name;
    chip.classList.add("visible");
    input.value = "";
    dropdown.classList.remove("open");
  }

  input.addEventListener("focus", () => renderList(input.value));
  input.addEventListener("input", () => renderList(input.value));
  input.addEventListener("blur", () => setTimeout(() => dropdown.classList.remove("open"), 150));

  chip.addEventListener("click", () => {
    delete selections[targetKey];
    chip.classList.remove("visible");
    input.value = "";
    input.focus();
  });
}

/* ---------------- Model 1: Counter Pick ---------------- */
function difficultyBadgeClass(label) {
  if (label === "Kolay") return "diff-kolay";
  if (label === "Orta") return "diff-orta";
  if (label === "Zor") return "diff-zor";
  return "";
}

function renderCounters(data) {
  const area = document.getElementById("counters-results");
  area.innerHTML = "";

  if (data.source === "learned_model_fallback") {
    const banner = document.createElement("div");
    banner.className = "source-banner";
    banner.innerHTML = `<span>⚠</span><span>Bu rakibe karşı doğrudan maç verisi yok - sonuçlar <strong>Blade & Chest</strong> modelinin tahminidir, gerçek gözlem değildir.</span>`;
    area.appendChild(banner);
  }

  if (data.results.length === 0) {
    area.innerHTML += `<div class="hint">Bu eşleşme için hiçbir veri bulunamadı.</div>`;
    return;
  }

  data.results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "counter-card";
    const pct = (r.win_rate * 100).toFixed(1);
    const diffBadge = r.difficulty_label
      ? `<span class="badge ${difficultyBadgeClass(r.difficulty_label)}">${r.difficulty_label}</span>`
      : "";
    const confText = r.confidence_score !== null
      ? `güven-ayarlı: <b>%${(r.confidence_score * 100).toFixed(1)}</b> · ${r.games} maç`
      : `${r.games} maç`;

    card.innerHTML = `
      <div class="counter-rank">#${i + 1}</div>
      <img class="champ-icon" src="${r.icon_url || ""}" onerror="this.style.visibility='hidden'">
      <div class="counter-info">
        <div class="name-row">
          <strong>${r.champion}</strong>
          ${diffBadge}
        </div>
        <div class="win-bar-track"><div class="win-bar-fill" style="width:${pct}%"></div></div>
        <div class="win-rate-num">Kazanma oranı: <b>%${pct}</b> · ${confText}</div>
        <div class="explanation">${r.explanation}</div>
      </div>
    `;
    area.appendChild(card);
  });
}

async function submitCounters() {
  const enemy = selections["counters-enemy"];
  const lane = laneSelections["counters-lane"];
  const area = document.getElementById("counters-results");

  if (!enemy) {
    area.innerHTML = `<div class="hint error">Lütfen bir rakip şampiyon seç.</div>`;
    return;
  }

  area.innerHTML = `<div class="hint">Yükleniyor...</div>`;
  try {
    const data = await fetchJSON(`/api/counters?enemy_champion=${encodeURIComponent(enemy)}&lane=${lane}&top_n=5`);
    renderCounters(data);
  } catch (err) {
    area.innerHTML = `<div class="hint error">${err.message}</div>`;
  }
}

/* ---------------- Model 2: Matchup Intelligence ---------------- */
function statBox(label, value, sub, cls) {
  return `
    <div class="stat-box">
      <div class="label">${label}</div>
      <div class="value ${cls || ""}">${value}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ""}
    </div>
  `;
}

function renderItemGrid(items) {
  if (!items || items.length === 0) return `<div class="hint" style="padding:8px 0;">Belirgin bir tercih yok.</div>`;
  return `<div class="item-grid">${items.map((it) => `
    <div class="item-chip">
      <img src="${it.icon_url || ""}" onerror="this.style.visibility='hidden'">
      <div>
        <div>${it.name}</div>
        <div class="pick-rate">%${(it.pick_rate * 100).toFixed(0)} maçta</div>
        ${(it.reasons || []).map((r) => `<div class="item-reason">${r}</div>`).join("")}
      </div>
    </div>
  `).join("")}</div>`;
}

function renderMatchup(report) {
  const area = document.getElementById("matchup-results");
  area.innerHTML = "";

  const card = document.createElement("div");
  card.className = "card";

  const header = `
    <div class="matchup-header">
      <div class="side"><img src="${report.my_champion_icon || ""}"><span>${report.my_champion}</span></div>
      <div class="vs">VS</div>
      <div class="side"><img src="${report.enemy_champion_icon || ""}"><span>${report.enemy_champion}</span></div>
    </div>
  `;

  let winSub = `${report.games} maç`;
  if (report.win_rate_source === "learned_model_fallback") {
    winSub = "Blade & Chest tahmini (doğrudan veri yok)";
  } else if (report.confidence_score !== null && report.confidence_score !== undefined) {
    winSub = `güven-ayarlı: %${(report.confidence_score * 100).toFixed(1)} · ${report.games} maç`;
  }

  const stats = [
    statBox("Kazanma Oranı", `%${(report.win_rate * 100).toFixed(1)}`, winSub,
      report.win_rate >= 0.5 ? "positive" : "negative"),
  ];

  if (report.expected_gold_diff !== undefined) {
    stats.push(statBox("Beklenen Altın Farkı", `${report.expected_gold_diff >= 0 ? "+" : ""}${report.expected_gold_diff.toFixed(0)}`,
      "maç genelinde", report.expected_gold_diff >= 0 ? "positive" : "negative"));
    stats.push(statBox("Beklenen CS Farkı", `${report.expected_cs_diff >= 0 ? "+" : ""}${report.expected_cs_diff.toFixed(1)}`,
      "maç genelinde", report.expected_cs_diff >= 0 ? "positive" : "negative"));
  }

  let sourceBanner = "";
  if (report.build_source_label) {
    let extra = "";
    if (report.build_source === "similarity_fallback" && report.similar_opponents) {
      extra = `<div class="tag-list">${report.similar_opponents.map((c) => `<span class="tag">${c}</span>`).join("")}</div>`;
    }
    if (report.build_source === "stat_based_fallback" && report.stat_reasoning && report.stat_reasoning.length) {
      extra = `<div class="tag-list">${report.stat_reasoning.map((r) => `<span class="tag">${r}</span>`).join("")}</div>`;
    }
    sourceBanner = `
      <div class="source-banner">
        <span>ℹ</span>
        <div><strong>Kaynak:</strong> ${report.build_source_label}${extra}</div>
      </div>
    `;
  }

  let buildSection = "";
  if (report.top_core_items !== null && report.top_core_items !== undefined) {
    buildSection = `
      <div class="build-section">
        <h3>Çizme</h3>
        ${renderItemGrid(report.top_boots)}
      </div>
      <div class="build-section">
        <h3>Çekirdek Item'lar</h3>
        ${renderItemGrid(report.top_core_items)}
      </div>
      <div class="build-section">
        <h3>Rün Dizilimi</h3>
        <div class="rune-row">
          ${(report.rune_combo?.labeled_perks || []).map((p) => `
            <div class="rune-chip">
              <img src="${p.icon_url || ""}" onerror="this.style.visibility='hidden'">
              <span>${p.name}</span>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  } else {
    buildSection = `<div class="hint">Bu eşleşme için yeterli item/rün verisi yok.</div>`;
  }

  card.innerHTML = header + `<div class="stat-grid">${stats.join("")}</div>` + sourceBanner + buildSection;
  area.appendChild(card);
}

async function submitMatchup() {
  const my = selections["matchup-my"];
  const enemy = selections["matchup-enemy"];
  const lane = laneSelections["matchup-lane"];
  const area = document.getElementById("matchup-results");

  if (!my || !enemy) {
    area.innerHTML = `<div class="hint error">Lütfen iki şampiyon da seç.</div>`;
    return;
  }

  area.innerHTML = `<div class="hint">Analiz ediliyor...</div>`;
  try {
    const data = await fetchJSON(
      `/api/matchup?my_champion=${encodeURIComponent(my)}&enemy_champion=${encodeURIComponent(enemy)}&lane=${lane}`
    );
    renderMatchup(data);
  } catch (err) {
    area.innerHTML = `<div class="hint error">${err.message}</div>`;
  }
}

/* ---------------- Init ---------------- */
async function init() {
  initTabs();

  buildLanePicker(document.querySelector('.lane-picker[data-target="counters-lane"]'), "counters-lane");
  buildLanePicker(document.querySelector('.lane-picker[data-target="matchup-lane"]'), "matchup-lane");

  document.getElementById("counters-submit").addEventListener("click", submitCounters);
  document.getElementById("matchup-submit").addEventListener("click", submitMatchup);

  try {
    ALL_CHAMPIONS = await fetchJSON("/api/champions");
  } catch (err) {
    console.error("Şampiyon listesi yüklenemedi", err);
  }

  buildChampionPicker(document.querySelector('.champion-picker[data-target="counters-enemy"]'), "counters-enemy");
  buildChampionPicker(document.querySelector('.champion-picker[data-target="matchup-my"]'), "matchup-my");
  buildChampionPicker(document.querySelector('.champion-picker[data-target="matchup-enemy"]'), "matchup-enemy");
}

init();
