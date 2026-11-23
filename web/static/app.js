// Gödel-Wordle (Prime Decoding) Web Frontend

const GLYPHS = [
  "●", "▲", "◆", "★", "■", "⬢", "✦", "❖", "◉", "⬟",
  "✿", "✚", "✖", "◈", "⬣", "▼", "◀", "▶", "☀", "☾",
  "♠", "♣", "♥", "♦", "⚡", "❄", "🔥", "☘", "⚙", "♛"
];

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZαβγδ";

// 30 distinct visually pleasant colors
const COLORS = [
  "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
  "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
  "#0d9488", "#9333ea", "#ca8a04", "#e11d48", "#0284c7",
  "#16a34a", "#c026d3", "#b45309", "#be123c", "#1d4ed8",
  "#047857", "#a21caf", "#4338ca", "#0e7490", "#6d28d9",
  "#15803d", "#b91c1c", "#0369a1", "#a16207", "#9f1239"
];

const FEEDBACK_META = {
  EXACT:     { icon: "✓", label: "完全正确", cls: "fb-exact" },
  MISPLACED: { icon: "⇄", label: "栏位错误", cls: "fb-misplaced" },
  PARTIAL:   { icon: "◐", label: "部分正确", cls: "fb-partial" },
  WRONG:     { icon: "⊘", label: "完全错误", cls: "fb-wrong" }
};

const MARK_CYCLE = ["none", "confirmed", "excluded", "suspect"];
const MARK_ICON = { none: "⚪", confirmed: "🟢", excluded: "❌", suspect: "⭐" };

let state = {
  session: null,
  draftSlots: [],       // Array<Array<number>> of length L
  activeSlotIdx: 0,
  displayMode: "hybrid", // "hybrid" | "prime" | "symbol"
  cheatVisible: false,
  cardCounterEnabled: localStorage.getItem("primewordle_card_counter") !== "0",
  newestFirst: true,
  symbolMarks: {},       // { "0": "confirmed", ... }
  scratchpad: "",
  notesTimer: null
};

// ---------- Helpers ----------
function el(id) {
  return document.getElementById(id);
}

function symbolColor(idx) {
  return COLORS[idx % COLORS.length];
}

function symbolLabel(idx, primes) {
  const glyph = GLYPHS[idx % GLYPHS.length];
  const letter = LETTERS[idx % LETTERS.length];
  const prime = primes[idx];
  if (state.displayMode === "prime") {
    return `${prime}`;
  }
  if (state.displayMode === "symbol") {
    return `${glyph}${letter}`;
  }
  return `${glyph}${prime}`;
}

function groupExponents(indices) {
  const map = new Map();
  for (const idx of indices) {
    map.set(idx, (map.get(idx) || 0) + 1);
  }
  return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
}

function computeBigProduct(indices, primes) {
  if (!indices || indices.length === 0) return null;
  let prod = 1n;
  for (const idx of indices) {
    prod *= BigInt(primes[idx]);
  }
  return prod.toString();
}

function formatBigProduct(prodStr) {
  if (!prodStr) return "—";
  if (prodStr.length <= 12) return `∏ = ${prodStr}`;
  return `∏ ≈ ${prodStr.slice(0, 6)}…(${prodStr.length}位)`;
}

function showError(msg) {
  el("error-msg").textContent = msg || "";
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `请求失败 (${res.status})`);
  }
  return data;
}

// ---------- Game Lifecycle ----------
async function createNewGame(customCfg = null) {
  showError("");
  const cfg = customCfg || {
    num_primes: Number(el("slider-p").value),
    max_multiplier: Number(el("slider-k").value),
    code_length: Number(el("slider-l").value),
    allow_underfill: el("chk-underfill").checked,
    allow_duplicate_slots: el("chk-dup-slots").checked,
    seed: el("input-seed").value.trim() || null
  };

  try {
    const session = await api("/api/games", {
      method: "POST",
      body: JSON.stringify(cfg)
    });
    loadSessionIntoUI(session, false);
    refreshArchive();
  } catch (err) {
    showError(err.message);
  }
}

async function openSavedGame(gameId) {
  showError("");
  try {
    const session = await api(`/api/games/${gameId}`);
    loadSessionIntoUI(session, Boolean(session.peeked && session.target));
    refreshArchive();
  } catch (err) {
    showError(err.message);
  }
}

function loadSessionIntoUI(session, showCheat) {
  state.session = session;
  state.cheatVisible = showCheat;
  state.activeSlotIdx = 0;
  state.draftSlots = Array.from({ length: session.config.code_length }, () => []);
  state.symbolMarks = { ...(session.notes?.symbol_marks || {}) };
  state.scratchpad = session.notes?.scratchpad || "";

  // Sync sliders with loaded game config
  el("slider-p").value = session.config.num_primes;
  el("val-p").textContent = session.config.num_primes;
  el("slider-k").value = session.config.max_multiplier;
  el("val-k").textContent = session.config.max_multiplier;
  el("slider-l").value = session.config.code_length;
  el("val-l").textContent = session.config.code_length;
  el("chk-underfill").checked = session.config.allow_underfill;
  el("chk-dup-slots").checked = session.config.allow_duplicate_slots;

  el("scratchpad-input").value = state.scratchpad;
  localStorage.setItem("primewordle_last_game_id", session.game_id);

  renderAll();
}

async function submitGuess() {
  if (!state.session || state.session.status !== "playing") return;
  showError("");
  const cfg = state.session.config;

  for (let i = 0; i < cfg.code_length; i++) {
    const len = state.draftSlots[i].length;
    if (len < 1) {
      showError(`第 ${i + 1} 格不能为空（至少需要 1 个符号）。`);
      state.activeSlotIdx = i;
      renderComposer();
      return;
    }
    if (!cfg.allow_underfill && len !== cfg.max_multiplier) {
      showError(`当前已关闭“不满填”：第 ${i + 1} 格必须填满 ${cfg.max_multiplier} 个符号。`);
      state.activeSlotIdx = i;
      renderComposer();
      return;
    }
  }

  try {
    const updated = await api(`/api/games/${state.session.game_id}/guess`, {
      method: "POST",
      body: JSON.stringify({ guess_slots: state.draftSlots })
    });
    state.session = updated;
    if (updated.status === "won") {
      state.cheatVisible = true;
    }
    renderAll();
    refreshArchive();
  } catch (err) {
    showError(err.message);
  }
}

async function toggleCheat() {
  if (!state.session) return;
  if (state.cheatVisible) {
    state.cheatVisible = false;
    renderCheatBanner();
    return;
  }
  try {
    if (!state.session.target) {
      state.session = await api(`/api/games/${state.session.game_id}/peek`, {
        method: "POST"
      });
    }
    state.cheatVisible = true;
    renderAll();
    refreshArchive();
  } catch (err) {
    showError(err.message);
  }
}

async function randomFill() {
  if (!state.session || state.session.status !== "playing") return;
  try {
    const res = await api(`/api/games/${state.session.game_id}/random_guess`, {
      method: "POST"
    });
    state.draftSlots = res.guess_slots;
    showError("");
    renderComposer();
  } catch (err) {
    showError(err.message);
  }
}

function scheduleSaveNotes() {
  if (!state.session) return;
  el("notes-save-status").textContent = "保存中...";
  clearTimeout(state.notesTimer);
  state.notesTimer = setTimeout(async () => {
    try {
      await api(`/api/games/${state.session.game_id}/notes`, {
        method: "POST",
        body: JSON.stringify({
          symbol_marks: state.symbolMarks,
          scratchpad: state.scratchpad
        })
      });
      el("notes-save-status").textContent = "已保存";
    } catch {
      el("notes-save-status").textContent = "保存失败";
    }
  }, 350);
}

// ---------- Rendering ----------
function renderAll() {
  renderStatusStrip();
  renderCheatBanner();
  renderVictoryBanner();
  renderComposer();
  renderHistory();
  renderNotebook();
}

function renderStatusStrip() {
  const s = state.session;
  if (!s) return;
  const c = s.config;
  el("pill-config").textContent = `P=${c.num_primes} · K=${c.max_multiplier} · L=${c.code_length}`;
  el("pill-underfill").textContent = c.allow_underfill ? "可不满填 (1~K)" : `必须填满 (${c.max_multiplier})`;
  el("pill-turns").textContent = `已校验 ${s.turns} 轮`;
  el("pill-tokens").textContent = `轨迹 ${s.token_count} Tokens`;

  const statusEl = el("pill-status");
  if (s.status === "won") {
    statusEl.textContent = s.peeked ? "已通关 (含透视)" : "🎉 已通关";
    statusEl.className = "pill pill-won";
  } else {
    statusEl.textContent = "进行中";
    statusEl.className = "pill pill-playing";
  }
}

function renderCheatBanner() {
  const banner = el("cheat-banner");
  const s = state.session;
  if (!s || !state.cheatVisible || !s.target) {
    banner.classList.add("hidden");
    el("btn-cheat").textContent = "👁 透视答案";
    return;
  }
  banner.classList.remove("hidden");
  el("btn-cheat").textContent = "🙈 隐藏答案";

  const container = el("cheat-slots");
  container.innerHTML = "";
  s.target.slots.forEach((slot, i) => {
    const prodStr = s.target.products[i];
    const box = document.createElement("div");
    box.className = "slot-box";
    box.title = `点击把答案第 ${i + 1} 格填入输入框\n完整哥德尔乘积: ${prodStr}`;
    box.addEventListener("click", () => {
      if (state.session.status !== "playing") return;
      state.draftSlots[i] = [...slot];
      state.activeSlotIdx = i;
      renderComposer();
    });

    const top = document.createElement("div");
    top.className = "slot-top";
    top.innerHTML = `<span>答案 #${i + 1}</span><span>${slot.length} 符号</span>`;

    const chips = document.createElement("div");
    chips.className = "slot-chips";
    for (const [symIdx, exp] of groupExponents(slot)) {
      const chip = document.createElement("span");
      chip.className = "sym-chip";
      chip.style.background = symbolColor(symIdx);
      chip.innerHTML = `${symbolLabel(symIdx, s.primes)}${exp > 1 ? `<sup>${exp}</sup>` : ""}`;
      chips.appendChild(chip);
    }

    const prod = document.createElement("div");
    prod.className = "slot-product";
    prod.textContent = formatBigProduct(prodStr);

    box.append(top, chips, prod);
    container.appendChild(box);
  });
}

function renderVictoryBanner() {
  const banner = el("victory-banner");
  const s = state.session;
  if (!s || s.status !== "won") {
    banner.classList.add("hidden");
    return;
  }
  banner.classList.remove("hidden");
  const suffix = s.peeked ? "（本局开启过透视模式）" : "（纯靠推理通关！）";
  el("victory-text").textContent = `🎉 破解成功！共用 ${s.turns} 轮校验，轨迹共 ${s.token_count} Tokens ${suffix}`;
}

function renderComposer() {
  const s = state.session;
  if (!s) return;
  const cfg = s.config;
  const isPlaying = s.status === "playing";
  const allDeductions = s.deduction?.slots || [];

  const slotsContainer = el("active-slots");
  slotsContainer.innerHTML = "";

  for (let i = 0; i < cfg.code_length; i++) {
    const slot = state.draftSlots[i];
    const slotDed = allDeductions[i];
    const isSolved = Boolean(state.cardCounterEnabled && slotDed?.solved);
    const box = document.createElement("div");
    box.className = `slot-box${i === state.activeSlotIdx ? " active" : ""}${isSolved ? " slot-solved" : ""}`;
    box.addEventListener("click", () => {
      state.activeSlotIdx = i;
      renderComposer();
    });
    box.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      if (!isPlaying) return;
      state.draftSlots[i] = [];
      state.activeSlotIdx = i;
      renderComposer();
    });

    const top = document.createElement("div");
    top.className = "slot-top";
    const solvedBadge = isSolved ? ` <span style="color:var(--exact);font-weight:700;">✓已解</span>` : "";
    top.innerHTML = `<span>格子 #${i + 1}${solvedBadge}</span><span>${slot.length}/${cfg.max_multiplier}</span>`;

    const chips = document.createElement("div");
    chips.className = "slot-chips";
    if (slot.length === 0) {
      const hint = document.createElement("span");
      hint.className = "slot-empty-hint";
      hint.textContent = "点击下方添加";
      chips.appendChild(hint);
    } else {
      for (const [symIdx, exp] of groupExponents(slot)) {
        const chip = document.createElement("span");
        chip.className = "sym-chip removable";
        chip.style.background = symbolColor(symIdx);
        chip.title = "点击移除 1 个该符号";
        chip.innerHTML = `${symbolLabel(symIdx, s.primes)}${exp > 1 ? `<sup>${exp}</sup>` : ""}`;
        chip.addEventListener("click", (e) => {
          e.stopPropagation();
          if (!isPlaying) return;
          const pos = state.draftSlots[i].lastIndexOf(symIdx);
          if (pos !== -1) state.draftSlots[i].splice(pos, 1);
          state.activeSlotIdx = i;
          renderComposer();
        });
        chips.appendChild(chip);
      }
    }

    const prodStr = computeBigProduct(slot, s.primes);
    const prod = document.createElement("div");
    prod.className = "slot-product";
    prod.title = prodStr ? `哥德尔乘积: ${prodStr}` : "";
    prod.textContent = formatBigProduct(prodStr);

    box.append(top, chips, prod);
    slotsContainer.appendChild(box);
  }

  el("active-slot-num").textContent = state.activeSlotIdx + 1;
  el("active-slot-count").textContent = state.draftSlots[state.activeSlotIdx].length;
  el("active-slot-max").textContent = cfg.max_multiplier;

  // Card Counter toggle button & per-slot deduction summary bar
  const ccBtn = el("btn-card-counter");
  ccBtn.textContent = state.cardCounterEnabled ? "🃏 记牌器：开" : "🃏 记牌器：关";
  ccBtn.className = `btn btn-small ${state.cardCounterEnabled ? "btn-counter-on" : "btn-counter-off"}`;

  const activeDed = allDeductions[state.activeSlotIdx];
  const dedBar = el("deduction-bar");
  if (!state.cardCounterEnabled) {
    dedBar.classList.add("hidden");
  } else {
    dedBar.classList.remove("hidden");
    const fmtList = (arr) =>
      arr && arr.length > 0 ? arr.map((idx) => symbolLabel(idx, s.primes)).join("、") : "无";

    if (!activeDed || s.history.length === 0) {
      dedBar.innerHTML = `<span>🃏 <strong>第 ${state.activeSlotIdx + 1} 格记牌器</strong>：提交至少 1 轮后自动标注 ✕ 排除 / ? 疑似 / ✓ 必含</span>`;
    } else if (activeDed.solved) {
      const exactStr = groupExponents(activeDed.exact_slot)
        .map(([idx, exp]) => `${symbolLabel(idx, s.primes)}${exp > 1 ? `^${exp}` : ""}`)
        .join(" × ");
      dedBar.innerHTML = `<span class="deduce-tag-req">✅ 第 ${state.activeSlotIdx + 1} 格已完全锁定：${exactStr}</span>`;
    } else {
      dedBar.innerHTML = `
        <span>🃏 <strong>第 ${state.activeSlotIdx + 1} 格记牌器</strong>：</span>
        <span><span class="deduce-tag-ex">✕ 已排除</span>：${fmtList(activeDed.excluded)}</span>
        <span><span class="deduce-tag-req">✓ 必含</span>：${fmtList(activeDed.required)}</span>
        <span><span class="deduce-tag-unc">? 疑似(对一半)</span>：${fmtList(activeDed.uncertain)}</span>
      `;
    }
  }

  // Symbol Palette
  const palette = el("symbol-palette");
  palette.innerHTML = "";
  for (let symIdx = 0; symIdx < cfg.num_primes; symIdx++) {
    const mark = state.symbolMarks[String(symIdx)] || "none";
    const dedStatus = state.cardCounterEnabled
      ? activeDed?.per_symbol?.[String(symIdx)] || "unknown"
      : "unknown";

    const btn = document.createElement("button");
    const classes = ["palette-btn"];
    if (mark !== "none") classes.push(`mark-${mark}`);
    if (dedStatus !== "unknown") classes.push(`deduce-${dedStatus}`);
    btn.className = classes.join(" ");
    btn.style.background = symbolColor(symIdx);
    btn.disabled = !isPlaying;

    const dedTipMap = {
      excluded: " [记牌器：本格绝对不含 ✕]",
      uncertain: " [记牌器：本格部分正确候选 ?]",
      required: " [记牌器：本格已推出必含 ✓]",
      exact: " [记牌器：本格已完全锁定 ✓]",
      unknown: ""
    };
    btn.title = `符号 S_${symIdx} (对应质数 ${s.primes[symIdx]})${dedTipMap[dedStatus] || ""}`;

    const dot = mark !== "none" ? `<span class="palette-mark-dot">${MARK_ICON[mark]}</span>` : "";
    let cornerBadge = "";
    if (dedStatus === "excluded") {
      cornerBadge = `<span class="deduce-corner-badge badge-ex">✕</span>`;
    } else if (dedStatus === "uncertain") {
      cornerBadge = `<span class="deduce-corner-badge badge-unc">?</span>`;
    } else if (dedStatus === "required" || dedStatus === "exact") {
      cornerBadge = `<span class="deduce-corner-badge badge-req">✓</span>`;
    }

    btn.innerHTML = `${symbolLabel(symIdx, s.primes)}${dot}${cornerBadge}`;
    btn.addEventListener("click", () => {
      if (!isPlaying) return;
      const targetSlot = state.draftSlots[state.activeSlotIdx];
      if (targetSlot.length >= cfg.max_multiplier) {
        showError(`格子 #${state.activeSlotIdx + 1} 已达最大乘数 K=${cfg.max_multiplier}，请先移除或切换格子。`);
        return;
      }
      showError("");
      targetSlot.push(symIdx);
      // Auto-advance when K=1 for crisp Wordle feel
      if (cfg.max_multiplier === 1 && state.activeSlotIdx < cfg.code_length - 1) {
        state.activeSlotIdx += 1;
      }
      renderComposer();
    });
    palette.appendChild(btn);
  }

  el("btn-submit").disabled = !isPlaying;
  el("btn-copy-last").disabled = !isPlaying || s.history.length === 0;
  el("btn-random-fill").disabled = !isPlaying;
  el("btn-clear-all").disabled = !isPlaying;
}

function renderHistory() {
  const s = state.session;
  const list = el("history-list");
  list.innerHTML = "";

  if (!s || s.history.length === 0) {
    list.innerHTML = `<div class="empty-state">尚未提交猜测。请在上方构造你的第一轮组合！</div>`;
    return;
  }

  const records = state.newestFirst ? [...s.history].reverse() : s.history;
  for (const rec of records) {
    const row = document.createElement("div");
    row.className = "history-row";

    const header = document.createElement("div");
    header.className = "history-row-header";

    const exactCount = rec.feedback.filter((f) => f === "EXACT").length;
    header.innerHTML = `
      <div>
        <span class="turn-badge">#${rec.turn}</span>
        <span> · 命中全对 ${exactCount}/${s.config.code_length}</span>
      </div>
    `;

    const loadBtn = document.createElement("button");
    loadBtn.className = "btn btn-small btn-secondary";
    loadBtn.textContent = "↩ 载入此轮";
    loadBtn.disabled = s.status !== "playing";
    loadBtn.addEventListener("click", () => {
      state.draftSlots = rec.guess_slots.map((slot) => [...slot]);
      showError("");
      renderComposer();
    });
    header.appendChild(loadBtn);

    const slotsGrid = document.createElement("div");
    slotsGrid.className = "history-slots";

    rec.guess_slots.forEach((slot, idx) => {
      const fb = rec.feedback[idx];
      const meta = FEEDBACK_META[fb];
      const prodStr = rec.guess_products[idx];

      const cell = document.createElement("div");
      cell.className = `h-slot state-${fb}`;
      cell.title = `第 ${idx + 1} 格 | ${meta.label}\n哥德尔乘积: ${prodStr}`;

      const chips = document.createElement("div");
      chips.className = "slot-chips";
      for (const [symIdx, exp] of groupExponents(slot)) {
        const chip = document.createElement("span");
        chip.className = "sym-chip";
        chip.style.background = symbolColor(symIdx);
        chip.innerHTML = `${symbolLabel(symIdx, s.primes)}${exp > 1 ? `<sup>${exp}</sup>` : ""}`;
        chips.appendChild(chip);
      }

      const foot = document.createElement("div");
      foot.className = "h-slot-foot";
      foot.innerHTML = `
        <span class="${meta.cls}" style="padding:1px 5px;border-radius:4px;">${meta.icon} ${meta.label}</span>
        <span style="color:var(--text-muted);font-family:Consolas,monospace;font-size:10px;">${prodStr.length <= 7 ? prodStr : prodStr.slice(0, 5) + "…"}</span>
      `;

      cell.append(chips, foot);
      slotsGrid.appendChild(cell);
    });

    row.append(header, slotsGrid);
    list.appendChild(row);
  }
}

function renderNotebook() {
  const s = state.session;
  const grid = el("notebook-symbols");
  grid.innerHTML = "";
  if (!s) return;

  // Tally appearances by feedback across all history
  const stats = Array.from({ length: s.config.num_primes }, () => ({
    EXACT: 0,
    MISPLACED: 0,
    PARTIAL: 0,
    WRONG: 0
  }));

  for (const rec of s.history) {
    rec.guess_slots.forEach((slot, i) => {
      const fb = rec.feedback[i];
      const uniq = new Set(slot);
      for (const symIdx of uniq) {
        if (stats[symIdx]) stats[symIdx][fb] += 1;
      }
    });
  }

  for (let symIdx = 0; symIdx < s.config.num_primes; symIdx++) {
    const key = String(symIdx);
    const mark = state.symbolMarks[key] || "none";
    const st = stats[symIdx];

    const card = document.createElement("div");
    card.className = `nb-sym-card${mark !== "none" ? ` mark-${mark}` : ""}`;
    card.title = "点击切换标记：未标记 → 确认 → 排除 → 怀疑";
    card.addEventListener("click", () => {
      const nextIdx = (MARK_CYCLE.indexOf(mark) + 1) % MARK_CYCLE.length;
      state.symbolMarks[key] = MARK_CYCLE[nextIdx];
      renderNotebook();
      renderComposer();
      scheduleSaveNotes();
    });

    card.innerHTML = `
      <div class="nb-sym-top">
        <span class="sym-chip" style="background:${symbolColor(symIdx)};padding:2px 6px;">
          ${symbolLabel(symIdx, s.primes)}
        </span>
        <span>${MARK_ICON[mark]}</span>
      </div>
      <div class="nb-stats">
        <span style="color:var(--exact)">✓${st.EXACT}</span>
        <span style="color:var(--misplaced)">⇄${st.MISPLACED}</span>
        <span style="color:var(--partial)">◐${st.PARTIAL}</span>
        <span style="color:var(--wrong)">⊘${st.WRONG}</span>
      </div>
    `;
    grid.appendChild(card);
  }
}

async function refreshArchive() {
  try {
    const res = await api("/api/games");
    const list = el("archive-list");
    list.innerHTML = "";
    if (!res.games || res.games.length === 0) {
      list.innerHTML = `<div class="empty-state">暂无存档</div>`;
      return;
    }

    for (const g of res.games) {
      const item = document.createElement("div");
      const isCurrent = state.session && state.session.game_id === g.game_id;
      item.className = `archive-item${isCurrent ? " current" : ""}`;
      const c = g.config;
      const statusText = g.status === "won" ? "🎉 已通关" : "⏳ 进行中";
      const peekTag = g.peeked ? " · 👁" : "";
      item.innerHTML = `
        <div class="archive-meta">
          <strong>P=${c.num_primes} K=${c.max_multiplier} L=${c.code_length} ${c.allow_underfill ? "" : "(满填)"}</strong>
          <span class="archive-sub">${statusText} · ${g.turns} 轮${peekTag} · ID:${g.game_id.slice(0, 6)}</span>
        </div>
      `;
      const btn = document.createElement("button");
      btn.className = "btn btn-small btn-secondary";
      btn.textContent = isCurrent ? "当前" : "打开";
      btn.disabled = isCurrent;
      btn.addEventListener("click", () => openSavedGame(g.game_id));
      item.appendChild(btn);
      list.appendChild(item);
    }
  } catch {
    // Ignore transient archive errors
  }
}

// ---------- Event Wiring ----------
function bindEvents() {
  // Sliders
  ["p", "k", "l"].forEach((key) => {
    const slider = el(`slider-${key}`);
    const out = el(`val-${key}`);
    slider.addEventListener("input", () => {
      out.textContent = slider.value;
    });
  });

  // Presets
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      el("slider-p").value = btn.dataset.p;
      el("val-p").textContent = btn.dataset.p;
      el("slider-k").value = btn.dataset.k;
      el("val-k").textContent = btn.dataset.k;
      el("slider-l").value = btn.dataset.l;
      el("val-l").textContent = btn.dataset.l;
      el("chk-underfill").checked = btn.dataset.u === "1";
      createNewGame();
    });
  });

  el("btn-new-game").addEventListener("click", () => createNewGame());
  el("btn-victory-new").addEventListener("click", () => createNewGame());
  el("btn-toggle-config").addEventListener("click", () => {
    el("config-panel").classList.toggle("collapsed");
  });

  // Display mode switcher
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.displayMode = btn.dataset.mode;
      renderAll();
    });
  });

  // Cheat controls
  el("btn-cheat").addEventListener("click", toggleCheat);
  el("btn-hide-cheat").addEventListener("click", () => {
    state.cheatVisible = false;
    renderCheatBanner();
  });
  el("btn-fill-answer").addEventListener("click", () => {
    if (!state.session || !state.session.target || state.session.status !== "playing") return;
    state.draftSlots = state.session.target.slots.map((s) => [...s]);
    showError("");
    renderComposer();
  });

  // Card counter toggle
  el("btn-card-counter").addEventListener("click", () => {
    state.cardCounterEnabled = !state.cardCounterEnabled;
    localStorage.setItem("primewordle_card_counter", state.cardCounterEnabled ? "1" : "0");
    renderComposer();
  });

  // Composer tools
  el("btn-submit").addEventListener("click", submitGuess);
  el("btn-random-fill").addEventListener("click", randomFill);
  el("btn-clear-all").addEventListener("click", () => {
    if (!state.session) return;
    state.draftSlots = Array.from({ length: state.session.config.code_length }, () => []);
    state.activeSlotIdx = 0;
    showError("");
    renderComposer();
  });
  el("btn-copy-last").addEventListener("click", () => {
    if (!state.session || state.session.history.length === 0) return;
    const last = state.session.history[state.session.history.length - 1];
    state.draftSlots = last.guess_slots.map((s) => [...s]);
    showError("");
    renderComposer();
  });

  // History sort toggle
  el("btn-sort-history").addEventListener("click", () => {
    state.newestFirst = !state.newestFirst;
    el("btn-sort-history").textContent = state.newestFirst
      ? "排序：最新在上 ⬇"
      : "排序：按轮次正序 ⬆";
    renderHistory();
  });

  // Sidebar tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      el(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "archive") refreshArchive();
    });
  });

  el("btn-refresh-archive").addEventListener("click", refreshArchive);

  // Scratchpad autosave
  el("scratchpad-input").addEventListener("input", (e) => {
    state.scratchpad = e.target.value;
    scheduleSaveNotes();
  });

  // Keyboard shortcuts (ignore when typing in textarea or input)
  document.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT") return;
    if (!state.session || state.session.status !== "playing") return;

    const L = state.session.config.code_length;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      state.activeSlotIdx = (state.activeSlotIdx + 1) % L;
      renderComposer();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      state.activeSlotIdx = (state.activeSlotIdx - 1 + L) % L;
      renderComposer();
    } else if (e.key === "Backspace") {
      e.preventDefault();
      const slot = state.draftSlots[state.activeSlotIdx];
      if (slot.length > 0) {
        slot.pop();
        renderComposer();
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      submitGuess();
    }
  });
}

// ---------- Boot ----------
async function init() {
  bindEvents();
  const lastId = localStorage.getItem("primewordle_last_game_id");
  if (lastId) {
    try {
      const session = await api(`/api/games/${lastId}`);
      loadSessionIntoUI(session, Boolean(session.peeked && session.target));
      refreshArchive();
      return;
    } catch {
      // Fallback to creating a fresh game
    }
  }
  await createNewGame();
}

init();
