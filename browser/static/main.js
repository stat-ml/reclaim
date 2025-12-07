const state = {
  retrievals: [],
  selectedId: null,
  activeClaimId: null,
};

const els = {
  fileInput: document.getElementById("file-input"),
  status: document.getElementById("status"),
  retrievalList: document.getElementById("retrieval-list"),
  retrievalTitle: document.getElementById("retrieval-title"),
  retrievalQuestion: document.getElementById("retrieval-question"),
  retrievalText: document.getElementById("retrieval-text"),
  claimsList: document.getElementById("claims-list"),
};

const setStatus = (text, isError = false) => {
  els.status.textContent = text || "";
  els.status.style.color = isError ? "#d64545" : "#4f5668";
};

els.fileInput.addEventListener("change", handleFileInput);

async function handleFileInput(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  setStatus(`Reading ${file.name} ...`);
  try {
    const fileContent = await file.text();
    const parsed = JSON.parse(fileContent);
    const normalized = normalizeRetrievals(parsed);

    if (!normalized.length) {
      throw new Error("No retrievals found in file");
    }

    const prepared = [];
    for (let idx = 0; idx < normalized.length; idx += 1) {
      const row = normalized[idx];
      setStatus(`Decoding retrieval ${idx + 1} of ${normalized.length} ...`);

      const tokensResponse = await decodeTokens(row.greedy_tokens || []);
      const tokenCount = tokensResponse.tokens.length;

      const { claims, tokenClaims } = buildClaims(
        row.claims || [],
        tokenCount,
        row.id
      );

      const tokens = tokensResponse.tokens.map((tok, tokenIdx) => ({
        ...tok,
        claims: tokenClaims[tokenIdx] || [],
      }));

      prepared.push({
        id: row.id,
        label: row.label,
        question: row.question || "",
        retrieval: row.retrieval || "",
        tokens,
        text: tokensResponse.text,
        claims,
      });
    }

    state.retrievals = prepared;
    state.selectedId = prepared[0]?.id ?? null;
    state.activeClaimId = null;
    render();
    setStatus(`Loaded ${prepared.length} retrievals.`);
  } catch (err) {
    console.error(err);
    setStatus(`Failed to load file: ${err.message}`, true);
    resetView();
  }
}

function normalizeRetrievals(raw) {
  const rows = [];
  if (Array.isArray(raw)) {
    raw.forEach((row, idx) => {
      rows.push({
        ...row,
        id: row.id ?? `ret-${idx + 1}`,
        label: row.label ?? `Retrieval ${idx + 1}`,
      });
    });
  } else if (raw && typeof raw === "object") {
    Object.entries(raw).forEach(([key, row], idx) => {
      rows.push({
        ...row,
        id: row.id ?? key ?? `ret-${idx + 1}`,
        label: row.label ?? `Retrieval ${key}`,
      });
    });
  }
  return rows;
}

async function decodeTokens(tokenIds) {
  const response = await fetch("/decode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token_ids: tokenIds }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Token decode failed");
  }

  return response.json();
}

function buildClaims(claimRows, tokenCount, retrievalId) {
  const tokenClaims = Array.from({ length: tokenCount }, () => []);
  const claims = [];

  claimRows.forEach((claim, idx) => {
    const claimId = `${retrievalId}-claim-${idx + 1}`;
    const aligned = Array.isArray(claim.aligned_token_ids)
      ? claim.aligned_token_ids
      : [];
    aligned.forEach((tokenIdx) => {
      if (Number.isInteger(tokenIdx) && tokenIdx >= 0 && tokenIdx < tokenCount) {
        tokenClaims[tokenIdx].push(claimId);
      }
    });

    claims.push({
      id: claimId,
      text: claim.claim_text || claim.decoded_claim || `Claim ${idx + 1}`,
      raw: claim,
      aligned_token_ids: aligned,
    });
  });

  return { claims, tokenClaims };
}

function render() {
  renderRetrievalList();
  renderMain();
}

function resetView() {
  state.retrievals = [];
  state.selectedId = null;
  state.activeClaimId = null;
  render();
}

function renderRetrievalList() {
  els.retrievalList.innerHTML = "";
  state.retrievals.forEach((item, idx) => {
    const li = document.createElement("li");
    li.className =
      "retrieval-item" + (item.id === state.selectedId ? " active" : "");
    li.dataset.id = item.id;

    const badge = document.createElement("span");
    badge.className = "retrieval-badge";
    badge.textContent = `${idx + 1}`;

    const label = document.createElement("span");
    label.className = "retrieval-label";
    label.textContent = item.label || `Retrieval ${idx + 1}`;

    li.appendChild(badge);
    li.appendChild(label);

    li.addEventListener("click", () => {
      state.selectedId = item.id;
      state.activeClaimId = null;
      renderMain();
      renderRetrievalList();
    });

    els.retrievalList.appendChild(li);
  });
}

function renderMain() {
  const selected = state.retrievals.find((r) => r.id === state.selectedId);
  if (!selected) {
    els.retrievalTitle.textContent = "No retrieval selected";
    els.retrievalQuestion.textContent = "";
    els.retrievalText.classList.add("empty-state");
    els.retrievalText.innerHTML =
      "<div class='placeholder'><p>Pick a JSON file to get started.</p></div>";
    els.claimsList.innerHTML = "";
    return;
  }

  els.retrievalTitle.textContent = selected.label || selected.id;
  els.retrievalQuestion.textContent = selected.question || "";

  renderRetrievalText(selected);
  renderClaims(selected);
}

function renderRetrievalText(selected) {
  els.retrievalText.classList.remove("empty-state");
  els.retrievalText.innerHTML = "";

  const tokens = selected.tokens || [];
  if (!tokens.length) {
    els.retrievalText.textContent = selected.retrieval || "No retrieval text.";
    return;
  }

  tokens.forEach((tok) => {
    const span = document.createElement("span");
    span.className = "token";
    if (tok.claims && tok.claims.length) {
      span.classList.add("has-claim");
      span.dataset.claims = tok.claims.join(" ");
    }
    span.textContent = tok.text;
    els.retrievalText.appendChild(span);
  });

  applyHighlighting();
}

function renderClaims(selected) {
  els.claimsList.innerHTML = "";
  if (!selected.claims.length) {
    const empty = document.createElement("li");
    empty.className = "claim-item";
    empty.textContent = "No claims found for this retrieval.";
    els.claimsList.appendChild(empty);
    return;
  }

  selected.claims.forEach((claim, idx) => {
    const li = document.createElement("li");
    li.className =
      "claim-item" + (claim.id === state.activeClaimId ? " active" : "");
    li.dataset.claimId = claim.id;

    const title = document.createElement("div");
    title.className = "claim-title";
    title.textContent = `Claim ${idx + 1}`;

    const text = document.createElement("div");
    text.className = "claim-text";
    text.textContent = claim.text;

    li.appendChild(title);
    li.appendChild(text);

    li.addEventListener("mouseenter", () => {
      state.activeClaimId = claim.id;
      renderActiveClaimStyles();
      applyHighlighting();
    });
    li.addEventListener("mouseleave", () => {
      state.activeClaimId = null;
      renderActiveClaimStyles();
      applyHighlighting();
    });

    els.claimsList.appendChild(li);
  });

  renderActiveClaimStyles();
}

function renderActiveClaimStyles() {
  const claimItems = els.claimsList.querySelectorAll(".claim-item");
  claimItems.forEach((item) => {
    const isActive = item.dataset.claimId === state.activeClaimId;
    item.classList.toggle("active", isActive);
  });
}

function applyHighlighting() {
  const spans = els.retrievalText.querySelectorAll(".token");
  spans.forEach((span) => {
    const claimIds = (span.dataset.claims || "")
      .split(" ")
      .filter(Boolean);
    const hasClaim = claimIds.length > 0;
    const matchesActive =
      hasClaim &&
      (!state.activeClaimId || claimIds.includes(state.activeClaimId));
    const shouldDim =
      state.activeClaimId &&
      hasClaim &&
      !claimIds.includes(state.activeClaimId);

    span.classList.toggle("highlight", matchesActive);
    span.classList.toggle("dim", Boolean(shouldDim));
  });
}
