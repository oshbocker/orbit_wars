// ── Orbit Wars interactive browser client ──────────────────────────────────

const CANVAS_SIZE = 600;
const BOARD_SIZE = 100;
const SCALE = CANVAS_SIZE / BOARD_SIZE;
const SUN_X = 50, SUN_Y = 50, SUN_RADIUS = 10;
const SUN_SAFE_RADIUS = SUN_RADIUS + 2;

// Wong colorblind-safe palette
const COLORS = {
  player: '#4488ff',   // blue (you)
  opponent: '#ee8833', // orange
  neutral: '#888888',
  sun: '#ffcc00',
  bg: '#0a0a1a',
  grid: '#151530',
};

function ownerColor(owner, myPlayer) {
  if (owner === myPlayer) return COLORS.player;
  if (owner === -1) return COLORS.neutral;
  return COLORS.opponent;
}

// ── Sun collision check ────────────────────────────────────────────────────
function passesThroughSun(x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const fx = x1 - SUN_X, fy = y1 - SUN_Y;
  const a = dx * dx + dy * dy;
  if (a === 0) return false;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - SUN_SAFE_RADIUS * SUN_SAFE_RADIUS;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return false;
  const sq = Math.sqrt(disc);
  const t1 = (-b - sq) / (2 * a);
  const t2 = (-b + sq) / (2 * a);
  return (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1) || (t1 < 0 && t2 > 0);
}

// ── State ──────────────────────────────────────────────────────────────────
let state = null;       // current observation
let myPlayer = 0;
let selectedPlanet = null;   // planet id or null
let targetPlanet = null;     // planet id or null (awaiting ship count)
let targetCoords = null;     // {x, y} board coords of aim point
let queuedMoves = [];        // [{fromId, toId, targetX, targetY, angle, ships, sunWarn}]
let shipsCommitted = {};     // {planetId: ships already queued}
let gameActive = false;
let mouseX = -1, mouseY = -1;  // board coords

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ── Helpers ────────────────────────────────────────────────────────────────
function planetById(id) {
  if (!state) return null;
  for (const p of state.planets) {
    if (p[0] === id) return p;
  }
  return null;
}

function availableShips(planetId) {
  const p = planetById(planetId);
  if (!p) return 0;
  const committed = shipsCommitted[planetId] || 0;
  return Math.max(0, Math.floor(p[5]) - committed);
}

function computeScores() {
  if (!state) return { you: 0, opp: 0 };
  let you = 0, opp = 0;
  for (const p of state.planets) {
    if (p[1] === myPlayer) you += Math.floor(p[5]);
    else if (p[1] >= 0) opp += Math.floor(p[5]);
  }
  for (const f of (state.fleets || [])) {
    if (f[1] === myPlayer) you += Math.floor(f[6]);
    else opp += Math.floor(f[6]);
  }
  return { you, opp };
}

// ── Drawing ────────────────────────────────────────────────────────────────
function toCanvas(x, y) {
  return [x * SCALE, y * SCALE];
}

function draw() {
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= BOARD_SIZE; i += 10) {
    const c = i * SCALE;
    ctx.beginPath(); ctx.moveTo(c, 0); ctx.lineTo(c, CANVAS_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, c); ctx.lineTo(CANVAS_SIZE, c); ctx.stroke();
  }

  if (!state) return;

  // Sun with glow
  const [sx, sy] = toCanvas(SUN_X, SUN_Y);
  const sr = SUN_RADIUS * SCALE;
  const grad = ctx.createRadialGradient(sx, sy, sr * 0.2, sx, sy, sr * 1.5);
  grad.addColorStop(0, '#ffee88');
  grad.addColorStop(0.5, '#ffcc0066');
  grad.addColorStop(1, 'transparent');
  ctx.fillStyle = grad;
  ctx.fillRect(sx - sr * 1.5, sy - sr * 1.5, sr * 3, sr * 3);
  ctx.beginPath();
  ctx.arc(sx, sy, sr, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.sun;
  ctx.fill();

  // Planets
  for (const p of state.planets) {
    const [px, py] = toCanvas(p[2], p[3]);
    const pr = Math.max(p[4] * SCALE, 3);
    const color = ownerColor(p[1], myPlayer);

    // Selection glow
    if (p[0] === selectedPlanet) {
      ctx.beginPath();
      ctx.arc(px, py, pr + 4, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff88';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (targetPlanet !== null && p[0] === targetPlanet) {
      ctx.beginPath();
      ctx.arc(px, py, pr + 4, 0, Math.PI * 2);
      ctx.strokeStyle = '#ff888888';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Ship count label
    const ships = Math.floor(p[5]);
    ctx.fillStyle = '#fff';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(ships, px, py - pr - 4);

    // Production pips
    const prod = p[6];
    if (prod > 0) {
      ctx.fillStyle = '#ffffff88';
      ctx.font = '8px monospace';
      ctx.fillText('P' + prod, px, py + pr + 10);
    }
  }

  // Free-aim crosshair
  if (targetCoords !== null && targetPlanet === null) {
    const [cx, cy] = toCanvas(targetCoords.x, targetCoords.y);
    ctx.strokeStyle = '#ff888888';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();
  }

  // Fleets as chevrons
  for (const f of (state.fleets || [])) {
    const [fx, fy] = toCanvas(f[2], f[3]);
    const angle = f[4];
    const color = ownerColor(f[1], myPlayer);
    const size = Math.min(4 + Math.log(f[6] + 1), 10);

    ctx.save();
    ctx.translate(fx, fy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, -size * 0.5);
    ctx.lineTo(-size * 0.3, 0);
    ctx.lineTo(-size * 0.6, size * 0.5);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();

    // Fleet ship count
    if (f[6] >= 5) {
      ctx.fillStyle = '#ffffffaa';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(Math.floor(f[6]), fx, fy - 8);
    }
  }

  // Queued move arrows
  for (const m of queuedMoves) {
    const fromP = planetById(m.fromId);
    if (!fromP) continue;
    const [x1, y1] = toCanvas(fromP[2], fromP[3]);
    const [x2, y2] = toCanvas(m.targetX, m.targetY);

    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = m.sunWarn ? '#ff4444aa' : '#44ff44aa';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow head
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(a - 0.4), y2 - 8 * Math.sin(a - 0.4));
    ctx.lineTo(x2 - 8 * Math.cos(a + 0.4), y2 - 8 * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fillStyle = m.sunWarn ? '#ff4444aa' : '#44ff44aa';
    ctx.fill();

    // Ship count on arrow
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    ctx.fillStyle = '#fff';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(m.ships, mx, my - 6);
  }

  // Live aim line from selected planet to cursor
  if (selectedPlanet !== null && targetPlanet === null && mouseX >= 0) {
    const fromP = planetById(selectedPlanet);
    if (fromP) {
      const [x1, y1] = toCanvas(fromP[2], fromP[3]);
      const [x2, y2] = toCanvas(mouseX, mouseY);
      const sunWarn = passesThroughSun(fromP[2], fromP[3], mouseX, mouseY);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = sunWarn ? '#ff444466' : '#ffffff33';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

// ── UI updates ─────────────────────────────────────────────────────────────
function updateUI() {
  const scores = computeScores();
  document.getElementById('score-you').textContent = scores.you;
  document.getElementById('score-opp').textContent = scores.opp;
  document.getElementById('step-display').textContent =
    `Step: ${state ? state.step : 0} / 498`;

  document.getElementById('btn-step').disabled = !gameActive;

  // Move queue list
  const ul = document.getElementById('move-queue');
  ul.innerHTML = '';
  for (let i = 0; i < queuedMoves.length; i++) {
    const m = queuedMoves[i];
    const li = document.createElement('li');
    if (m.sunWarn) li.className = 'sun-warning';
    const fromP = planetById(m.fromId);
    const fromName = fromP ? `P${m.fromId}` : '?';
    const toName = m.toId !== null ? `P${m.toId}` : `(${m.targetX.toFixed(0)},${m.targetY.toFixed(0)})`;
    li.innerHTML = `<span>${fromName} → ${toName}: ${m.ships} ships${m.sunWarn ? ' ⚠' : ''}</span>`;
    const btn = document.createElement('button');
    btn.className = 'remove-btn';
    btn.textContent = '✕';
    btn.onclick = () => removeMove(i);
    li.appendChild(btn);
    ul.appendChild(li);
  }

  draw();
}

function setPrompt(text) {
  document.getElementById('prompt').textContent = text;
}

// ── Interaction ────────────────────────────────────────────────────────────
function removeMove(index) {
  const m = queuedMoves[index];
  shipsCommitted[m.fromId] = (shipsCommitted[m.fromId] || 0) - m.ships;
  if (shipsCommitted[m.fromId] <= 0) delete shipsCommitted[m.fromId];
  queuedMoves.splice(index, 1);
  updateUI();
}

function clearSelection() {
  selectedPlanet = null;
  targetPlanet = null;
  targetCoords = null;
  document.getElementById('ship-input').classList.add('hidden');
  setPrompt(gameActive ? 'Select a planet or click Step to advance.' : '');
}

function handleCanvasClick(e) {
  if (!gameActive || !state) return;

  const rect = canvas.getBoundingClientRect();
  const bx = (e.clientX - rect.left) / SCALE;
  const by = (e.clientY - rect.top) / SCALE;

  // Find clicked planet (within radius + tolerance)
  let clicked = null;
  for (const p of state.planets) {
    const dx = p[2] - bx, dy = p[3] - by;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < p[4] + 1.5) {
      clicked = p;
      break;
    }
  }

  if (selectedPlanet === null) {
    // No selection yet — select own planet
    if (clicked && clicked[1] === myPlayer) {
      const avail = availableShips(clicked[0]);
      if (avail <= 0) {
        setPrompt(`Planet ${clicked[0]} has no ships available.`);
        return;
      }
      selectedPlanet = clicked[0];
      targetPlanet = null;
      document.getElementById('ship-input').classList.add('hidden');
      setPrompt(`Selected P${clicked[0]} (${avail} ships). Click a target.`);
      updateUI();
    }
  } else if (targetPlanet === null && targetCoords === null) {
    // Have source — pick target (planet or empty space for free-aim)
    if (clicked && clicked[0] === selectedPlanet) {
      // Clicked same planet — deselect
      clearSelection();
      setPrompt('Selection cancelled. Click a planet to select.');
      updateUI();
      return;
    }

    // Set target: planet center or raw click coordinates
    if (clicked) {
      targetPlanet = clicked[0];
      targetCoords = { x: clicked[2], y: clicked[3] };
    } else {
      // Free-aim at empty space (lead orbiting planets, etc.)
      targetPlanet = null;
      targetCoords = { x: bx, y: by };
    }

    const avail = availableShips(selectedPlanet);
    if (avail <= 0) {
      setPrompt(`No ships left on P${selectedPlanet}.`);
      clearSelection();
      updateUI();
      return;
    }
    const shipInput = document.getElementById('ship-count');
    shipInput.max = avail;
    shipInput.value = avail;
    document.getElementById('ship-max').textContent = `/ ${avail}`;
    const targetLabel = targetPlanet !== null
      ? `P${targetPlanet}`
      : `(${bx.toFixed(1)}, ${by.toFixed(1)})`;
    setPrompt(`P${selectedPlanet} → ${targetLabel}. Enter ship count.`);
    document.getElementById('ship-input').classList.remove('hidden');
    shipInput.focus();
    shipInput.select();
    updateUI();
  }
}

function handleSendShips() {
  if (selectedPlanet === null || targetCoords === null) return;

  const fromP = planetById(selectedPlanet);
  if (!fromP) { clearSelection(); return; }

  const avail = availableShips(selectedPlanet);
  const raw = parseInt(document.getElementById('ship-count').value, 10);
  const ships = Math.min(Math.max(1, raw || 0), avail);
  if (ships <= 0 || isNaN(raw)) {
    setPrompt('Enter a valid ship count.');
    return;
  }

  const tx = targetCoords.x, ty = targetCoords.y;
  const angle = Math.atan2(ty - fromP[3], tx - fromP[2]);
  const sunWarn = passesThroughSun(fromP[2], fromP[3], tx, ty);

  queuedMoves.push({
    fromId: selectedPlanet,
    toId: targetPlanet,
    targetX: tx,
    targetY: ty,
    angle,
    ships,
    sunWarn,
  });
  shipsCommitted[selectedPlanet] = (shipsCommitted[selectedPlanet] || 0) + ships;

  clearSelection();
  setPrompt('Move queued. Select another planet or click Step.');
  updateUI();
}

async function doStep() {
  if (!gameActive) return;

  const action = queuedMoves.map(m => [m.fromId, m.angle, m.ships]);

  // Clear queue
  queuedMoves = [];
  shipsCommitted = {};
  clearSelection();
  setPrompt('Processing...');
  document.getElementById('btn-step').disabled = true;
  updateUI();

  try {
    const resp = await fetch('/api/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    const data = await resp.json();

    if (data.error) {
      setPrompt('Error: ' + data.error);
      return;
    }

    state = data.observation;
    updateUI();

    if (data.done) {
      gameActive = false;
      const won = data.reward > 0;
      const text = won ? 'You won!' : (data.reward === 0 ? 'Draw!' : 'You lost.');
      document.getElementById('game-over-text').textContent = text;
      document.getElementById('game-over-text').style.color = won ? '#44ff44' : '#ff4444';
      document.getElementById('game-over').classList.remove('hidden');
      document.getElementById('btn-step').disabled = true;
      setPrompt('Game over.');
    } else {
      setPrompt('Select a planet or click Step to advance.');
    }
  } catch (err) {
    setPrompt('Network error: ' + err.message);
  }
}

async function newGame() {
  document.getElementById('game-over').classList.add('hidden');
  setPrompt('Starting new game...');

  try {
    const resp = await fetch('/api/new-game', { method: 'POST' });
    const data = await resp.json();
    state = data.observation;
    myPlayer = state.player || 0;
    queuedMoves = [];
    shipsCommitted = {};
    selectedPlanet = null;
    targetPlanet = null;
    targetCoords = null;
    gameActive = true;
    document.getElementById('ship-input').classList.add('hidden');
    setPrompt('Select a planet or click Step to advance.');
    updateUI();
  } catch (err) {
    setPrompt('Failed to start game: ' + err.message);
  }
}

function exportLog() {
  window.open('/api/export', '_blank');
}

// ── Event listeners ────────────────────────────────────────────────────────
canvas.addEventListener('click', handleCanvasClick);
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouseX = (e.clientX - rect.left) / SCALE;
  mouseY = (e.clientY - rect.top) / SCALE;
  if (selectedPlanet !== null && targetPlanet === null) draw();
});
canvas.addEventListener('mouseleave', () => {
  mouseX = -1; mouseY = -1;
  draw();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    clearSelection();
    updateUI();
  }
});

document.getElementById('btn-step').addEventListener('click', doStep);
document.getElementById('btn-new-game').addEventListener('click', newGame);
document.getElementById('btn-play-again').addEventListener('click', newGame);
document.getElementById('btn-export').addEventListener('click', exportLog);
document.getElementById('btn-send').addEventListener('click', handleSendShips);

// Enter key in ship input sends the fleet
document.getElementById('ship-count').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleSendShips();
});

// Initial draw
draw();
