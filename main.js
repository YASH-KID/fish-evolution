// main.js
// Wires up the UI controls and runs the animation loop. "Speed" controls how
// many fixed physics ticks run per rendered frame, so Max fast-forwards
// evolution itself rather than just uncapping the frame rate.
//
// This app is GPU-only: it waits for TensorFlow.js's WebGL backend to be
// ready before creating the world or starting the loop at all, and shows a
// loading/error state on the canvases in the meantime instead of falling
// back to any CPU path (there isn't one anymore).

const tankCanvas = document.getElementById('tankCanvas');
const tankCtx = tankCanvas.getContext('2d');
const nnCanvas = document.getElementById('nnCanvas');
const nnCtx = nnCanvas.getContext('2d');

tankCanvas.width = TANK_WIDTH;
tankCanvas.height = TANK_HEIGHT;

const genEl = document.getElementById('statGeneration');
const aliveEl = document.getElementById('statAlive');
const timerEl = document.getElementById('statTimer');
const bestEverEl = document.getElementById('statBestEver');
const lastGenEl = document.getElementById('statLastGen');
const followingEl = document.getElementById('statFollowing');
const backendEl = document.getElementById('statBackend');
const tickTimeEl = document.getElementById('statTickTime');

const speedButtons = {
  1: document.getElementById('speed1x'),
  5: document.getElementById('speed5x'),
  40: document.getElementById('speedMax'),
};
const restartBtn = document.getElementById('restartBtn');

let ticksPerFrame = 1;
let world = null;

Object.values(speedButtons).forEach((btn) => { btn.disabled = true; });
restartBtn.disabled = true;

function setSpeed(mode) {
  ticksPerFrame = mode;
  Object.entries(speedButtons).forEach(([key, btn]) => {
    btn.classList.toggle('active', Number(key) === mode);
  });
}

speedButtons[1].addEventListener('click', () => setSpeed(1));
speedButtons[5].addEventListener('click', () => setSpeed(5));
speedButtons[40].addEventListener('click', () => setSpeed(40));
restartBtn.addEventListener('click', () => { if (world) world.restart(); });

function formatTime(t) {
  return `${t.toFixed(1)}s`;
}

function updateStats() {
  genEl.textContent = world.generation + 1;
  aliveEl.textContent = `${world.aliveCount()} / ${POPULATION_SIZE}`;
  timerEl.textContent = formatTime(world.timer);
  bestEverEl.textContent = formatTime(world.bestEverTime);
  lastGenEl.textContent = world.generation > 0 ? formatTime(world.lastGenBestTime) : '—';
  const best = world.getBestFish();
  followingEl.textContent = best ? `Fish (fitness ${best.fitness.toFixed(1)}s)` : '—';
  backendEl.textContent = 'GPU (WebGL)';
  tickTimeEl.textContent = `${world.lastTickMs.toFixed(2)}ms`;
}

function drawStatusMessage(ctx, lines, color) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.fillStyle = '#02090f';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = color;
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';
  const lineHeight = 24;
  const startY = h / 2 - ((lines.length - 1) * lineHeight) / 2;
  lines.forEach((line, i) => ctx.fillText(line, w / 2, startY + i * lineHeight));
}

function frame() {
  const generationBefore = world.generation;
  for (let i = 0; i < ticksPerFrame; i++) {
    const isLastTick = i === ticksPerFrame - 1;
    world.step(FIXED_DT, isLastTick);
    // If a generation boundary happened mid-batch, stop the batch here so we
    // still render at least once per completed generation (keeps the UI legible
    // even at Max speed instead of skipping straight past several generations).
    if (world.generation !== generationBefore) break;
  }

  renderTank(tankCtx, world);
  renderNN(nnCtx, world.getBestFish());
  updateStats();

  requestAnimationFrame(frame);
}

async function start() {
  drawStatusMessage(tankCtx, ['Loading GPU (WebGL)…'], '#7fa3b8');
  drawStatusMessage(nnCtx, ['Loading GPU (WebGL)…'], '#7fa3b8');

  const timeoutMs = 15000;
  const pollMs = 200;
  let waited = 0;
  while (!GpuBrainBatch.isAvailable() && waited < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, pollMs));
    waited += pollMs;
  }

  if (!GpuBrainBatch.isAvailable()) {
    const message = [
      'GPU (WebGL) unavailable.',
      'This app requires GPU support via TensorFlow.js,',
      'which loads from a CDN - check your internet',
      'connection and reload.',
    ];
    drawStatusMessage(tankCtx, message, '#ff8f8f');
    drawStatusMessage(nnCtx, message, '#ff8f8f');
    return;
  }

  await tf.setBackend('webgl');
  await tf.ready();

  world = new World();
  setSpeed(1);
  Object.values(speedButtons).forEach((btn) => { btn.disabled = false; });
  restartBtn.disabled = false;

  requestAnimationFrame(frame);
}

start();
