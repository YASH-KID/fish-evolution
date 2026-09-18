// render.js
// Draws the two canvases: the tank (fish + shark simulation) and the
// live neural-network diagram for whichever fish is currently doing best.
// Dark-theme only.

const INPUT_LABELS = [
  'X', 'Y', 'Speed', 'Accel', 'Angle',
  'Near?', 'NearBear', 'NearX', 'NearY',
  'Far?', 'FarBear', 'FarX', 'FarY',
];
const OUTPUT_LABELS = ['Turn', 'Thrust', 'Burst', 'Juke'];

const FISH_HUES = [];
for (let i = 0; i < POPULATION_SIZE; i++) FISH_HUES.push(Math.round((360 / POPULATION_SIZE) * i));

function renderTank(ctx, world) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  // Deep, dark ocean background
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, '#0a1f2e');
  grad.addColorStop(1, '#02090f');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Tank glass border
  ctx.strokeStyle = '#1f4a63';
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, w - 4, h - 4);

  const best = world.getBestFish();

  // FOV cone for the best fish, drawn under everything else
  if (best && best.alive) {
    ctx.save();
    ctx.translate(best.x, best.y);
    ctx.rotate(best.angle);
    ctx.fillStyle = best.sawShark ? 'rgba(255, 90, 90, 0.18)' : 'rgba(120, 200, 255, 0.12)';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    const range = 260;
    ctx.arc(0, 0, range, -FOV_HALF_ANGLE, FOV_HALF_ANGLE);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Fish
  world.fish.forEach((f, i) => {
    if (!f.alive) return;
    const isBest = best === f;
    drawFish(ctx, f, FISH_HUES[i % FISH_HUES.length], isBest);
  });

  // Sharks
  world.sharks.forEach((shark) => drawShark(ctx, shark));
}

function drawFish(ctx, fish, hue, isBest) {
  ctx.save();
  ctx.translate(fish.x, fish.y);
  ctx.rotate(fish.angle);

  const len = 15; // nose-to-tail-base length
  const bodyColor = `hsl(${hue}, 75%, 58%)`;
  const finColor = `hsl(${hue}, 65%, 40%)`;

  // Burst-dash trail: a soft glow behind the fish when its "Burst" output fires
  if (fish.burstAmount > 0.15) {
    ctx.save();
    ctx.globalAlpha = Math.min(0.55, fish.burstAmount * 0.6);
    ctx.fillStyle = '#9fe8ff';
    ctx.beginPath();
    ctx.ellipse(-len * 0.9, 0, len * 0.9 * fish.burstAmount, len * 0.32 * fish.burstAmount, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Tail fin (forked), drawn first so the body overlaps its base
  ctx.fillStyle = finColor;
  ctx.beginPath();
  ctx.moveTo(-len * 0.55, 0);
  ctx.lineTo(-len * 1.15, -len * 0.5);
  ctx.lineTo(-len * 0.75, 0);
  ctx.lineTo(-len * 1.15, len * 0.5);
  ctx.closePath();
  ctx.fill();

  // Body: teardrop, tapering from a rounded head to a narrow tail base
  ctx.fillStyle = bodyColor;
  ctx.beginPath();
  ctx.moveTo(len * 0.9, 0);
  ctx.bezierCurveTo(len * 0.7, -len * 0.55, len * 0.1, -len * 0.5, -len * 0.55, -len * 0.12);
  ctx.bezierCurveTo(-len * 0.55, -len * 0.12, -len * 0.55, len * 0.12, -len * 0.55, len * 0.12);
  ctx.bezierCurveTo(len * 0.1, len * 0.5, len * 0.7, len * 0.55, len * 0.9, 0);
  ctx.closePath();
  ctx.fill();

  // Dorsal fin
  ctx.fillStyle = finColor;
  ctx.beginPath();
  ctx.moveTo(0, -len * 0.28);
  ctx.lineTo(len * 0.12, -len * 0.62);
  ctx.lineTo(len * 0.32, -len * 0.24);
  ctx.closePath();
  ctx.fill();

  // Eye
  ctx.fillStyle = '#0c1418';
  ctx.beginPath();
  ctx.arc(len * 0.55, -len * 0.08, 1.6, 0, Math.PI * 2);
  ctx.fill();

  if (isBest) {
    ctx.strokeStyle = '#ffd400';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(0, 0, len * 1.05, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.restore();
}

function drawShark(ctx, shark) {
  ctx.save();
  ctx.translate(shark.x, shark.y);
  ctx.rotate(shark.angle);

  const len = 34; // nose-to-tail-base length
  const bodyColor = '#8a9aa8';
  const bellyColor = '#c7d2da';
  const finColor = '#5c6b76';

  // Tail fin (crescent, asymmetric like a real shark's heterocercal tail)
  ctx.fillStyle = finColor;
  ctx.beginPath();
  ctx.moveTo(-len * 0.62, 0);
  ctx.quadraticCurveTo(-len * 1.15, -len * 0.55, -len * 1.35, -len * 0.15);
  ctx.quadraticCurveTo(-len * 0.95, -len * 0.05, -len * 0.8, 0);
  ctx.quadraticCurveTo(-len * 1.05, len * 0.32, -len * 1.2, len * 0.5);
  ctx.quadraticCurveTo(-len * 0.75, len * 0.15, -len * 0.62, 0);
  ctx.closePath();
  ctx.fill();

  // Body: torpedo shape, pointed nose, tapering to the tail
  ctx.fillStyle = bodyColor;
  ctx.beginPath();
  ctx.moveTo(len * 0.98, 0);
  ctx.bezierCurveTo(len * 0.85, -len * 0.3, len * 0.35, -len * 0.42, -len * 0.15, -len * 0.28);
  ctx.bezierCurveTo(-len * 0.4, -len * 0.22, -len * 0.6, -len * 0.1, -len * 0.62, 0);
  ctx.bezierCurveTo(-len * 0.6, len * 0.1, -len * 0.4, len * 0.22, -len * 0.15, len * 0.28);
  ctx.bezierCurveTo(len * 0.35, len * 0.42, len * 0.85, len * 0.3, len * 0.98, 0);
  ctx.closePath();
  ctx.fill();

  // Pale belly stripe
  ctx.fillStyle = bellyColor;
  ctx.beginPath();
  ctx.moveTo(len * 0.85, len * 0.05);
  ctx.bezierCurveTo(len * 0.3, len * 0.28, -len * 0.1, len * 0.24, -len * 0.55, len * 0.06);
  ctx.bezierCurveTo(-len * 0.3, len * 0.14, len * 0.2, len * 0.18, len * 0.85, len * 0.05);
  ctx.closePath();
  ctx.fill();

  // Pectoral fins
  ctx.fillStyle = finColor;
  ctx.beginPath();
  ctx.moveTo(len * 0.18, len * 0.18);
  ctx.lineTo(len * 0.02, len * 0.52);
  ctx.lineTo(len * 0.32, len * 0.24);
  ctx.closePath();
  ctx.fill();

  // Dorsal fin
  ctx.fillStyle = finColor;
  ctx.beginPath();
  ctx.moveTo(len * 0.08, -len * 0.26);
  ctx.lineTo(len * 0.2, -len * 0.72);
  ctx.lineTo(len * 0.42, -len * 0.22);
  ctx.closePath();
  ctx.fill();

  // Gill slits
  ctx.strokeStyle = '#465560';
  ctx.lineWidth = 1;
  for (let i = 0; i < 3; i++) {
    const gx = len * 0.5 - i * 5;
    ctx.beginPath();
    ctx.moveTo(gx, -len * 0.2);
    ctx.lineTo(gx - 3, -len * 0.02);
    ctx.stroke();
  }

  // Eye
  ctx.fillStyle = '#0c1418';
  ctx.beginPath();
  ctx.arc(len * 0.72, -len * 0.08, 2.4, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function activationColor(v) {
  // v in [-1, 1]: cyan (negative) -> dark neutral (zero) -> orange (positive)
  const clamped = Math.max(-1, Math.min(1, v));
  const neutral = [42, 58, 74]; // dark blue-grey, matches the panel background family
  if (clamped >= 0) {
    const t = clamped;
    const target = [255, 157, 64]; // orange
    return lerpColor(neutral, target, t);
  } else {
    const t = -clamped;
    const target = [56, 209, 224]; // cyan
    return lerpColor(neutral, target, t);
  }
}

function lerpColor(a, b, t) {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

function renderNN(ctx, fish) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0d151c';
  ctx.fillRect(0, 0, w, h);

  if (!fish) {
    ctx.fillStyle = '#7fa3b8';
    ctx.font = '14px sans-serif';
    ctx.fillText('No fish to display yet', 20, 30);
    return;
  }

  const brain = fish.brain;
  const layerSizes = brain.layerSizes;
  const activations = brain.activations;
  const numLayers = layerSizes.length;

  const leftPad = 60;
  const rightPad = 60;
  const colX = layerSizes.map((_, l) => leftPad + ((w - leftPad - rightPad) * l) / (numLayers - 1));
  const colY = layerSizes.map((n) => layoutY(h, n, 30));

  // Edges between every consecutive pair of layers
  for (let l = 0; l < numLayers - 1; l++) {
    const W = brain.weights[l];
    for (let i = 0; i < layerSizes[l + 1]; i++) {
      for (let j = 0; j < layerSizes[l]; j++) {
        drawEdge(ctx, colX[l], colY[l][j], colX[l + 1], colY[l + 1][i], W[i][j]);
      }
    }
  }

  // Nodes
  for (let l = 0; l < numLayers; l++) {
    const isInput = l === 0;
    const isOutput = l === numLayers - 1;
    for (let i = 0; i < layerSizes[l]; i++) {
      let label = '';
      let side = 'center';
      if (isInput) { label = INPUT_LABELS[i] || ''; side = 'left'; }
      else if (isOutput) { label = OUTPUT_LABELS[i] || ''; side = 'right'; }
      drawNode(ctx, colX[l], colY[l][i], activations[l][i], label, side);
    }
  }
}

function layoutY(h, count, topPad) {
  const usable = h - topPad * 2;
  const ys = [];
  for (let i = 0; i < count; i++) {
    ys.push(topPad + (usable * (i + 0.5)) / count);
  }
  return ys;
}

function drawEdge(ctx, x1, y1, x2, y2, weight) {
  const w = Math.max(0.4, Math.min(3.5, Math.abs(weight) * 2.2));
  ctx.strokeStyle = weight >= 0 ? `rgba(64, 156, 255, ${clampAlpha(weight)})` : `rgba(255, 90, 90, ${clampAlpha(-weight)})`;
  ctx.lineWidth = w;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function clampAlpha(v) {
  return Math.max(0.1, Math.min(0.8, Math.abs(v) * 0.45 + 0.12));
}

function drawNode(ctx, x, y, activation, label, labelSide) {
  const radius = 10;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = activationColor(activation);
  ctx.fill();
  ctx.strokeStyle = '#4a5f70';
  ctx.lineWidth = 1.2;
  ctx.stroke();

  if (label) {
    ctx.fillStyle = '#cfe3ee';
    ctx.font = '11px sans-serif';
    if (labelSide === 'left') {
      ctx.textAlign = 'right';
      ctx.fillText(label, x - radius - 6, y + 4);
    } else if (labelSide === 'right') {
      ctx.textAlign = 'left';
      ctx.fillText(label, x + radius + 6, y + 4);
    } else {
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y - radius - 6);
    }
  }
}
