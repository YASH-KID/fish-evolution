// simulation.js
// Physics, sensing, the hand-coded shark AI, and the genetic algorithm that
// breeds the next generation of fish brains.

const TANK_WIDTH = 1100;
const TANK_HEIGHT = 680;
const FIXED_DT = 1 / 60; // one simulation tick, in seconds

const FISH_MAX_SPEED = 140; // px/s
const FISH_MAX_ACCEL = 320; // px/s^2
const FISH_MAX_TURN_RATE = 4.5; // rad/s (very agile - fish out-turn the shark)
const FISH_RADIUS = 8;

const BURST_MAX_DURATION = 2; // seconds burst can be held before it forces a cooldown
const BURST_COOLDOWN = 3.5; // seconds burst is locked out afterward (within the requested 3-4s range)

const SHARK_SPEED = 180; // px/s (faster in a straight line than the fish)
const SHARK_MAX_TURN_RATE = 6.5; // rad/s - sharply more agile than the fish now
const SHARK_BURST_TURN_RATE = SHARK_MAX_TURN_RATE * 1.8; // temporary boost used to break out of an orbit
const SHARK_RADIUS = 15;
const STUCK_WINDOW = 90; // ~1.5s of samples at 60 ticks/sec
const STUCK_DISTANCE_SPREAD = 14; // px - if distance to target barely changes over the window, we're circling

const EAT_DISTANCE = FISH_RADIUS + SHARK_RADIUS + 3;
const FOV_HALF_ANGLE = (80 * Math.PI) / 180; // 80 degrees each side of straight ahead
const MAX_GEN_TIME = 60; // seconds of sim-time before a generation is forced to end
const POPULATION_SIZE = 40;
const ELITE_COUNT = 6;
const WALL_MARGIN = 20;
const SHARK_COUNT = 2;

function angleDiff(a, b) {
  // Smallest signed difference from angle a to angle b, in [-PI, PI]
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

class Fish {
  constructor(brain, x, y) {
    this.brain = brain;
    this.x = x;
    this.y = y;
    this.angle = Math.random() * Math.PI * 2;
    this.speed = FISH_MAX_SPEED * 0.4;
    this.acceleration = 0;
    this.alive = true;
    this.fitness = 0; // seconds survived
    this.sawShark = false;
    this.burstAmount = 0;
    this.jukeAmount = 0;
    this.burstHeldTime = 0; // consecutive seconds burst has been actively held
    this.burstCooldown = 0; // seconds remaining before burst can be used again
  }

  // sharks: array of all Shark instances. Each contributes 4 inputs (visible,
  // bearing, X, Y), ordered nearest-first, so the fish always gets "closest
  // threat" info in the same input slots regardless of which shark object
  // happens to be closer at any given moment.
  sense(sharks) {
    const posX = this.x / TANK_WIDTH;
    const posY = this.y / TANK_HEIGHT;
    const speedNorm = this.speed / FISH_MAX_SPEED;
    const accelNorm = this.acceleration / FISH_MAX_ACCEL;
    const angleNorm = this.angle / Math.PI; // roughly -1..1

    const sharkInfos = sharks
      .map((shark) => {
        const dx = shark.x - this.x;
        const dy = shark.y - this.y;
        const dist = Math.hypot(dx, dy);
        const angleToShark = Math.atan2(dy, dx);
        const bearing = angleDiff(this.angle, angleToShark); // -PI..PI relative to facing
        const visible = dist > 0 && Math.abs(bearing) <= FOV_HALF_ANGLE;
        return {
          visible,
          bearingNorm: visible ? bearing / Math.PI : 0,
          // Always-known shark position (like a fish's lateral line sensing
          // water displacement) - unlike visible/bearing, not gated by FOV.
          xNorm: shark.x / TANK_WIDTH,
          yNorm: shark.y / TANK_HEIGHT,
          dist,
        };
      })
      .sort((a, b) => a.dist - b.dist);

    this.sawShark = sharkInfos.some((s) => s.visible);

    const inputs = [posX, posY, speedNorm, accelNorm, angleNorm];
    for (const info of sharkInfos) {
      inputs.push(info.visible ? 1 : 0, info.bearingNorm, info.xNorm, info.yNorm);
    }
    return inputs;
  }

  // Physics integration given a [turn, thrust, burst, juke] output vector.
  // Outputs always come from the GPU batched forward pass (see World.step) -
  // there is no CPU forward-pass path anymore, this app runs on GPU only.
  applyOutputs(outputs, dt) {
    if (!this.alive) return;
    const [turnOut, thrustOut, burstOut, jukeOut] = outputs;

    // Burst: a full-speed sprint (faster than the shark in a straight line),
    // but only for BURST_MAX_DURATION seconds of continuous use before it locks
    // out for BURST_COOLDOWN seconds - a fish can't just hold it forever.
    const burstRequested = burstOut > 0;
    if (this.burstCooldown > 0) {
      this.burstCooldown = Math.max(0, this.burstCooldown - dt);
      this.burstAmount = 0;
    } else if (burstRequested) {
      this.burstAmount = burstOut;
      this.burstHeldTime += dt;
      if (this.burstHeldTime >= BURST_MAX_DURATION) {
        this.burstCooldown = BURST_COOLDOWN;
        this.burstHeldTime = 0;
        this.burstAmount = 0;
      }
    } else {
      this.burstAmount = 0;
      this.burstHeldTime = 0; // letting go resets the meter
    }

    this.jukeAmount = Math.max(0, jukeOut);

    const effectiveTurnRate = FISH_MAX_TURN_RATE * (1 + this.jukeAmount * 1.5) * (1 - this.burstAmount * 0.4);
    this.angle += turnOut * effectiveTurnRate * dt;

    this.acceleration = thrustOut * FISH_MAX_ACCEL;
    // Burst pushes top speed to 224 px/s - a bit faster than the shark's 180 px/s -
    // but the duration/cooldown above is what keeps it from being a free escape.
    const effectiveMaxSpeed = FISH_MAX_SPEED * (1 + this.burstAmount * 0.6);
    let newSpeed = clamp(this.speed + this.acceleration * dt, 0, effectiveMaxSpeed);
    newSpeed *= 1 - this.jukeAmount * 0.5; // a hard juke bleeds off momentum
    this.speed = newSpeed;

    let nextX = this.x + Math.cos(this.angle) * this.speed * dt;
    let nextY = this.y + Math.sin(this.angle) * this.speed * dt;

    // Soft wall bump: clamp position and bleed off speed instead of a hard bounce.
    let bumped = false;
    if (nextX < WALL_MARGIN) { nextX = WALL_MARGIN; bumped = true; }
    if (nextX > TANK_WIDTH - WALL_MARGIN) { nextX = TANK_WIDTH - WALL_MARGIN; bumped = true; }
    if (nextY < WALL_MARGIN) { nextY = WALL_MARGIN; bumped = true; }
    if (nextY > TANK_HEIGHT - WALL_MARGIN) { nextY = TANK_HEIGHT - WALL_MARGIN; bumped = true; }
    if (bumped) this.speed *= 0.5;

    this.x = nextX;
    this.y = nextY;

    this.fitness += dt;
  }
}

class Shark {
  constructor() {
    this.x = TANK_WIDTH / 2;
    this.y = TANK_HEIGHT / 2;
    this.angle = Math.random() * Math.PI * 2;
    this.targetId = null;
    this.distSamples = [];
  }

  // True intercept solve: aim not at the target's current spot, but at the
  // point where a straight run at SHARK_SPEED will actually meet it, given
  // the target's current heading and speed. This is what stops the classic
  // "always turn toward where you are right now" pursuit from settling into
  // an endless orbit around a target that's also turning.
  interceptAngle(target) {
    const relX = target.x - this.x;
    const relY = target.y - this.y;
    const tvx = Math.cos(target.angle) * target.speed;
    const tvy = Math.sin(target.angle) * target.speed;

    const a = tvx * tvx + tvy * tvy - SHARK_SPEED * SHARK_SPEED;
    const b = 2 * (relX * tvx + relY * tvy);
    const c = relX * relX + relY * relY;

    let t = 0;
    if (Math.abs(a) < 1e-6) {
      if (Math.abs(b) > 1e-6) t = -c / b;
    } else {
      const discriminant = b * b - 4 * a * c;
      if (discriminant >= 0) {
        const sqrtDisc = Math.sqrt(discriminant);
        const t1 = (-b + sqrtDisc) / (2 * a);
        const t2 = (-b - sqrtDisc) / (2 * a);
        const candidates = [t1, t2].filter((v) => v > 0 && Number.isFinite(v));
        if (candidates.length) t = Math.min(...candidates);
      }
    }
    t = clamp(t, 0, 2.5);

    const ix = target.x + tvx * t;
    const iy = target.y + tvy * t;
    return Math.atan2(iy - this.y, ix - this.x);
  }

  update(fishList, dt) {
    // Hand-coded pursuit brain (not ML): chase the most catchable living fish -
    // weighing both how close it is AND how fast it's currently moving, so the
    // shark opportunistically picks off a slow fish over a merely-nearer fast one.
    let nearest = null;
    let nearestDist = Infinity;
    let bestScore = Infinity;
    for (const f of fishList) {
      if (!f.alive) continue;
      const d = Math.hypot(f.x - this.x, f.y - this.y);
      const score = d + f.speed * 0.6;
      if (score < bestScore) {
        bestScore = score;
        nearestDist = d;
        nearest = f;
      }
    }

    if (nearest) {
      if (nearest !== this.targetId) {
        // Switched targets - old distance history no longer means anything.
        this.targetId = nearest;
        this.distSamples = [];
      }

      this.distSamples.push(nearestDist);
      if (this.distSamples.length > STUCK_WINDOW) this.distSamples.shift();

      // Anti-stall check: if the distance to the target has barely moved
      // over the last ~1.5s while still out of eating range, we're circling
      // it rather than closing in - burst the turn rate to cut inside the loop.
      let turnRate = SHARK_MAX_TURN_RATE;
      if (this.distSamples.length >= STUCK_WINDOW) {
        const maxD = Math.max(...this.distSamples);
        const minD = Math.min(...this.distSamples);
        if (maxD - minD < STUCK_DISTANCE_SPREAD && minD > EAT_DISTANCE * 1.5) {
          turnRate = SHARK_BURST_TURN_RATE;
        }
      }

      const desiredAngle = this.interceptAngle(nearest);
      if (Number.isFinite(desiredAngle)) {
        const diff = angleDiff(this.angle, desiredAngle);
        const maxStep = turnRate * dt;
        this.angle += clamp(diff, -maxStep, maxStep);
      }
    } else {
      this.targetId = null;
      this.distSamples = [];
    }

    let nextX = this.x + Math.cos(this.angle) * SHARK_SPEED * dt;
    let nextY = this.y + Math.sin(this.angle) * SHARK_SPEED * dt;

    if (nextX < WALL_MARGIN) nextX = WALL_MARGIN;
    if (nextX > TANK_WIDTH - WALL_MARGIN) nextX = TANK_WIDTH - WALL_MARGIN;
    if (nextY < WALL_MARGIN) nextY = WALL_MARGIN;
    if (nextY > TANK_HEIGHT - WALL_MARGIN) nextY = TANK_HEIGHT - WALL_MARGIN;

    this.x = nextX;
    this.y = nextY;

    // Eat any fish within range.
    for (const f of fishList) {
      if (!f.alive) continue;
      if (Math.hypot(f.x - this.x, f.y - this.y) <= EAT_DISTANCE) {
        f.alive = false;
      }
    }
  }
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function randomSpawnPos() {
  // Spawn fish away from the tank center, where the shark starts.
  const margin = 60;
  let x, y;
  do {
    x = margin + Math.random() * (TANK_WIDTH - margin * 2);
    y = margin + Math.random() * (TANK_HEIGHT - margin * 2);
  } while (Math.hypot(x - TANK_WIDTH / 2, y - TANK_HEIGHT / 2) < 100);
  return { x, y };
}

class World {
  constructor() {
    this.generation = 0;
    this.timer = 0;
    this.bestEverTime = 0;
    this.bestEverBrain = null;
    this.lastGenBestTime = 0;
    this.fish = [];
    this.sharks = Array.from({ length: SHARK_COUNT }, () => new Shark());
    this.gpuBrain = new GpuBrainBatch();
    this.lastTickMs = 0;
    this.spawnGeneration(null);
  }

  // brains: null for a fresh random population, or an array of NeuralNet to use directly.
  spawnGeneration(brains) {
    this.fish = [];
    for (let i = 0; i < POPULATION_SIZE; i++) {
      const brain = brains ? brains[i] : new NeuralNet();
      const pos = randomSpawnPos();
      this.fish.push(new Fish(brain, pos.x, pos.y));
    }
    this.sharks = Array.from({ length: SHARK_COUNT }, () => new Shark());
    this.timer = 0;

    // Brains don't change again until the NEXT generation, so the batched
    // GPU weight tensors only need rebuilding here - not every tick.
    this.gpuBrain.build(this.fish);
  }

  allDead() {
    return this.fish.every((f) => !f.alive);
  }

  // Returns the currently-alive fish with the highest running fitness,
  // or the overall fittest fish (alive or not) if none are left this instant.
  getBestFish() {
    let best = null;
    for (const f of this.fish) {
      if (f.alive && (!best || f.fitness > best.fitness)) best = f;
    }
    if (best) return best;
    for (const f of this.fish) {
      if (!best || f.fitness > best.fitness) best = f;
    }
    return best;
  }

  aliveCount() {
    return this.fish.reduce((n, f) => n + (f.alive ? 1 : 0), 0);
  }

  // wantActivations: pass false to skip reading back the best fish's full
  // per-layer activation trace (only needed right before a render) - saves a
  // GPU->CPU sync on the "hidden" ticks that happen between rendered frames
  // at 5x/Max speed, where nobody will ever look at that data anyway.
  step(dt, wantActivations = true) {
    const t0 = performance.now();

    const inputsAll = this.fish.map((f) => f.sense(this.sharks));
    const bestFish = this.getBestFish();
    const bestIndex = wantActivations && bestFish ? this.fish.indexOf(bestFish) : -1;
    const outputsAll = this.gpuBrain.forward(this.fish, inputsAll, bestIndex);
    this.fish.forEach((f, i) => f.applyOutputs(outputsAll[i], dt));

    for (const shark of this.sharks) shark.update(this.fish, dt);
    this.timer += dt;

    this.lastTickMs = performance.now() - t0;

    if (this.allDead() || this.timer >= MAX_GEN_TIME) {
      this.finishGeneration();
    }
  }

  finishGeneration() {
    const sorted = this.fish.slice().sort((a, b) => b.fitness - a.fitness);
    this.lastGenBestTime = sorted[0].fitness;
    if (sorted[0].fitness > this.bestEverTime) {
      this.bestEverTime = sorted[0].fitness;
      this.bestEverBrain = sorted[0].brain.clone();
    }

    const nextBrains = this.breed(sorted);
    this.generation += 1;
    this.spawnGeneration(nextBrains);
  }

  breed(sortedFish) {
    const brains = [];

    // Elitism: carry the best few brains forward unchanged.
    for (let i = 0; i < ELITE_COUNT; i++) {
      brains.push(sortedFish[i].brain.clone());
    }

    // Weighted pool: top half of the population, weighted by fitness.
    const poolSize = Math.max(2, Math.floor(sortedFish.length / 2));
    const pool = sortedFish.slice(0, poolSize);
    const epsilon = 0.001;
    const weights = pool.map((f) => f.fitness + epsilon);
    const totalWeight = weights.reduce((a, b) => a + b, 0);

    const pickParent = () => {
      let r = Math.random() * totalWeight;
      for (let i = 0; i < pool.length; i++) {
        r -= weights[i];
        if (r <= 0) return pool[i].brain;
      }
      return pool[pool.length - 1].brain;
    };

    // Reproduction is asexual (clone one fitness-weighted parent, then mutate)
    // rather than two-parent crossover: since hidden-layer node counts now
    // evolve independently per individual, two parents can have differently
    // shaped brains, and there's no clean way to splice mismatched layer
    // shapes together. Cloning a single parent sidesteps that entirely.
    while (brains.length < POPULATION_SIZE) {
      const parent = pickParent();
      const child = parent.clone();
      child.mutate(0.15, 0.5);
      child.mutateStructure();
      brains.push(child);
    }

    return brains;
  }

  restart() {
    this.generation = 0;
    this.bestEverTime = 0;
    this.bestEverBrain = null;
    this.lastGenBestTime = 0;
    this.spawnGeneration(null);
  }
}
