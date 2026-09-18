// neuralNetwork.js
// A hand-rolled feedforward neural network with an EVOLVABLE topology:
// 13 inputs -> up to 3 hidden layers (tanh) -> 4 outputs (tanh). The number
// of hidden layers is capped at 3, but the node count *within* each hidden
// layer can grow or shrink over generations via structural mutation - this
// is a simplified, mutation-only (no crossover) neuroevolution scheme,
// which sidesteps the "parents have different shapes" problem that sexual
// crossover runs into once topology itself is evolving.

const INPUT_SIZE = 13;
const OUTPUT_SIZE = 4;
const INITIAL_HIDDEN_SIZES = [8, 6, 4]; // starting point: 7,8,6,4,2 -> now widened per the 9-in/4-out redesign
const HIDDEN_LAYER_COUNT = 3; // fixed cap - only node counts within these layers evolve
const MIN_HIDDEN_NODES = 2;
const MAX_HIDDEN_NODES = 14;

function tanh(x) {
  return Math.tanh(x);
}

function randomWeight() {
  return Math.random() * 2 - 1;
}

function randomMatrix(rows, cols) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) row.push(randomWeight());
    m.push(row);
  }
  return m;
}

function randomVector(n) {
  const v = [];
  for (let i = 0; i < n; i++) v.push(randomWeight());
  return v;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

class NeuralNet {
  constructor(hiddenSizes = INITIAL_HIDDEN_SIZES.slice()) {
    this.layerSizes = [INPUT_SIZE, ...hiddenSizes, OUTPUT_SIZE];
    // weights[l] connects layer l -> layer l+1, shape (layerSizes[l+1] x layerSizes[l])
    this.weights = [];
    this.biases = [];
    for (let l = 0; l < this.layerSizes.length - 1; l++) {
      this.weights.push(randomMatrix(this.layerSizes[l + 1], this.layerSizes[l]));
      this.biases.push(randomVector(this.layerSizes[l + 1]));
    }

    // Last computed activations per layer (including input/output), kept
    // around purely for live visualization.
    this.activations = this.layerSizes.map((n) => new Array(n).fill(0));
  }

  forward(inputs) {
    let current = inputs;
    this.activations[0] = inputs;

    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const next = new Array(W.length);
      for (let i = 0; i < W.length; i++) {
        let sum = b[i];
        const row = W[i];
        for (let j = 0; j < current.length; j++) sum += row[j] * current[j];
        next[i] = tanh(sum);
      }
      this.activations[l + 1] = next;
      current = next;
    }

    return current;
  }

  clone() {
    const hiddenSizes = this.layerSizes.slice(1, this.layerSizes.length - 1);
    const copy = new NeuralNet(hiddenSizes);
    copy.weights = this.weights.map((matrix) => matrix.map((row) => row.slice()));
    copy.biases = this.biases.map((vec) => vec.slice());
    copy.layerSizes = this.layerSizes.slice();
    return copy;
  }

  // Nudges every weight/bias with `rate` probability, keeping most of the
  // inherited behavior intact.
  mutate(rate = 0.15, amount = 0.5) {
    const nudge = (val) => {
      if (Math.random() < rate) {
        return clamp(val + (Math.random() * 2 - 1) * amount, -2, 2);
      }
      return val;
    };
    this.weights = this.weights.map((matrix) => matrix.map((row) => row.map(nudge)));
    this.biases = this.biases.map((vec) => vec.map(nudge));
  }

  // Adds one neuron to hidden layer `layerIdx` (an index into layerSizes,
  // so 1, 2, or 3). Its incoming weights are random; its outgoing weights
  // start small so the new neuron doesn't immediately wreck behavior that
  // was already working.
  addNodeToLayer(layerIdx) {
    if (this.layerSizes[layerIdx] >= MAX_HIDDEN_NODES) return;

    const incoming = this.weights[layerIdx - 1]; // rows = this layer, cols = previous layer
    incoming.push(randomVector(this.layerSizes[layerIdx - 1]));
    this.biases[layerIdx - 1].push(randomWeight());

    const outgoing = this.weights[layerIdx]; // rows = next layer, cols = this layer
    for (const row of outgoing) row.push((Math.random() * 2 - 1) * 0.2);

    this.layerSizes[layerIdx] += 1;
  }

  // Removes one random neuron from hidden layer `layerIdx`.
  removeNodeFromLayer(layerIdx) {
    if (this.layerSizes[layerIdx] <= MIN_HIDDEN_NODES) return;

    const idx = Math.floor(Math.random() * this.layerSizes[layerIdx]);
    this.weights[layerIdx - 1].splice(idx, 1);
    this.biases[layerIdx - 1].splice(idx, 1);
    for (const row of this.weights[layerIdx]) row.splice(idx, 1);

    this.layerSizes[layerIdx] -= 1;
  }

  // Rolls the dice on growing/shrinking each of the 3 hidden layers. Adding a
  // node is deliberately harder than removing one, so brains only grow when
  // it keeps paying off across generations rather than ballooning by default.
  mutateStructure(addProb = 0.04, removeProb = 0.1) {
    for (let layerIdx = 1; layerIdx <= HIDDEN_LAYER_COUNT; layerIdx++) {
      if (Math.random() < addProb) this.addNodeToLayer(layerIdx);
      else if (Math.random() < removeProb) this.removeNodeFromLayer(layerIdx);
    }
  }

  // Zero-pads every hidden layer up to a fixed `maxHidden` width so a whole
  // population of differently-shaped brains can be stacked into one batch of
  // uniform-shaped tensors and run through the GPU together. Padding is
  // mathematically inert: a padded neuron always has bias 0 and all-zero
  // weights, so it always outputs tanh(0)=0 and contributes nothing to real
  // neurons downstream - the padded forward pass is numerically identical to
  // this network's own (smaller) forward pass, just wastefully wider.
  toPaddedArrays(maxHidden) {
    const padRows = (matrix, targetRows) => {
      const cols = matrix.length ? matrix[0].length : 0;
      const padded = matrix.map((row) => row.slice());
      while (padded.length < targetRows) padded.push(new Array(cols).fill(0));
      return padded;
    };
    const padCols = (matrix, targetCols) => matrix.map((row) => {
      const padded = row.slice();
      while (padded.length < targetCols) padded.push(0);
      return padded;
    });
    const padVec = (vec, targetLen) => {
      const padded = vec.slice();
      while (padded.length < targetLen) padded.push(0);
      return padded;
    };

    const [h1, h2, h3] = this.layerSizes.slice(1, 4);

    return {
      W1: padRows(this.weights[0], maxHidden), // [maxHidden x 9], only rows padded (input dim is fixed)
      b1: padVec(this.biases[0], maxHidden),
      W2: padRows(padCols(this.weights[1], maxHidden), maxHidden), // [maxHidden x maxHidden]
      b2: padVec(this.biases[1], maxHidden),
      W3: padRows(padCols(this.weights[2], maxHidden), maxHidden), // [maxHidden x maxHidden]
      b3: padVec(this.biases[2], maxHidden),
      W4: padCols(this.weights[3], maxHidden), // [4 x maxHidden], only cols padded (output dim is fixed)
      b4: this.biases[3].slice(),
      realHiddenSizes: [h1, h2, h3],
    };
  }
}
