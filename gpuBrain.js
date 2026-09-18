// gpuBrain.js
// Runs the whole population's brains as ONE batched matrix computation on
// the GPU via TensorFlow.js's WebGL backend, instead of separate per-fish CPU
// forward passes. Individual fish can have differently-shaped hidden layers
// (see neuralNetwork.js's structural mutation), so every brain is zero-padded
// up to a fixed width first (NeuralNet.toPaddedArrays) - padding is
// mathematically inert, so this produces the exact same outputs as each
// fish's own smaller network, just computed together as one batch.
//
// This app runs on GPU only now - there is no CPU fallback path. If
// TensorFlow.js can't load (see index.html - it comes from a CDN), the app
// will not start; main.js shows a loading/error state instead of silently
// falling back.
//
// Honest tradeoff, worth knowing: this network is tiny (a few hundred
// parameters per fish, even at 40 fish). GPU dispatch has a fixed cost every
// tick - uploading tensors, running the shader, blocking on the result -
// that a workload this small doesn't come close to amortizing; the real
// bottleneck for a model this size is that fixed per-call overhead, not raw
// FLOPs. The optimization below is aimed squarely at that: it minimizes how
// many GPU->CPU sync round trips happen per tick, rather than trying to give
// the GPU more busywork to look occupied.

const GPU_MAX_HIDDEN = MAX_HIDDEN_NODES; // from neuralNetwork.js - uniform padded width

// Best-effort hint to the browser/OS to route this page's GPU work to a
// discrete/high-performance GPU rather than an integrated one, by creating a
// throwaway high-performance WebGL context before TensorFlow.js creates its
// own. This is NOT reliable by itself on Windows laptops with hybrid
// graphics (Intel + NVIDIA) - the OS's per-application GPU assignment
// usually wins regardless of what a page asks for. See README for the actual
// fix (Windows Graphics settings / NVIDIA Control Panel).
function warmUpHighPerformanceContext() {
  try {
    const canvas = document.createElement('canvas');
    const attrs = { powerPreference: 'high-performance', failIfMajorPerformanceCaveat: false };
    return !!(canvas.getContext('webgl2', attrs) || canvas.getContext('webgl', attrs));
  } catch (e) {
    return false;
  }
}
warmUpHighPerformanceContext();

class GpuBrainBatch {
  constructor() {
    this.weights = null; // { W1,b1,W2,b2,W3,b3,W4,b4 } as tf.Tensor, batch dim = population size
    this.built = false;
  }

  static isAvailable() {
    return typeof tf !== 'undefined';
  }

  // Builds batched weight tensors for the given fish array. Call once per
  // generation (brains don't change mid-generation) - NOT once per tick.
  build(fishArray) {
    this.dispose();
    if (!GpuBrainBatch.isAvailable()) return;

    const padded = fishArray.map((f) => f.brain.toPaddedArrays(GPU_MAX_HIDDEN));

    this.weights = tf.tidy(() => ({
      W1: tf.tensor(padded.map((p) => p.W1)), // [N, maxHidden, 9]
      b1: tf.tensor(padded.map((p) => p.b1)).reshape([padded.length, 1, GPU_MAX_HIDDEN]),
      W2: tf.tensor(padded.map((p) => p.W2)), // [N, maxHidden, maxHidden]
      b2: tf.tensor(padded.map((p) => p.b2)).reshape([padded.length, 1, GPU_MAX_HIDDEN]),
      W3: tf.tensor(padded.map((p) => p.W3)),
      b3: tf.tensor(padded.map((p) => p.b3)).reshape([padded.length, 1, GPU_MAX_HIDDEN]),
      W4: tf.tensor(padded.map((p) => p.W4)), // [N, 4, maxHidden]
      b4: tf.tensor(padded.map((p) => p.b4)).reshape([padded.length, 1, OUTPUT_SIZE]),
    }));
    this.realHiddenSizes = padded.map((p) => p.realHiddenSizes);
    this.built = true;
  }

  // inputsAll: array of N input arrays (length INPUT_SIZE each).
  // bestIndex: index of the one fish whose full activation trace should be
  // read back for the live NN panel, or -1 to skip that entirely (used on
  // "hidden" ticks between rendered frames at 5x/Max speed, where nobody
  // will look at that data anyway - skipping it cuts sync round trips).
  // Returns: array of N output arrays (length OUTPUT_SIZE each) - this part
  // always has to be read back, every tick, since physics needs it.
  forward(fishArray, inputsAll, bestIndex = -1) {
    if (!this.built) return inputsAll.map(() => [0, 0, 0, 0]);

    let outputsAll;
    tf.tidy(() => {
      const x = tf.tensor(inputsAll).reshape([inputsAll.length, 1, INPUT_SIZE]);
      const w = this.weights;

      const h1 = x.matMul(w.W1, false, true).add(w.b1).tanh(); // [N,1,maxHidden]
      const h2 = h1.matMul(w.W2, false, true).add(w.b2).tanh();
      const h3 = h2.matMul(w.W3, false, true).add(w.b3).tanh();
      const out = h3.matMul(w.W4, false, true).add(w.b4).tanh(); // [N,1,4]

      outputsAll = out.reshape([inputsAll.length, OUTPUT_SIZE]).arraySync();

      if (bestIndex >= 0) {
        const [s1, s2, s3] = this.realHiddenSizes[bestIndex];
        // Slice just the one needed row out of each hidden layer (on the GPU,
        // cheap) and combine into a single readback instead of three separate
        // syncs of the full [N, maxHidden] tensors we'd otherwise discard.
        const h1Row = h1.slice([bestIndex, 0, 0], [1, 1, GPU_MAX_HIDDEN]).reshape([GPU_MAX_HIDDEN]);
        const h2Row = h2.slice([bestIndex, 0, 0], [1, 1, GPU_MAX_HIDDEN]).reshape([GPU_MAX_HIDDEN]);
        const h3Row = h3.slice([bestIndex, 0, 0], [1, 1, GPU_MAX_HIDDEN]).reshape([GPU_MAX_HIDDEN]);
        const combined = tf.concat([h1Row, h2Row, h3Row]).arraySync();

        const f = fishArray[bestIndex];
        f.brain.activations[0] = inputsAll[bestIndex];
        f.brain.activations[1] = combined.slice(0, s1);
        f.brain.activations[2] = combined.slice(GPU_MAX_HIDDEN, GPU_MAX_HIDDEN + s2);
        f.brain.activations[3] = combined.slice(GPU_MAX_HIDDEN * 2, GPU_MAX_HIDDEN * 2 + s3);
        f.brain.activations[4] = outputsAll[bestIndex];
      }
    });

    return outputsAll;
  }

  dispose() {
    if (this.weights) {
      Object.values(this.weights).forEach((t) => t.dispose());
      this.weights = null;
    }
    this.built = false;
  }
}
