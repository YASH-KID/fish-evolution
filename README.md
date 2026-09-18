# Fish Evolution

Watch a neural network learn to survive, live, in two panels:

- **Left panel — Live Brain:** the neural network of whichever fish is currently doing the best. Circles are neurons (colored by how activated they are right now — cyan for negative, orange for positive); lines are the weighted connections between them (blue = positive weight, red = negative, thickness = strength). This is the actual brain running the fish you see highlighted with a gold outline in the pond.
- **Right panel — The Pond:** 40 fish, each controlled by its own neural network, try to avoid **two sharks**. Neither shark is a neural network — each is an independent hand-coded pursuit AI that calculates a true intercept course (it aims at where a fish is *going to be*, not just where it is right now), with an anti-stall check that bursts its turn rate if it ever finds itself circling a fish without closing the distance, and it opportunistically targets whichever fish is easiest to catch (closest *and* slowest) rather than just the nearest one. The two sharks don't coordinate with each other — they just happen to create a much tighter squeeze between them.

This app runs entirely on GPU — see "GPU-only" below for exactly what that means and a real limitation on Windows laptops with two GPUs (Intel + NVIDIA) that's worth understanding before you go looking for your GPU usage meter to spike.

## How to open it

Double-click the **Fish Evolution** icon on your Desktop. It opens as its
own app window (no browser tabs/address bar).

You can also just double-click `index.html` in this folder to open it in
any browser — it works the same way, just inside a normal browser tab.

## Setting it up on a new computer

Double-click **`Setup Fish Evolution.bat`**. It opens a small setup window
that checks for Chrome or Edge (the only thing this app needs) and creates
the Desktop shortcut for you — no manual steps. Nothing else gets
installed, since this app has no other dependencies. (`Create Desktop
Shortcut.bat` does the same shortcut-creation step without the checklist
window, if you'd rather skip straight to it.)

## How it works

Each fish's brain takes **13 inputs** every instant:
1. its X position in the pond
2. its Y position in the pond
3. its current speed
4. its current acceleration
5. the direction it's facing

...then, for **each of the two sharks, ordered nearest-first** (so inputs 6-9 always describe whichever shark is currently closest, and 10-13 describe the farther one, regardless of which shark object that actually is):
6 & 10. whether that shark is visible (fish only see a 160° cone in front of them — 80° to each side)
7 & 11. that shark's bearing relative to the fish, if visible
8 & 12. that shark's exact X position (always known, regardless of line of sight)
9 & 13. that shark's exact Y position (always known, regardless of line of sight)

...and runs them through **three hidden layers** (starting at 8, 6, and 4 neurons) before producing **4 outputs**:
- **Turn** — how sharply to steer.
- **Thrust** — how much to speed up or brake.
- **Burst** — a sprint: up to 224 px/s top speed, a bit faster than either shark's 180 px/s, at the cost of turning sharpness. But it can only be *held* for 2 seconds at a time — hit that limit and it locks out for 3.5 seconds before it can be used again (shown in the pond as a light trailing glow while active).
- **Juke** — a temporary sharp dodge: up to 2.5x turn rate, at the cost of bleeding off speed.

The hidden layers' *node counts* evolve too, not just their weights — each new fish has a chance of a hidden layer gaining or losing a neuron from its parent (bounded between 2 and 14 nodes per layer; the number of hidden layers itself stays fixed at 3). Losing a node is more likely than gaining one, so brains only grow when it keeps earning its keep across generations rather than ballooning by default. Because two individuals can end up with differently-shaped brains, reproduction is asexual: each new fish clones one fitness-weighted parent and then mutates (both its weights and its structure), rather than splicing two parents together.

Every generation lasts until both sharks together have eaten every fish, or **60 seconds** pass, whichever comes first. The fish that survived longest pass their brains on to the next generation (the top 6 cloned unchanged, the rest mutated from fitness-weighted parents), so behavior should visibly improve over generations — fish that swim in straight lines toward a shark die out fast; fish that turn away, dash, or juke tend to survive longer and pass that on. With two sharks hunting at once, expect generations to end much faster on average than they used to.

## GPU-only

This app has no CPU compute path anymore — every fish's brain is batched into one computation and run through TensorFlow.js's WebGL backend, every tick. There's no toggle; it either runs on GPU or it doesn't run. If TensorFlow.js can't load (it comes from a CDN — needs an internet connection the first time), the pond shows a loading message for up to 15 seconds and then an error, rather than silently falling back to anything.

**If GPU usage reads 0% on your discrete GPU** (this was the case here until the Windows Graphics setting below was changed): on a laptop with two GPUs — a low-power integrated one (e.g. Intel UHD) and a discrete one (e.g. an NVIDIA RTX) — Windows decides *per application* which GPU that app's rendering goes to, and it defaults browsers to the integrated GPU to save battery. This is a system-level decision, not something a webpage can fully override — the `powerPreference: 'high-performance'` hint this app requests (in `gpuBrain.js`) is only advisory, and Windows/the driver frequently ignores it in favor of the per-app setting. That's why Intel usage can climb while NVIDIA usage stays flat: Chrome's GPU process is bound to the integrated chip regardless of what any page inside it asks for.

**The actual fix** (I can't make this change for you — it's a system settings change, not something I do on your behalf):
1. Open **Settings → System → Display → Graphics** (or search "Graphics settings" in the Start menu).
2. Add a desktop app, browse to Chrome's executable — typically `C:\Program Files\Google\Chrome\Application\chrome.exe` — and add it.
3. Click the newly added Chrome entry → **Options** → select **High performance** → **Save**.
4. Also open the **NVIDIA Control Panel → Manage 3D Settings → Program Settings**, add `chrome.exe`, and set "Preferred graphics processor for this program" to **High-performance NVIDIA processor**.
5. Fully quit Chrome (check Task Manager for lingering `chrome.exe` processes) and relaunch it via the Fish Evolution desktop shortcut.
6. Open a new tab to `chrome://gpu` and check the `GL_RENDERER` line — it should now mention NVIDIA instead of Intel. That confirms the fix took.

**Even after that fix, temper expectations.** These brains are tiny — a few hundred numbers each, even batched across 40 fish. GPU dispatch has a fixed cost every tick (uploading tensors, running the shader, blocking on the result) that a workload this small doesn't come close to needing — the RTX 5050 can do orders of magnitude more work per second than this simulation asks for. What "optimized for GPU" means concretely here is minimizing wasted round trips, not maximizing FLOPs:
- All 40 fish are batched into a *single* GPU computation per tick, not 40 separate ones.
- The live NN panel only needs one fish's full activation trace (whichever is currently best) — that's sliced out on the GPU and read back separately from the population-wide outputs, and skipped entirely on ticks that won't get rendered (the "hidden" ticks between frames at 5x/Max speed).
- Weight tensors are uploaded once per generation, not once per tick — they don't change until the next generation is bred.

Even with all of that, don't be surprised if your GPU usage monitor barely moves once it's correctly pointed at the NVIDIA GPU. That's expected for a model this size — it's not a sign anything is broken.

## Controls

- **1x / 5x / Max** — simulation speed. Max runs many generations quickly, though at GPU speed each tick still has real latency, so it won't blaze through generations the way a CPU-only version would.
- **Restart** — wipes all progress and starts back at generation 1 with brand-new random brains.

## Stats bar

- **Generation** — which generation is currently running.
- **Alive** — how many of the 40 fish are still alive this generation.
- **Gen Time** — how long the current generation has been running.
- **Last Gen Best** — the longest any fish survived in the previous generation.
- **Best Ever** — the longest any fish has ever survived, across all generations so far.
- **Backend** — always `GPU (WebGL)` once the app has started.
- **Tick Time** — how long the most recent tick (brain + physics) took to compute, in milliseconds. This is the number to watch if you're curious how much GPU dispatch overhead is costing per tick.

## Files

```
index.html                     → page structure (both panels, controls, stats)
style.css                      → dark ocean visual theme
neuralNetwork.js               → the NeuralNet class: forward pass, weight mutation, structural (add/remove-node) mutation
gpuBrain.js                    → batches the whole population and runs it on the GPU via TensorFlow.js (WebGL) - the only compute path
simulation.js                  → Fish/Shark physics + sensing, and the genetic algorithm
render.js                      → draws both canvases every frame
main.js                        → waits for GPU readiness, wires up controls, runs the animation loop
icon.ico                       → custom app icon, used by the Desktop shortcut
Setup Fish Evolution.bat       → double-click to set up / recreate the Desktop shortcut (checklist wizard)
setup.ps1                      → the setup wizard logic the .bat runs
Create Desktop Shortcut.bat    → lighter alternative: creates the shortcut with no checklist window
```

This app now requires an internet connection (at least the first load) to fetch TensorFlow.js from a CDN — GPU mode is the only mode, and there's no offline fallback.
