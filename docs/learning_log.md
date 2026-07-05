# Learning Log — PilotNet Reproduction

This is where I capture what I'm learning as I build this project. It's written for me — though anyone reading along is welcome.

---

## Phase 1: Project setup and the architecture (`model.py`)

### What I built
The PilotNet neural network from the 2016 NVIDIA paper, in PyTorch. About 250,000 parameters across 5 convolutional layers and 4 fully connected layers (three hidden layers of 100, 50, and 10 units, then a 1-unit output), with a hardcoded normalization step at the input.

### What I actually learned

**A neural network is just a class in Python.** Every PyTorch model is a class that inherits from `nn.Module`. You define the layers in `__init__`, you define how data flows through them in `forward`, and PyTorch handles everything else — gradients, GPU placement, saving weights.

**Why normalization belongs inside the model, not outside.** When you load a trained model later to actually use it (in our case, to drive a car), you don't want to have to remember "oh and don't forget to divide every pixel by 127.5 first." If the normalization is baked into the model, the model just works. You can't accidentally feed it raw images and get garbage out. The principle I want to remember: *the model should own its input contract.*

**Why no activation function on the final layer.** The last layer outputs a steering angle. Steering can be negative (turn left) or positive (turn right). If we'd put a ReLU or ELU activation on the output, we'd be squashing the values in a way that doesn't match what we want. General rule: regression problems get a raw output; classification problems get softmax or similar.

**The smoke test pattern.** Every model file ends with a small block that runs only when you execute the file directly. It creates a dummy input and runs it through the model just to confirm the shapes work out. This catches stupid bugs before you spend an hour wondering why training is broken.

### Question I asked that mattered
"Why 66×200 for the input image?" The honest answer: because the paper specified it, and we're doing a faithful reproduction. But the deeper answer is that the input size and the layer sizes had to be designed together — pick 66×200 and the conv layers' shrinking math works out to a clean 1×18 grid at the end. Pick something else and the math gets messy.

---

## Phase 2: Loading the data (`dataset.py`)

### What I built
A PyTorch `Dataset` class that reads the Udacity simulator's `driving_log.csv` and serves up `(image, steering_angle)` pairs ready for training. About 8,000 images of a virtual car driving around a track, each labeled with the steering wheel position at that moment.

### What I actually learned

**A PyTorch Dataset is a contract.** You give PyTorch two methods — one that says how many samples you have (`__len__`), and one that returns sample number i (`__getitem__`). That's it. Once you've written those two methods, PyTorch handles batching, shuffling, parallel loading, and feeding data to the GPU automatically.

**The preprocessing pipeline matters as much as the model.** Before any image reaches the network, we:
1. **Crop** off the sky at the top and the car hood at the bottom — these are pixels that contain no useful information for steering.
2. **Resize** what's left to 66×200, the size the model expects.
3. **Convert color order from BGR to RGB** — OpenCV (the image library we use) stores images as Blue-Green-Red for historical reasons, but the rest of the Python world uses Red-Green-Blue. Get this wrong and your network "sees" the wrong colors.
4. **Transpose the dimensions** — OpenCV gives images as (Height, Width, Channels) but PyTorch expects (Channels, Height, Width). Easy to forget, very common source of shape mismatch errors.

**Don't make the model learn what you already know.** The sky doesn't help predict steering. The car hood doesn't help predict steering. Cropping them out is using my own knowledge of the problem to make the network's job easier. The network has limited capacity — every bit of it that has to learn "ignore the sky" is capacity not available for learning "follow the lane line."

### Question I asked that mattered
"Why 66×200 specifically?" — answered above.

"What about the hardware analogies?" — I asked for the explanations to be in plain language instead of engineering analogies. Going forward, the analogies use everyday concepts.

### What surprised me
The header detection logic for the CSV. The Udacity sample data has a header row but data I record myself in the simulator won't. The code auto-detects which case it's dealing with by trying to parse the first row as a number. Small piece of robustness that I wouldn't have thought of, but it'll matter when I record my own driving data later.


---

## Phase 3: Augmentation and training setup (`augment.py` + `train.py`)

### What I built
Data augmentation that triples the dataset by using all three cameras (with adjusted steering labels for the side cameras) and doubles it again with horizontal flips — going from ~8,000 raw rows to ~48,000 training samples. Then a full training script that splits the data carefully, runs the model through many epochs, and saves checkpoints.

### What I actually learned

**Augmentation is "free" training data through clever interpretation.** The 8,000 raw rows of the CSV already had three image paths each (left, center, right cameras mounted on the same virtual car). By treating the side cameras as "what the road looks like if the car had drifted slightly off-center" and adjusting the steering label by ±0.20 to represent the recovery needed, we get three times the training examples with no new data collection. Adding mirrored versions doubles it again. The network ends up learning recovery behavior too — what to do when the car has drifted.

**Data leakage in train/validation splits.** If we naively split the 48,000 augmented samples randomly, the same driving moment could appear in both sets (e.g., the center camera in training and the flipped left camera in validation). The validation loss would be artificially low because the model has technically seen that moment during training. The fix: split on the raw 8,000 rows first, then expose all 6 augmented variants only within each split. This is the kind of subtle mistake that breaks ML projects silently and is a common interview question.

**The four moving parts of any training loop:** a model that predicts, a loss function that measures wrongness, an optimizer that adjusts weights based on that wrongness, and an outer loop that iterates over the data. Every PyTorch training script has these four parts. The "PyTorch trinity" inside each step: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, in that order.

**Why we evaluate on a held-out validation set.** Training loss going down just means the network is fitting the training data — it could just be memorizing. Validation loss going down means it's actually learning the task in a way that generalizes. If training loss keeps falling but validation loss starts rising, the network is memorizing and we need to stop.

**Hyperparameters as command-line flags.** Batch size, learning rate, number of epochs — all configurable from the command line via `argparse`. Means we can experiment by typing `python -m src.train --lr 1e-3 --epochs 50` instead of editing code each time.

### Questions I asked that mattered
"Are we splitting each image into three parts?" No — each driving moment already had three separate complete images (one from each camera). The CSV has three image paths per row. We weren't splitting anything; we were using more of what was already there.

"What's the random seed for?" Reproducibility. Random shuffles in code aren't truly random — they come from a formula seeded with a starting number. Set the seed, get the same shuffle every time. Means experiments can be re-run identically, which matters for fair comparison.

### What surprised me
That a fully working training script is only ~120 lines of code. I expected ML training to be much more complex underneath. Most of it is just bookkeeping — loop over data, run the model, save checkpoints, track losses. The "intelligence" lives almost entirely in the model file and the loss function. The trainer itself is plumbing.

---

## Phase 4: Evaluation, deployment, and observations

### What I built
Three closing pieces: the inference server (`drive.py`) that connects the trained model to the live Udacity simulator over WebSockets; the quantitative evaluation script (`evaluate.py`) that computes aggregate metrics, a sample predictions grid, and an error histogram across all held-out validation data; and the final repo polish — a polished README, a demo video, and the supporting artifacts.

### What I actually learned

**Loading a saved model is a two-step process.** A `.pth` file contains only the weights, not the architecture. You build the same model class first, then call `load_state_dict` to pour the weights in. Always call `.eval()` after loading for inference — some layer types behave differently during training vs inference, and even though PilotNet doesn't use those layers, it's the right habit. Loading would silently break if I changed the model architecture without keeping old checkpoints aligned.

**Real-time inference uses WebSockets.** The simulator is a separate program that talks to my Python script over a WebSocket connection on port 4567. For every camera frame the simulator sends, my code preprocesses the image, runs the model, and sends back a steering command. The "the model owns its input contract" principle from earlier paid off here — `preprocess()` does the exact same crop/resize/RGB/transpose pipeline regardless of whether the image came from disk (training) or from the simulator (inference). One place to change preprocessing, one source of truth.

**The driving wasn't smooth.** My trained model drove the track but with constant micro-corrections. This is called *steering jitter* and it happens because PilotNet predicts each frame independently with no memory. Modern AV systems fix this with temporal smoothing or recurrent networks.

**The car eventually got stuck on the bridge.** A cobblestone bridge with a texture different from the regular road caused the model to drift slightly. Once drifted, the model was in a state it hadn't seen well during training, made worse predictions, and the failure compounded until the car beached itself on the curb. This is the textbook *distributional shift* / *covariate shift* failure mode of behavioral cloning. DAgger and similar methods exist specifically to address this. Knowing the name of the failure mode is half the battle in interviews.

**The MSE from evaluation matched the val loss from training exactly.** Both were 0.01093. That's not luck — it's a consistency check confirming that the evaluation script is computing the same loss over the same validation set with the same weights. Subtle, but a real sanity check that I'd recommend doing on any project where evaluation is implemented separately from training.

**Real-world toolchain friction.** The Udacity simulator is an unsigned 2017 Unity build. On modern macOS Apple Silicon, getting it to launch required stripping the quarantine attribute, ensuring the binary was executable, and pinning specific old versions of `python-socketio` and `python-engineio` so the protocol matched. Not unusual for production ML — older deployed artifacts often need specific library versions to stay alive.

### What surprised me
How quickly the model converged on the basic task. After just 2 epochs of training, validation loss was already at 0.018 — well within the "good" range. Most of the remaining training was diminishing-returns refinement. This shaped my view of training schedules: the first few epochs do the heavy lifting; long training runs are about squeezing the last few percent.

### What I'd do differently next time
Record significantly more training data, especially on the bridge sections and sharp curves where the model fails. The fix for distributional shift starts with making the training distribution wider — augmentation can only stretch existing data so far. Beyond that, swapping the per-frame regression for a recurrent model (LSTM or Transformer head over a temporal window) would likely eliminate the jitter and handle the bridge transitions more gracefully.

---

## Phase 5: Closed-loop observations across multiple runs

### What happened
Ran the simulator again to show the demo to someone else. Surprisingly, the model produced a meaningfully different result from the original run — the car still jittered aggressively at the cobblestone bridge, but this time it survived the bridge and kept driving for over a minute. In the original run captured in the demo video, the same model on the same weights had failed at the same bridge after ~30 seconds.

### What I actually learned

**The system is non-deterministic across runs even though the model is fixed.** Two runs with the exact same trained weights produced visibly different trajectories. This was initially confusing — how can a deterministic neural network produce different outputs? The answer is that the network itself is (mostly) deterministic, but the *system* it operates in isn't. Several things vary subtly between runs:
- Frame timing from the simulator depends on what else the OS is doing
- Floating-point math on the MPS GPU backend has tiny non-determinism
- The simulator's physics engine has small numerical variation

Individually, these are negligible. But the system is a closed feedback loop: model output → car position → next image → next model output → next car position. Each tiny variation gets amplified through the loop, especially near decision boundaries where the model's prediction was already uncertain.

**The bridge is a decision boundary, which is why jitter intensified there.** On the regular road the model is confident, so frame-to-frame predictions stay similar. On the cobblestone bridge — visually unlike anything in the training distribution — the model's confidence drops, predictions become noisier, and the noise gets amplified by the closed-loop physics. Sometimes the noise lands in a recoverable state, sometimes not. The non-determinism IS the jitter, observed at the system level.

**An AV system is more than its model.** Knowing what the architecture predicts isn't enough; you also have to reason about how those predictions interact with physics, timing, and the world in real time. Production AV teams spend enormous effort on closed-loop testing for exactly this reason.

### What surprised me
That I could *observe* the difference between an open-loop and closed-loop system directly, on my own laptop, by just running the same simulator twice. In the training data world (open loop), the model produces a single prediction per image and that's the end of the story. In the simulator world (closed loop), the model and the physics and the timing all interact, and the same model can produce different journeys. Realizing this changed how I think about deploying ML systems generally — the model's accuracy on a static test set isn't a complete picture of how it'll behave in a live system.