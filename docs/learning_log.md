# Learning Log — PilotNet Reproduction

This is where I capture what I'm learning as I build this project. It's written for me — though anyone reading along is welcome.

---

## Phase 1: Project setup and the architecture (`model.py`)

### What I built
The PilotNet neural network from the 2016 NVIDIA paper, in PyTorch. About 250,000 parameters across 5 convolutional layers and 4 fully connected layers, with a hardcoded normalization step at the input.

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