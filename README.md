# PilotNet Reproduction

A faithful PyTorch reproduction of NVIDIA's PilotNet from the 2016 paper [*End to End Learning for Self-Driving Cars*](https://arxiv.org/abs/1604.07316), trained on the Udacity Self-Driving Car Simulator.

The model takes a single forward-facing camera frame and predicts the steering angle directly — no lane detection, no path planning, no hand-crafted features. End to end.

## Demo

*Coming soon — a GIF of the model autonomously steering the car around the Udacity simulator track.*

## What's in this repo

- `src/model.py` — PilotNet architecture, faithful to Figure 4 of the paper
- `src/dataset.py` — PyTorch `Dataset` for the Udacity simulator's CSV-based logs
- `src/augment.py` — left/right camera offset and horizontal flip augmentation
- `src/train.py` — training loop with MPS (Apple Silicon GPU) support
- `src/drive.py` — connects to the simulator over WebSockets for real-time inference
- `docs/paper_notes.md` — my notes on the paper, what surprised me, what I'd do differently
- `notebooks/` — exploratory data analysis
- `results/` — training curves, demo video, evaluation metrics

## The architecture

PilotNet is a 9-layer convolutional neural network: 1 normalization layer, 5 convolutional layers, and 3 fully connected layers, totaling ~250,000 parameters. Input is a 66×200 RGB image (after cropping the sky and hood out of the original camera frame). Output is a single scalar — the steering angle in radians.

| Layer | Type | Output shape | Kernel | Stride |
|-------|------|--------------|--------|--------|
| 1 | Normalization | 3×66×200 | — | — |
| 2 | Conv + ELU | 24×31×98 | 5×5 | 2 |
| 3 | Conv + ELU | 36×14×47 | 5×5 | 2 |
| 4 | Conv + ELU | 48×5×22 | 5×5 | 2 |
| 5 | Conv + ELU | 64×3×20 | 3×3 | 1 |
| 6 | Conv + ELU | 64×1×18 | 3×3 | 1 |
| 7 | FC + ELU | 100 | — | — |
| 8 | FC + ELU | 50 | — | — |
| 9 | FC + ELU | 10 | — | — |
| 10 | FC (output) | 1 | — | — |

## How to run

```bash
# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train
python src/train.py --data-dir data/udacity --epochs 30

# Drive (with simulator running in autonomous mode)
python src/drive.py --model checkpoints/best.pth
```

## Roadmap

- [x] Project scaffolding
- [ ] Model architecture
- [ ] Dataset loader
- [ ] Augmentation pipeline
- [ ] Training loop
- [ ] First successful training run on Udacity sample data
- [ ] Inference connected to simulator
- [ ] Demo video of autonomous driving
- [ ] Extension: custom data collection
- [ ] Extension: attention visualization (what is the model looking at?)

## Why this project

I built this to deeply understand the end-to-end deep learning approach to autonomous driving and to develop hands-on experience with the full ML pipeline — paper to architecture to training to real-time deployment. The PilotNet paper is foundational in the AV space, and reproducing it forced me to grapple with concrete design decisions: input cropping, ELU vs ReLU, normalization placement, augmentation strategy.

## Acknowledgments

- Bojarski et al., NVIDIA, *End to End Learning for Self-Driving Cars*, 2016
- Udacity Self-Driving Car Engineer Nanodegree program for the open-source simulator

---
*Author: Temmy ([@IamTemmy](https://github.com/IamTemmy))*