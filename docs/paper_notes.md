# Paper Notes — *End to End Learning for Self-Driving Cars*

Bojarski et al., NVIDIA, 2016 — https://arxiv.org/abs/1604.07316
This is the paper this repo reproduces. Notes are in my own words.

## The one-sentence idea
Skip the usual pipeline (lane detection → path planning → control) and train a
single CNN to map raw camera pixels straight to a steering command. The network
decides what to look at on its own.

## Why that's a big deal
The traditional stack has hand-designed stages, and each stage is tuned for a
human-legible sub-goal ("detect lane lines well") rather than the thing you
actually care about (driving well). End-to-end training points every internal
weight at the final objective — good steering — so the network can discover
features nobody thought to hand-code. NVIDIA showed a fairly small net could
keep a real car on the road.

## Data collection (the part that makes it work)
- They recorded human driving: video plus the steering angle at each moment.
- The steering label is 1/r (inverse turning radius), not the raw wheel angle.
  This keeps the target well-behaved on straights — r → ∞ means 1/r → 0 instead
  of the value blowing up.
- Key trick: three cameras (left, center, right). The off-center cameras get a
  shifted steering label, which teaches the car to *recover* toward center.
  Without recovery examples a pure imitation model drifts and never learns to
  correct, because a good human demo never shows the car off-center.
- They also synthesize extra shifted/rotated viewpoints to widen recovery data.

## Architecture (Figure 4)
- Input: 66×200 RGB.
- 1 hardcoded normalization layer (fixed, not learned).
- 5 conv layers: three 5×5 stride-2, then two 3×3 stride-1.
- 3 fully connected hidden layers (100, 50, 10), then a 1-unit output.
- ~250k parameters. Small by modern standards.

## What I'm reproducing vs. what I'm not
- Reproducing: the architecture, normalization-inside-the-model, the three-camera
  recovery trick, training-from-demonstration.
- Not matching exactly: NVIDIA used real road video; I use the Udacity simulator.
  My label is the sim's normalized [-1, 1] steering value, not 1/r. My off-center
  data comes only from the sim's three cameras plus horizontal flips, not from
  synthesized viewpoint shifts.

## The limitation the paper doesn't dwell on (but I hit)
It's behavioral cloning, so it inherits covariate / distributional shift: the
model only trains on states a good human visits, but at inference it visits its
own slightly-wrong states and has no idea what to do there. The three-camera
trick is a partial patch, not a cure. This is exactly what DAgger addresses, and
it's the failure I observed on the bridge (see learning_log.md, Phases 4–5).

## Questions to carry into the extension phase
- Does temporal context (stacked frames / recurrence) reduce the jitter the
  single-frame model shows?
- Would DAgger-style aggregation measurably help on the bridge, or does the sim's
  simplicity cap the gain?
- Do saliency maps confirm the net actually looks at lane edges, like the paper
  claims for their model?

---
*Notes by Temmy. To extend as the project's research phase progresses.*
