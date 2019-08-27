# Agent incentives for value changes

This is a simple experiment illustrating agent incentives to change values under noisy/drifting agent values.

While a perfectly rational agent (without value drift) has no incentives to change its values (follows from e.g. [Tom Everitt's work](https://arxiv.org/abs/1902.09980)), agents under value drift may prefer to intentionally move their value towards a favorable attractor.

## Experiments

In this experiment, the agent moves in the 1D (`revmod-1d.py`) or 2D (`revmod.py`) space. Their perceived values are to stay in this point, e.g. they perceive their future rewards as the negative squared distance of the future point to their current point. The agent has no memory and will always optimize for their current values. The agent is aware of this behaviour while planning.

The value drift is modelled as normal distribution step biased by a (fixed) potential function, creating two attractors. Once the noisy update is selected (but before it is applied), the agent selects a direction to move in (with bounded radius) and the resulting move is the sum of the two.

## 1D results

<iframe src=''></iframe>
