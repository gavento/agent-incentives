# Agent incentives for value changes

This is a simple experiment illustrating agent incentives to change values under noisy/drifting agent values.

While a perfectly rational agent (without value drift) has no incentives to change its values (follows from e.g. [Tom Everitt's work](https://arxiv.org/abs/1902.09980)), agents under value drift may prefer to intentionally move their value towards a favorable attractor.

## Experiments

In this experiment, the agent moves in the 1D (`revmod-1d.py`) or 2D (`revmod-2d.py`) space. Their perceived values are to stay in this point, e.g. they perceive their future rewards as the negative squared distance of the future point to their current point. The agent has no memory and will always optimize for their current values. The agent is aware of this behaviour while planning.

The value drift is modelled as normal distribution step biased by a (fixed) potential function, creating two attractors. Once the noisy update is selected (but before it is applied), the agent selects a direction to move in (with bounded radius) and the resulting move is the sum of the two. The space is qantized into a grid (20 resp 10x10). We assume 0.6, resp (0.6, 0.4) as the real values and start the agent there.

The algorithm iteratively computes all utilities _U(agent real values, agent state)_ with a dunamic algorithm using all possible moves and noise values (up to 1.5 stddev) until the maximal change is below 0.05.

## Results

* [1D, discounting 0.8](https://gavento.ucw.cz/view/revmod-1d-disc080.html) (agents everywhere counter the noise, trying to stay in place)
* [1D, discounting 0.95](https://gavento.ucw.cz/view/revmod-1d-disc095.html)
* [1D, discounting 0.99](https://gavento.ucw.cz/view/revmod-1d-disc099.html) (agents around value 0.6 purposefully move towards the right attractor)

* [2D, discounting 0.8](https://gavento.ucw.cz/view/revmod-2d-disc080.html) (agents everywhere counter the noise, trying to stay in place)
* [2D, discounting 0.95](https://gavento.ucw.cz/view/revmod-2d-disc095.html)
* [2D, discounting 0.99](https://gavento.ucw.cz/view/revmod-2d-disc099.html) (agents around value (0.6, 0.4) purposefully move towards the right attractor)
