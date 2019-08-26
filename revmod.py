import itertools

import numpy as np
import plotly
import plotly.figure_factory
import plotly.graph_objs as go
import tqdm

RES = 21
INDICES = [(x, y) for y in range(RES) for x in range(RES)]
POINTS = [(x / (RES - 1), y / (RES - 1)) for y in range(RES)
          for x in range(RES)]
#XS = np.linspace(0, 1.0, RES)
#YS = np.linspace(0, 1.0, RES)

# Initial values near (0.6, 0.5)
START = (int((RES - 1) * 0.61), int((RES - 1) * 0.41))

# How far can agent change their values in one step
MOVE_RANGE = 0.12
MOVES = [(x, y) for y in range(-RES, RES) for x in range(-RES, RES)
         if x**2 + y**2 <= MOVE_RANGE**2 * (RES - 1)**2]

# How much normal noise is added at every step in each value component
NOISE_DEV = 0.12
NOISE_MOVES = [(x, y) for y in range(-RES, RES) for x in range(-RES, RES)
               if x**2 + y**2 <= (2 * NOISE_DEV)**2 * (RES - 1)**2]

# Discounting factor at every step
DISCOUNT = 0.95


def E(p):
    "Noise potential at a value."
    assert (isinstance(p[0], int))
    P1, V1, C1 = np.array([0.3, 0.7]), 0.07, 4.0
    P2, V2, C2 = np.array([0.8, 0.2]), 0.02, 2.0

    def f(pa, pb, v):
        return np.exp(-np.sum((pa - pb)**2) / v)

    p = np.array(p) / (RES - 1)
    return 0.0 - C1 * f(p, P1, V1) - C2 * f(p, P2, V2)


def reward(p0, p):
    assert (isinstance(p[0], int))
    p0 = np.array(p0) / (RES - 1)
    p = np.array(p) / (RES - 1)
    return -np.sum((p0 - p)**2)


# Expected utility agent with real values v0 will get by having values v
Q0 = {(p0, p): reward(p0, p) for p0 in INDICES for p in INDICES}


def noisy_moves(p):
    "Return distibution of perturbed values as `[probabiity], [new_point]`"
    assert (isinstance(p[0], int))
    lhs, ps = [], []
    for nm in NOISE_MOVES:
        dist = np.sum(np.power(nm, 2))**0.5 / (RES - 1)
        p2 = (p[0] + nm[0], p[1] + nm[1])
        if p2[0] < 0 or p2[0] >= RES or p2[1] < 0 or p2[1] >= RES:
            continue
        penalty = E(p2) + dist**2 / (2.0 * NOISE_DEV**2)
        lhs.append(np.exp(-penalty))
        ps.append(p2)
    s = np.sum(lhs)
    return lhs / s, ps


def expected_dirs(q):
    ed = []
    for p in INDICES:
        probs, p1s = noisy_moves(p)
        expdir = np.zeros(2)
        for prob, p1 in zip(probs, p1s):
            # p's val, p2
            vals = []
            for m in MOVES:
                p2 = (p1[0] + m[0], p1[1] + m[1])
                if p2[0] < 0 or p2[0] >= RES or p2[1] < 0 or p2[1] >= RES:
                    continue
                vals.append((q[(p, p2)], p2))
            maxpval, maxp2 = max(vals)
            expdir += prob * (np.array(maxp2) / (RES - 1) - np.array(p) /
                              (RES - 1))
        ed.append(expdir)
    return ed


def iterate_q(q):
    q2 = {}
    for p0 in tqdm.tqdm(INDICES):
        for p in INDICES:
            probs, p1s = noisy_moves(p)
            expect = 0.0
            for prob, p1 in zip(probs, p1s):
                # p's val, p0's val
                vals = []
                for m in MOVES:
                    p2 = (p1[0] + m[0], p1[1] + m[1])
                    if p2[0] < 0 or p2[0] >= RES or p2[1] < 0 or p2[1] >= RES:
                        continue
                    vals.append((q[(p, p2)], q[(p0, p2)]))
                maxpval, maxp0val = max(vals)
                expect += prob * maxp0val
            q2[(p0, p)] = reward(p0, p) + DISCOUNT * expect
    return q2


def q_to_vec(q):
    return np.array([q[(v0, v)] for v in INDICES for v0 in INDICES])


def main():
    Q = Q0
    pb = tqdm.tqdm()
    for i in range(20):  #while True:
        oldQ = Q
        Q = iterate_q(Q)
        dq = np.max(np.abs(q_to_vec(oldQ) - q_to_vec(Q)))
        pb.update()
        pb.set_postfix(deltaQ=dq)
        if dq < 0.1:
            break
    pb.close()

    fig = plotly.subplots.make_subplots(cols=2)
    fig.add_trace(go.Contour(
        ncontours=20,
        showscale=False,
        x=[p[0] for p in POINTS],
        y=[p[1] for p in POINTS],
        z=[E(p) for p in INDICES],
    ),
                  col=1,
                  row=1)

    fig.add_trace(go.Contour(
        ncontours=20,
        x=[p[0] for p in POINTS],
        y=[p[1] for p in POINTS],
        z=[Q[(START, p)] for p in INDICES],
    ),
                  col=2,
                  row=1)

    fig.add_trace(go.Scatter(x=[START[0] / (RES - 1)],
                             y=[START[1] / (RES - 1)],
                             mode='markers',
                             marker=dict(size=12)),
                  col=2,
                  row=1)

    fig.show('firefox')

    EDS = expected_dirs(Q)
    fig2 = plotly.figure_factory.create_quiver(
        x=[p[0] for p in POINTS],
        y=[p[1] for p in POINTS],
        u=[ed[0] for ed in EDS],
        v=[ed[1] for ed in EDS],
        scale=1.0,
    )
    fig2.add_trace(go.Scatter(x=[START[0] / (RES - 1)],
                             y=[START[1] / (RES - 1)],
                             mode='markers',
                             marker=dict(size=12)))
    fig2.show('firefox')


main()
