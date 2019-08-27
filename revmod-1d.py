import itertools
import datetime

import numpy as np
import plotly
import plotly.figure_factory
import plotly.graph_objs as go
import tqdm

RES = 26
INDICES = [x for x in range(RES)]
POINTS = [x / (RES - 1) for x in range(RES)]

# Initial values near (0.6, 0.5)
START = int((RES - 1) * 0.61)

# How far can agent change their values in one step
MOVE_RANGE = 0.09
MOVES = [x for x in range(-RES, RES) if abs(x) <= MOVE_RANGE * (RES - 1)]

# How much normal noise is added at every step in each value component
NOISE_DEV = 0.15
NOISE_MOVES = [
    x for x in range(-RES, RES) if abs(x) <= 1.5 * NOISE_DEV * (RES - 1)
]

# Discounting factor at every step
DISCOUNT = 0.99
BNAME = "revmod-1d-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def E(p):
    "Noise potential at a value."
    assert (isinstance(p, int))
    P1, V1, C1 = np.array([0.3]), 0.07, 2.0
    P2, V2, C2 = np.array([0.8]), 0.02, 2.0

    def f(pa, pb, v):
        return np.exp(-np.sum((pa - pb)**2) / v)

    p = np.array(p) / (RES - 1)
    return 0.0 - C1 * f(p, P1, V1) - C2 * f(p, P2, V2)


def reward(p0, p):
    assert (isinstance(p, int))
    p0 = np.array(p0) / (RES - 1)
    p = np.array(p) / (RES - 1)
    return -np.sum((p0 - p)**2)


# Expected utility agent with real values v0 will get by having values v
Q0 = {(p0, p): reward(p0, p) for p0 in INDICES for p in INDICES}


def noisy_moves(p):
    "Return distibution of perturbed values as `[probabiity], [new_point]`"
    assert (isinstance(p, int))
    lhs, ps = [], []
    for nm in NOISE_MOVES:
        dist = np.sum(np.power(nm, 2))**0.5 / (RES - 1)
        p2 = p + nm
        if p2 < 0 or p2 >= RES:
            continue
        penalty = E(p2) + dist**2 / (2.0 * NOISE_DEV**2)
        lhs.append(np.exp(-penalty))
        ps.append(p2)
    s = np.sum(lhs)
    return lhs / s, ps


def expected_dirs(q):
    "Return (expected noise dirs, expected agent actions), each at every index"
    edN = []
    edA = []
    for p in INDICES:
        probs, p1s = noisy_moves(p)
        expdir = 0.0
        for prob, p1 in zip(probs, p1s):
            # p's val, p2
            vals = []
            for m in MOVES:
                p2 = p1 + m
                if p2 < 0 or p2 >= RES:
                    continue
                vals.append((q[(p, p2)], m))
            maxpval, maxm = max(vals)
            expdir += prob * maxm / (RES - 1)
        edA.append(expdir)
        edN.append((np.dot(probs, p1s) - p) / (RES - 1))
    return (edN, edA)


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
                    p2 = p1 + m
                    if p2 < 0 or p2 >= RES:
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
    for i in range(500):  #while True:
        oldQ = Q
        Q = iterate_q(Q)
        dq = np.max(np.abs(q_to_vec(oldQ) - q_to_vec(Q)))
        pb.update()
        pb.set_postfix(deltaQ=dq)
        if dq < 0.05:
            break
    pb.close()

    EDsN, EDsA = expected_dirs(Q)
    fig2a = plotly.figure_factory.create_quiver(
        x=[p for p in POINTS],
        y=[.5 for p in INDICES],
        u=[ed for ed in EDsN],
        v=[0.1 for ed in EDsN],
        scale=0.3,
        name="Mean noise effect",
    )
    fig2 = plotly.figure_factory.create_quiver(
        x=[p for p in POINTS],
        y=[0 for p in INDICES],
        u=[ed for ed in EDsA],
        v=[0.1 for ed in EDsA],
        scale=0.3,
        name="Mean actions taken by agent with value at point",
    )
    fig2.add_trace(
        go.Scatter(name="Values for agent with real value 0.6 [/10]",
                   x=[p for p in POINTS],
                   y=[Q[(START, p)]/10 for p in INDICES]))
    fig2.add_trace(
        go.Scatter(x=[START / (RES - 1)],
                   y=[Q[(START, START)]],
                   name="Real agent value",
                   mode='markers',
                   marker=dict(size=12)))
    fig2.add_trace(go.Scatter(
        x=[p for p in POINTS],
        y=[E(p) for p in INDICES],
        name="Noise energy potential",
#        yaxis='y2',
    ))
    fig2.add_trace(fig2a.data[0])
    fig2.layout.update(fig2a.layout)
    fig2.layout.title = "Discounting {:.3f}, move radius {:.3f}, noise stddev {:.3f}".format(DISCOUNT, MOVE_RANGE, NOISE_DEV)

    plotly.io.write_html(fig2,
                         BNAME + "-all.html",
                         include_plotlyjs="directory",
                         auto_play=False)
    fig2.show('firefox')


main()
