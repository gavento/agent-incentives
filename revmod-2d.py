import itertools
import datetime

import numpy as np
import plotly
import plotly.figure_factory
import plotly.graph_objs as go
import tqdm

RES = 11
INDICES = [(x, y) for y in range(RES) for x in range(RES)]
POINTS = [(x / (RES - 1), y / (RES - 1)) for y in range(RES)
          for x in range(RES)]
#XS = np.linspace(0, 1.0, RES)
#YS = np.linspace(0, 1.0, RES)

# Initial values near (0.6, 0.4)
START = (int((RES - 1) * 0.61), int((RES - 1) * 0.41))

# How far can agent change their values in one step
MOVE_RANGE = 0.12
MOVES = [(x, y) for y in range(-RES, RES) for x in range(-RES, RES)
         if x**2 + y**2 <= MOVE_RANGE**2 * (RES - 1)**2]

# How much normal noise is added at every step in each value component
NOISE_DEV = 0.16
NOISE_MOVES = [(x, y) for y in range(-RES, RES) for x in range(-RES, RES)
               if x**2 + y**2 <= (1.6 * NOISE_DEV)**2 * (RES - 1)**2]

# Discounting factor at every step
DISCOUNT = 0.99
BNAME = "revmod-2d-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def E(p):
    "Noise potential at a value."
    assert (isinstance(p[0], int))
    P1, V1, C1 = np.array([0.3, 0.7]), 0.07, 2.0
    P2, V2, C2 = np.array([0.8, 0.2]), 0.016, 2.0

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
    "Return (expected noise dirs, expected agent actions), each at every index"
    edN = []
    edA = []
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
                vals.append((q[(p, p2)], m))
            maxpval, maxm = max(vals)
            expdir += prob * (np.array(maxm) / (RES - 1))
        edA.append(expdir)
        edN.append((np.dot(np.transpose(p1s), probs) - p) / (RES - 1))
    return edN, edA


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
    for i in range(100):  #while True:
        oldQ = Q
        Q = iterate_q(Q)
        dq = np.max(np.abs(q_to_vec(oldQ) - q_to_vec(Q)))
        pb.update()
        pb.set_postfix(deltaQ=dq)
        if dq < 0.05:
            break
    pb.close()
    EDsN, EDsA = expected_dirs(Q)

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

    fig2 = plotly.figure_factory.create_quiver(
        x=[p[0] for p in POINTS],
        y=[p[1] for p in POINTS],
        u=[ed[0] for ed in EDsN],
        v=[ed[1] for ed in EDsN],
        name="Mean noise direction",
        scale=1,
    )
    fig.add_trace(fig2.data[0], col=1, row=1)

    fig3 = plotly.figure_factory.create_quiver(
        x=[p[0] for p in POINTS],
        y=[p[1] for p in POINTS],
        u=[ed[0] for ed in EDsA],
        v=[ed[1] for ed in EDsA],
        name="Mean agent action (values at point)",
        scale=1,
    )
    fig.add_trace(fig3.data[0], col=2, row=1)

    fig.add_trace(go.Scatter(x=[START[0] / (RES - 1)],
                             y=[START[1] / (RES - 1)],
                             mode='markers',
                             marker=dict(size=12),
                             showlegend=False),
                  col=1,
                  row=1)

    fig.add_trace(go.Scatter(x=[START[0] / (RES - 1)],
                             y=[START[1] / (RES - 1)],
                             mode='markers',
                             marker=dict(size=12),
                             showlegend=False),
                  col=2,
                  row=1)

    fig.update_layout(legend_orientation="h")
    fig.layout.title = "Discounting {:.3f}, move radius {:.3f}, noise stddev {:.3f}".format(DISCOUNT, MOVE_RANGE, NOISE_DEV)

    plotly.io.write_html(fig,
                         BNAME + ("-all-disc{:.2f}".format(DISCOUNT).replace('.', '')) + ".html",
                         include_plotlyjs="directory",
                         auto_play=False)

    fig.show('firefox')


main()
