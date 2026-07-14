"""Experiment 3 — Dynamic environment: field decay enables adaptation.

The environment is a knowledge-base category that changes every SHIFT_INTERVAL
rounds. Adopters observe it and deposit their token; passives only read the
field's dominant token and align to it, so they track the environment only if
the field does. With decay, stale counts fade and the dominant token follows
each shift, so passives re-align. Without decay the first category's counts
freeze the dominant token and passives stay stuck. Consensus (a cumulative
average with no forgetting) drifts toward the centroid of past environments.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import SharedKnowledgeBase

# Parameters
DIM = 512
M = 10
N = 50
K = 3             # adopters (sensors); the rest are passives
BETA = 0.3        # adopter observation weight
ALPHA = 0.3       # passive field-reading weight
SIGMA = 0.0       # observation noise
THETA = 0.3       # gating threshold (kept for reference)
SHIFT_INTERVAL = 50
N_PHASES = 4
TOTAL_ROUNDS = SHIFT_INTERVAL * N_PHASES
ENV_SEQ = [0, 1, 2, 3]          # each phase is a new category

# Decay-rate sweep: counts *= (1 - rho) each round; rho=0 is no decay.
RHOS = [0.3, 0.03, 0.0]
RHO_COLORS = ["tab:green", "tab:orange", "tab:red"]
RUNS = 20

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial:bold",
    "mathtext.it": "Arial:italic:bold",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "bf",
    "font.size": 9,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
})

# Orthonormal categories so an old-aligned belief has exactly zero similarity
# to a shifted environment.
KB = SharedKnowledgeBase(num_tokens=M, dim=DIM, seed=0, orthonormal=True)


def env_cat(t: int) -> int:
    return ENV_SEQ[min(t // SHIFT_INTERVAL, N_PHASES - 1)]


def adopter_obs(t: int) -> torch.Tensor:
    env = KB.E[env_cat(t)].unsqueeze(0).expand(K, DIM)
    if SIGMA > 0:
        return F.normalize(env + SIGMA * torch.randn(K, DIM), dim=1)
    return env


def run_stigmergy(rho: float) -> np.ndarray:
    adopter = F.normalize(torch.randn(K, DIM), dim=1)
    passive = F.normalize(torch.randn(N - K, DIM), dim=1)
    counts = torch.zeros(M)
    sims = []

    for t in range(TOTAL_ROUNDS):
        env = KB.E[env_cat(t)]

        obs = adopter_obs(t)
        adopter = F.normalize((1 - BETA) * adopter + BETA * obs, dim=1)
        a_toks = (adopter @ KB.E.t()).argmax(dim=1)

        # passives read the dominant token before adopters deposit
        dom = None if counts.max() < 1e-8 else int(counts.argmax())
        if dom is not None:
            passive = F.normalize((1 - ALPHA) * passive + ALPHA * KB.E[dom], dim=1)

        counts += torch.bincount(a_toks, minlength=M).float()
        counts *= (1.0 - rho)

        sims.append((passive @ env).mean().item())

    return np.array(sims)


def run_consensus() -> np.ndarray:
    """Cumulative sum of adopter token signals with no forgetting.
    In a static environment, the hub signal matches the dominant token in the
    stigmergic field, so convergence is identical. After an environment shift,
    the accumulated history dominates and the hub is slow to re-align."""
    adopter = F.normalize(torch.randn(K, DIM), dim=1)
    passive = F.normalize(torch.randn(N - K, DIM), dim=1)
    hub = torch.zeros(DIM)
    sims = []

    for t in range(TOTAL_ROUNDS):
        env = KB.E[env_cat(t)]

        obs = adopter_obs(t)
        adopter = F.normalize((1 - BETA) * adopter + BETA * obs, dim=1)
        a_toks = (adopter @ KB.E.t()).argmax(dim=1)

        # passive reads hub before adopters deposit — mirrors stigmergy ordering
        if hub.norm() > 1e-8:
            passive = F.normalize((1 - ALPHA) * passive + ALPHA * F.normalize(hub, dim=0), dim=1)

        hub = hub + KB.E[a_toks].mean(dim=0)

        sims.append((passive @ env).mean().item())

    return np.array(sims)


def avg(fn, *args):
    arr = np.stack([fn(*args) for _ in range(RUNS)])
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    print("Running dynamic-environment experiment...")
    stig = {r: avg(run_stigmergy, r) for r in RHOS}
    c_mean, _ = avg(run_consensus)

    rounds = np.arange(1, TOTAL_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    for rho, color in zip(RHOS, RHO_COLORS):
        m, _ = stig[rho]
        lbl = "Stigmergic w/o decay rate" if rho == 0.0 else f"Stigmergic w/ decay rate {rho}"
        ax.plot(rounds, m, color=color, linewidth=1.6, label=lbl, zorder=3)

    ax.plot(rounds, c_mean, color="0.5", linestyle="--", linewidth=1.2,
            label="Consensus", zorder=2)

    for i in range(1, N_PHASES):
        ax.axvline(i * SHIFT_INTERVAL, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Similarity to current environment", fontsize=10)
    ax.set_ylim(-0.15, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, columnspacing=1.3, handlelength=1.7,
              handletextpad=0.5, labelspacing=0.3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/Fig_dynamic_env.pdf", bbox_inches="tight")
    plt.savefig("figures/Fig_dynamic_env.png", bbox_inches="tight")
    print("Saved figures/Fig_dynamic_env.pdf/.png")


if __name__ == "__main__":
    main()
