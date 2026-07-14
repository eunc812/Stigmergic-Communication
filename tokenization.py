"""Experiment 2 — A shared knowledge base integrates heterogeneous observations.

N agents observe the same concept, each through independent noise. We compare
three ways of integrating the observations, scored as cosine alignment of the
integrated estimate to the true concept:

  1. Raw consensus           : average the raw belief vectors.
  2. Stigmergy w/o shared KB  : each agent tokenizes with its own private KB, so
     the same token index means a different direction for each agent.
  3. Stigmergy w/ shared KB   : all agents tokenize through one shared KB, so the
     dominant token decodes to one concept for everyone.

Raw consensus works at low noise but its centroid degrades as noise grows.
Private-KB tokenization fails even at zero noise, since the symbols carry no
common meaning. Only the shared KB grounds the observations in a common symbol
set and integrates them into the correct concept.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import SharedKnowledgeBase

# Parameters
DIM = 512
M = 10
N = 50            # agents, all observing the same concept
SIGMAS = np.linspace(0.0, 2.5, 26)   # observation-noise sweep
RUNS = 800
SEED = 0

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

KB = SharedKnowledgeBase(num_tokens=M, dim=DIM, seed=0)


def _dominant(tokens: torch.Tensor) -> int:
    return int(torch.bincount(tokens, minlength=M).argmax())


def trial(sigma: float, gen: torch.Generator):
    c = int(torch.randint(0, M, (1,), generator=gen))
    truth = KB.E[c]
    obs = F.normalize(truth.unsqueeze(0) + sigma * torch.randn(N, DIM, generator=gen), dim=1)

    # raw consensus
    cons = (F.normalize(obs.mean(dim=0), dim=0) @ truth).item()

    # private KB per agent, decode the dominant index through each own KB
    priv = F.normalize(torch.randn(N, M, DIM, generator=gen), dim=2)
    t_priv = torch.einsum("nd,nmd->nm", obs, priv).argmax(dim=1)
    dom_p = _dominant(t_priv)
    nokb = (F.normalize(priv[:, dom_p, :], dim=1) @ truth).mean().item()

    # shared KB for all agents
    t_sh = KB.tokenize_batch(obs)
    shared = (KB.E[_dominant(t_sh)] @ truth).item()

    return shared, cons, nokb


def main():
    shared, cons, nokb = [], [], []
    for s in SIGMAS:
        # Common random numbers: re-seed per noise level so every sigma reuses
        # the same concepts and base noise patterns (scaled by sigma). Adjacent
        # points become correlated, giving a smooth monotone curve.
        gen = torch.Generator().manual_seed(SEED)
        r = np.array([trial(s, gen) for _ in range(RUNS)])
        shared.append(r[:, 0].mean())
        cons.append(r[:, 1].mean())
        nokb.append(r[:, 2].mean())
        print(f"sigma={s:4.2f} | shared KB={shared[-1]:.3f} | consensus={cons[-1]:.3f} | no-share KB={nokb[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(SIGMAS, shared, "o-", color="tab:red", linewidth=2.4, markersize=4.5,
            label="Stigmergic w/ shared KB")
    ax.plot(SIGMAS, cons, "s-", color="tab:blue", linewidth=2.2, markersize=4.5,
            label="Raw consensus")
    ax.plot(SIGMAS, nokb, "^--", color="0.5", linewidth=2.0, markersize=4.5,
            label="Stigmergic w/o shared KB")

    ax.set_xlabel("Observation noise level", fontsize=10)
    ax.set_ylabel("Alignment to true concept", fontsize=10)
    ax.set_xlim(0, SIGMAS[-1])
    ax.set_ylim(-0.08, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout(pad=0.4)
    plt.savefig("figures/Fig_tokenization.pdf")
    plt.savefig("figures/Fig_tokenization.png")
    print("Saved figures/Fig_tokenization.pdf/.png")


if __name__ == "__main__":
    main()
