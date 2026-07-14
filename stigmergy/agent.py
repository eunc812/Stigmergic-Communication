import torch
import torch.nn.functional as F
from torch import Tensor

from .shared_knowledge_base import SharedKnowledgeBase


class Agent:
    """Stigmergic agent over a shared knowledge base.

    Each round the agent absorbs an observation, reads the field's dominant
    token and pulls its belief toward that category, tokenizes the updated
    belief, and deposits the token only if its belief diverges from the dominant
    by more than theta. Reading before depositing keeps the dominant token from
    oscillating under synchronous updates.
    """

    def __init__(self, agent_id: int, kb: SharedKnowledgeBase,
                 beta: float = 0.05, alpha: float = 0.1, theta: float = 0.3):
        self.id = agent_id
        self.kb = kb
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.belief = torch.zeros(kb.dim)

    def _cosine_distance(self, a: Tensor, b: Tensor) -> float:
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def step(self, obs: Tensor, dominant: int | None) -> int | None:
        if self.beta > 0:
            self.belief = F.normalize((1 - self.beta) * self.belief + self.beta * obs, dim=0)

        if dominant is not None:
            self.belief = F.normalize(
                (1 - self.alpha) * self.belief + self.alpha * self.kb.embed(dominant), dim=0)

        my_token = self.kb.tokenize(self.belief)

        if dominant is None:
            return my_token
        if self._cosine_distance(self.belief, self.kb.embed(dominant)) > self.theta:
            return my_token
        return None
