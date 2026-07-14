import torch
import torch.nn.functional as F
from torch import Tensor

# CIFAR-10 names, used only as readable labels.
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class SharedKnowledgeBase:
    """Shared vocabulary of category directions.

    Categories are random unit vectors fixed by a seed and shared by all agents
    and the field. Pass orthonormal=True to make them exactly orthogonal.
    """

    def __init__(self, num_tokens: int = 10, dim: int = 512,
                 seed: int = 0, names: list[str] | None = None,
                 orthonormal: bool = False):
        g = torch.Generator().manual_seed(seed)
        E = F.normalize(torch.randn(num_tokens, dim, generator=g), dim=1)
        if orthonormal:
            q, _ = torch.linalg.qr(E.t())
            E = q.t()
        self.E = E
        self.M, self.dim = self.E.shape
        self.names = names if names is not None else [f"c{i}" for i in range(num_tokens)]

    def tokenize(self, belief: Tensor) -> int:
        return int((self.E @ F.normalize(belief, dim=0)).argmax())

    def tokenize_batch(self, B: Tensor) -> Tensor:
        return (F.normalize(B, dim=1) @ self.E.t()).argmax(dim=1)

    def embed(self, token: int) -> Tensor:
        return self.E[token]
