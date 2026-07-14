import torch


class SharedField:
    """Token-frequency table. Agents deposit token indices, counts decay each
    round, and the most frequent token is the dominant one."""

    def __init__(self, num_tokens: int, decay: float = 0.9):
        self.counts = torch.zeros(num_tokens)
        self.decay = decay

    def is_empty(self) -> bool:
        return self.counts.max().item() < 1e-8

    def dominant(self) -> int | None:
        return None if self.is_empty() else int(self.counts.argmax())

    def deposit(self, tokens) -> None:
        if tokens is None or len(tokens) == 0:
            return
        idx = torch.as_tensor(tokens, dtype=torch.long)
        self.counts += torch.bincount(idx, minlength=self.counts.numel()).float()

    def evaporate(self) -> None:
        self.counts *= self.decay
