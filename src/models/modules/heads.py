import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, dim: int, num_labels: int):
        super().__init__()

        self.dim = dim
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.linear(hidden_states).view(-1, self.num_labels)
        return F.softmax(logits, dim=-1)

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.linear(hidden_states)
        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
