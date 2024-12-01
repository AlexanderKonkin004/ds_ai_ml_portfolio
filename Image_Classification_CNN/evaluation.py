import torch
from torch.utils.data import DataLoader
from typing import Callable, Tuple, Union, Dict

def evaluate_model(model: torch.nn.Module, dataset: DataLoader,
                   **losses: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[str, float]:

    device = next(model.parameters()).device
    
    results = {name: 0.0 for name in losses}
    model.train(False)
    for inputs, targets in dataset:
        inputs = inputs.to(device)
        targets = targets.to(device)
        for name, loss in losses.items():
            with torch.no_grad():
                preds = model(inputs)
                results[name] += loss(preds.squeeze(dim=1), targets).item() * len(inputs)
    results = {name: loss / len(dataset.dataset)
               for name, loss in results.items()}
    return results


def multiclass_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.argmax(-1)
    return (preds == targets).float().mean()
