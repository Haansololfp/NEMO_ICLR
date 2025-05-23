import torch
from torch.linalg import svdvals

def old_topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(
            torch.argsort(similarities, axis=1)[:, -(i + 1)] == labels
        ) / len(labels)
    return topsum


def topk(similarities, labels, k=5):
    # Ensure k does not exceed the number of examples
    k = min(k, similarities.shape[0])

    # Get the indices of the top k maximum values along each row
    # We use `topk` instead of `argsort` for better performance
    topk_indices = torch.topk(
        similarities, k=k, dim=1, largest=True, sorted=False
    ).indices

    # Check if the true labels are within these top k predictions
    correct = torch.zeros_like(labels, dtype=torch.float)
    for i in range(k):
        correct += (topk_indices[:, i] == labels).float()

    # Compute the average of correct predictions
    topk_accuracy = correct.sum() / len(labels)
    return topk_accuracy

def rankme(embeddings, epsilon = 1e-7):
    """
    https://arxiv.org/pdf/2210.02885
    """
    s = svdvals(embeddings)
    p = s / torch.norm(s, p = 1) + epsilon
    return -torch.sum(p * torch.log(p))