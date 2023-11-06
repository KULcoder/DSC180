
"""
Functions:
    calculate_acc
"""

def calculate_acc(logits, labels):
    """
    Given logits and correct labels:
    Return number of corrections and accuracy
    """
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred)).sum().detach().cpu().item()
    return correct, 100 * correct / len(logits)