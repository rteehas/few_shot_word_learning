import torch



def compute_exact_match(pred, true_answer):
    predicted = pred.strip().lower()
    correct = true_answer.strip().lower()

    return predicted == correct
