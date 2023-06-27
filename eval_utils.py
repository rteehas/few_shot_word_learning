import torch



def compute_exact_match(pred, true_answer):
    predicted = pred.strip().lower()
    correct = true_answer.strip().lower()

    return predicted == correct

def load_model_partial(fname, model):
    model_dict = model.state_dict()
    state_dict = torch.load(fname)
    partial_states = {k:v for k, v in state_dict["model_state_dict"].items() if "emb_gen" in k or "memory" in k}
    model_dict.update(partial_states)
    model.load_state_dict(model_dict)
    return model

