import torch
from torch.distributions import MultivariateNormal



def zero_init(model, tokenizer, new_tokens):
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight[-len(new_tokens):, :] = 0.
        model.get_output_embeddings().weight[-len(new_tokens):, :] = 0.
    return model, tokenizer

def mean_init(model, tokenizer, new_tokens):
    with torch.no_grad():
        input_mean = model.get_input_embeddings().weight.mean(dim=0)
        output_mean = model.get_output_embeddings().weight.mean(dim=0)

    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight[-len(new_tokens):, :] = input_mean
        model.get_output_embeddings().weight[-len(new_tokens):, :] = output_mean
    return model, tokenizer

def random_mean_init(model, tokenizer, new_tokens):
    with torch.no_grad():
        input_mean = model.get_input_embeddings().weight.mean(dim=0)
        output_mean = model.get_output_embeddings().weight.mean(dim=0)
        input_cov = torch.cov(model.get_input_embeddings().weight)
        output_cov = torch.cov(model.get_output_embeddings().weight)

    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        for tok in new_tokens:
            idx = tokenizer.convert_tokens_to_ids(tok)
            model.get_input_embeddings().weight[idx, :] = MultivariateNormal(input_mean, input_cov).sample()
            model.get_output_embeddings().weight[idx, :] = MultivariateNormal(output_mean, output_cov).sample()

    return model, tokenizer

def default_init(model, tokenizer, new_tokens):
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer