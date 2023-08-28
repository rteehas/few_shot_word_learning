from transformers.modeling_outputs import MaskedLMOutput

class MaskLMOutputWithNewToken(MaskedLMOutput):

    def __init__(self, loss, logits, hidden_states, attentions, new_token_loss, memories):

        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.new_token_loss = new_token_loss
        self.memories=memories