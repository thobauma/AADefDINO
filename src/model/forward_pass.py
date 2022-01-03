import torch

# Performs a forward pass given a sample `inp` and a classifier.
@torch.no_grad()
def forward_pass(inp, model, linear_classifier, n=4):
    intermediate_output = model.get_intermediate_layers(inp, n)
    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
    output = linear_classifier(output)
    return output.argmax(1)