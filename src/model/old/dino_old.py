import torch
from torch import nn
from torchvision import transforms as pth_transforms

# Define and load pretrained weights for linear classifier on ImageNet
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels) 
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class ViTModel(torch.nn.Module):
    def __init__(self, vits16, linear_layer):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ViTModel, self).__init__()
        self.vits16=vits16
        self.linear_layer=linear_layer
        self.transform = transform = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.vits16.eval()
        self.linear_layer.eval()

    def forward(self, x, grad=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if grad is False:
          with torch.no_grad():
            x = self.transform(x)
            
            # forward
            intermediate_output = self.vits16.get_intermediate_layers(x, 4)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            output = self.linear_layer(output)
            return output
        else:
          x = self.transform(x)
            
          # forward
          intermediate_output = self.vits16.get_intermediate_layers(x, 4)
          output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
          output = self.linear_layer(output)
          return output

    
def get_dino(device='cuda',
             N_LAST_BLOCKS = 4):
    # Load pretrained weights from PyTorch
    vits = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)
    linear_classifier = LinearClassifier(vits.embed_dim * N_LAST_BLOCKS, num_labels=1000)
    linear_classifier = linear_classifier.cuda()
    linear_state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth")["state_dict"]

    # Update state dict to avoid crash. Workaround.
    linear_state_dict['linear.weight'] = linear_state_dict.pop('module.linear.weight')
    linear_state_dict['linear.bias'] = linear_state_dict.pop('module.linear.bias')

    # Load pre-trained weights
    linear_classifier.load_state_dict(linear_state_dict, strict=True)
    
    
    model = ViTModel(vits16, linear_classifier)
    return model, linear_classifier
