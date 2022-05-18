from dino import vision_transformer as vits
from dino.eval_linear import LinearClassifier
from dino import utils
import torch
from torch import nn
from torchvision import transforms as pth_transforms

# Taken from DINO official repo
# Official repo: https://github.com/facebookresearch/dino
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=9):
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


def get_dino(model_name='vit_small', patch_size=16, n_last_blocks=4, avgpool_patchtokens=False, device='cuda',classifier = True, pretrained_classifier = True, num_labels=1000, args=None):
    if args is not None:
        model_name = args.arch
        patch_size = args.patch_size
        n_last_blocks = args.n_last_blocks
        avgpool_patchtokens = args.avgpool_patchtokens
        device = args.device
        num_labels = args.num_labels
    if model_name in vits.__dict__.keys():
        model = vits.__dict__[model_name](patch_size=patch_size, num_classes=0)
        embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        utils.load_pretrained_weights(model, "", "", model_name, patch_size)
    else:
        print(f"Unknow architecture: {model_name}")
        sys.exit(1)
    if device == 'cuda':
        model.cuda()
    model.eval()
    print(f"Model {model_name} built.")
    if classifier != True:
        return model
    print("Embed dim {}".format(embed_dim))
    linear_classifier = LinearClassifier(embed_dim, num_labels=num_labels)
    linear_classifier = linear_classifier.cuda()
    if pretrained_classifier:
        utils.load_pretrained_linear_weights(linear_classifier, model_name, patch_size)
#    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[arch.gpu])
    return model, linear_classifier

normalize = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

class ViTWrapper(torch.nn.Module):
    def __init__(self, vits16, linear_layer, transform=normalize, device='cuda', n_last_blocks=4, avgpool_patchtokens=False, ):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ViTWrapper, self).__init__()
        self.vits16=vits16
        self.linear_layer=linear_layer
        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens
        self.transform = transform
        
        self.vits16.eval()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.transform is not None:
            x = self.transform(x)

        # forward
        intermediate_output = self.vits16.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)

        output = self.linear_layer(output)
        return output

    def set_weights_for_training(self):
      self.linear_layer.train()
    
    def set_weights_for_testing(self):
      self.linear_layer.eval() 
