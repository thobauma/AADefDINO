from dino import vision_transformer as vits
from dino.eval_linear import LinearClassifier
from dino import utils


def get_dino(model_name='vit_small', patch_size=16,n_last_blocks=4, avgpool_patchtokens=False, device='cuda',classifier = True, pretrained_classifier = True, num_labels=1000):
    if model_name in vits.__dict__.keys():
        model = vits.__dict__[model_name](patch_size=patch_size, num_classes=0)
        embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        utils.load_pretrained_weights(model, "", "", model_name, patch_size)
    else:
        print(f"Unknow architecture: {model_name}")
        sys.exit(1)
    model.cuda()
    model.eval()
    print(f"Model {model_name} built.")
    if classifier != True:
        return model
    
    linear_classifier = LinearClassifier(embed_dim, num_labels=num_labels)
    linear_classifier = linear_classifier.cuda()
    if pretrained_classifier:
        utils.load_pretrained_linear_weights(linear_classifier, model_name, patch_size)
#    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[arch.gpu])
    return model, linear_classifier

