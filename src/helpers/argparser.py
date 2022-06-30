# %%
import argparse
from pathlib import Path

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class convertStringToPathAction(argparse.Action):
    """convert string to a Path"""
    def __call__(self, parser, namespace, values, option_string) -> None:
        if isinstance(values, str) or isinstance(values, Path):
            path = Path(values)
            if option_string in ['--out_dir', '--log_dir']:
                path.parent.mkdir(parents=True, exist_ok=True)
                setattr(namespace, self.dest, path)
                return
            if not path.exists():
                raise ValueError("path does not exist")
            setattr(namespace, self.dest, path)
        else:
            raise TypeError("only accepts paths or str.")




DATA = Path('/cluster/scratch/data')
ORI = DATA/'ori'

parser = argparse.ArgumentParser('default')
parser.add_argument('--filtered_data', default=ORI/'filtered', type=Path, action=convertStringToPathAction, help="""path to the filtered data""")
parser.add_argument('--data_root', default=DATA, type=Path, action=convertStringToPathAction, help="""path to the data root folder""")
parser.add_argument('--posthoc_data', default='', type=str, action=convertStringToPathAction, help=""""path to the posthoc data""")
parser.add_argument('--out_dir', default=None, type=str, action=convertStringToPathAction, help=""""output directory""")
parser.add_argument('--log_dir', default=DATA/'models', action=convertStringToPathAction, help='Path to save logs and checkpoints')
parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
parser.add_argument('--batch_size', default=64, type=int, help="batch size")
parser.add_argument('--device', default='cuda', type=str, help="""cuda or not""")
parser.add_argument('--pretrained_weights', default=None, type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
    training (highest LR used during training). The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.
    We recommend tweaking the LR depending on the checkpoint evaluated.""")
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
    for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
parser.add_argument('--log_interval', default=None, type=int, help="Log interval while training.")
parser.add_argument('--avgpool_patchtokens', default=False, type=bool_flag,
    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
    We typically set this to False for ViT-Small and to True with ViT-Base.""")
parser.add_argument('--pin_memory', default=True, type=bool_flag, help='Whether to use pinned memory or not.')
parser.add_argument('--head_path', default=DATA/'models/base_lin_clf/checkpoint.pth.tar', type=str, help='Path to the pretrained classification head')
