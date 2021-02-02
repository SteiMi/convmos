"""
Export the best checkpoint of a model run to a onnx model.
Usage: provide the config.ini for the model to export as "config.ini" in the same directory as this script and
run this script.
"""
from __future__ import print_function
from argparse import ArgumentParser
from functools import partial
from glob import glob
from os.path import join, sep
import numpy as np
import torch
from torch.optim import Adam

from ignite.engine import create_supervised_trainer
from ignite.handlers import Checkpoint

from config_loader import config
from model import GlobalNet, LocalNet, ConvMOS
from run import masked_mse_loss


def export(lr: float, model_name: str, save_dir: str):

    models_available = {
        'globalnet': GlobalNet,
        'localnet': LocalNet,
        'convmos': ConvMOS,
    }

    model = models_available[model_name]()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device=device)

    # E-OBS only provides observational data for land so we need to use a mask to avoid fitting on the sea
    land_mask_np = np.load('remo_eobs_land_mask.npy')
    # Convert booleans to 1 and 0, and convert numpy array to torch Tensor
    land_mask = torch.from_numpy(1 * land_mask_np).to(device)
    loss_fn = partial(masked_mse_loss, mask=land_mask)

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}

    checkpoint_files = glob(join(save_dir, 'best_checkpoint_*.pt'))
    if len(checkpoint_files) > 0:
        # Parse something like:
        # /scratch/scratch_remo/APRL-rr-11-11-sdnext-loglonet-prec-ger11-maskedloss-7/best_checkpoint_194_val_loss=-11.8250.pt
        # Sorry
        epoch_to_score = {
            int(c.split(sep)[-1].split('_')[2]): float(
                c.split(sep)[-1].split('=')[-1][:-3]
            )
            for c in checkpoint_files
        }
        print(epoch_to_score)
        best_epoch = max(epoch_to_score, key=epoch_to_score.get)
        best_checkpoint_file = next(
            cf
            for cf in checkpoint_files
            if int(cf.split(sep)[-1].split('_')[2]) == best_epoch
        )
        print('Loading best checkpoint', best_checkpoint_file)

        checkpoint = torch.load(best_checkpoint_file, map_location=device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
    else:
        print(
            'ERROR: cannot find any files matching',
            join(save_dir, 'best_checkpoint_*.pt'),
        )
        return

    # This uses all aux variables, the temperature/precipitation, and elevation
    input_depth = (
        len(list(filter(None, config.get('DataOptions', 'aux_variables').split(','))))
        + 2
    )
    input_width = config.getint('NN', 'input_width')
    input_height = config.getint('NN', 'input_height')

    dummy_input = torch.randn(1, input_depth, input_width, input_height, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        join(save_dir, 'convmos.onnx'),
        verbose=True,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--lr',
        type=float,
        default=config.getfloat('NN', 'learning_rate'),
        help='learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=config.get('NN', 'model'),
        help='Which model to use',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(config.get('NN', 'scratch'), config.get('SD', 'model_name')),
        help="save directory for checkpoint output",
    )

    args = parser.parse_args()

    export(args.lr, args.model, args.save_dir)
