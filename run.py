from __future__ import print_function
from argparse import ArgumentParser
from functools import partial
from glob import glob
from os.path import join
from typing import Callable, Dict, Optional
from denseweight import DenseWeight
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor
import xarray as xr

from matplotlib import pyplot as plt

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError(
        "No tensorboardX package is found. Please install with the command: \npip install tensorboardX"
    )

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    global_step_from_engine,
    TerminateOnNan,
)
from ignite.metrics import Loss, MeanAbsoluteError, RootMeanSquaredError

from config_loader import config
from dataset_direct import RemoDataset
from evaluate import calculate_metrics, calculate_metrics_flat, mean_metrics

from models.unet import UNet
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model import ConvMOS
from utils import init_mpl, set_size, write_results_file


def get_data_loaders(
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int = 0,
    overfit_on_few_samples: bool = False,
):

    # create_standardization_transform=True lets the Dataset create a torchvision "Normalize" transform based on the
    # mean/std of the training features. It also applies the newly created transform on all __get_item__ calls.
    train_ds = RemoDataset(
        transform=ToTensor(),
        target_transform=ToTensor(),
        train=True,
        debug_with_few_samples=overfit_on_few_samples,
        create_standardization_transform=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
    )

    if overfit_on_few_samples:
        # In this debug setting we want simply want to overfit and therefore test on the training data
        val_loader = train_loader
        test_loader = train_loader
    else:

        # The train dataset created a torchvision "Normalize" transform with the means/stds of the training data.
        # We want to use the same transform to standardize the test features.
        input_transform = Compose([ToTensor(), train_ds.standardize_transform])

        val_loader = DataLoader(
            RemoDataset(
                transform=input_transform,
                target_transform=ToTensor(),
                train=False,
                val=True,
                debug_with_few_samples=overfit_on_few_samples,
            ),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            RemoDataset(
                transform=input_transform,
                target_transform=ToTensor(),
                train=False,
                val=False,
                debug_with_few_samples=overfit_on_few_samples,
            ),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, test_loader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def masked_mse_loss(
    inp: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: str,
    dw: Optional[DenseWeight] = None,
) -> torch.Tensor:
    """MSE loss which ignores parts of the tensor. In our case, we ignore sea cells, as there are no labels there. Also allows for per-cell-and-time loss weighting via DenseWeight (making it a DenseLoss). """
    se = (inp - target) ** 2
    # We unfortunately have to move to cpu for the denseweight part
    if dw is not None:
        weights = (
            target.detach().clone().cpu().apply_(lambda y: dw.eval_single(y)).to(device)
        )
        se *= weights
    mse_per_cell = torch.mean(se, 0)
    masked = torch.mul(mse_per_cell, mask)
    return masked.sum() / mask.sum()


def extreme_masked_mse_loss(
    inp: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss which ignores parts of the tensor. In our case, we ignore sea cells, as there are no labels there and also targets < 50 mm. This does not calculate metrics per cell but overall. """
    inp_filtered = inp[:, :, mask]
    target_filtered = target[:, :, mask]
    # Filter samples corresponding to extreme precipitation
    extreme_mask = target_filtered >= 50.0
    # complete_mask = extreme_mask | mask
    se = (inp_filtered - target_filtered) ** 2
    extreme_se = torch.mul(se, extreme_mask)
    return torch.sum(extreme_se) / extreme_mask.sum()


def run(
    train_batch_size: int,
    val_batch_size: int,
    epochs: int,
    lr: float,
    model_name: str,
    architecture: str,
    global_module: str,
    local_module: str,
    output_activation: str,
    momentum: float,
    log_interval: int,
    log_dir: str,
    save_dir: str,
    save_step: int,
    val_step: int,
    num_workers: int,
    patience: int,
    land_mask: str,
    early_stopping: bool = True,
    eval_only: bool = False,
    overfit_on_few_samples: bool = False,
    weighted_loss: bool = False,
    alpha: float = 0.0,
):
    train_loader, val_loader, test_loader = get_data_loaders(
        train_batch_size,
        val_batch_size,
        num_workers=num_workers,
        overfit_on_few_samples=overfit_on_few_samples,
    )

    models_available: Dict[str, Callable] = {
        'convmos': partial(
            ConvMOS,
            architecture=architecture,
            global_module=global_module,
            local_module=local_module,
            output_activation=output_activation,
        ),
        'UNet': UNet,
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152,
    }

    model = models_available[model_name]()
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device=device)

    early_stopping_metric = 'mse'
    land_mask_np = None
    if land_mask:
        # E-OBS only provides observational data for land so we need to use a mask to avoid fitting on the sea
        land_mask_np = np.load(land_mask)
        # convert numpy array to torch Tensor
        land_mask_t = torch.from_numpy(land_mask_np).to(device)
        print('Loaded land mask:')
        print(land_mask_t)
        dw = None

        if weighted_loss:
            # Stop based on extreme_mse when using DenseLoss
            early_stopping_metric = 'extreme_mse'

            dw = DenseWeight(alpha=alpha)
            # I calculate that from the masked Y to avoid too many 0s or nans or whatever
            y_mask = np.repeat(
                land_mask_np[np.newaxis, np.newaxis, :, :],
                train_loader.dataset.Y.shape[0],
                axis=0,
            )
            masked_y = np.ma.masked_array(
                train_loader.dataset.Y, mask=np.invert(y_mask)
            )
            compressed_y = masked_y.compressed()
            ys = compressed_y.flatten()
            # ys = np.reshape(compressed_y, (-1, 1))
            # Sort for the plots
            ys = np.sort(ys)
            print(ys.shape)
            dw.fit(ys)

            # Plot distribution and weighting
            init_mpl(usetex=True)
            fig, ax = plt.subplots(figsize=set_size(fraction=1))
            ax.set_ylabel('Density')
            # ax.set_ylim(0.0, 0.009)
            par1 = ax.twinx()
            par1.set_ylabel('Weight')
            ax.hist(ys, density=True, log=True, bins=300)
            # ax.hist(ys, density=True, log=False, bins=300)
            ax.plot(ys, dw.y_dens.flatten())
            par1.plot(ys, dw.weights.flatten())
            ax.set_xlabel('Precipitation [mm]')
            # plt.xlim(left=-1, right=10)
            fig.tight_layout()
            plt.savefig(
                join(save_dir, 'dw_histogram.pdf'), bbox_inches='tight', dpi=200
            )

        loss_fn = partial(masked_mse_loss, mask=land_mask_t, device=device, dw=dw)
        extreme_loss_fn = partial(extreme_masked_mse_loss, mask=land_mask_t)
    else:
        loss_fn = torch.nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    metrics = {
        'rmse': RootMeanSquaredError(),
        'mae': MeanAbsoluteError(),
        'mse': Loss(loss_fn),
        'extreme_mse': Loss(extreme_loss_fn),
    }
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(save_dir, create_dir=True, require_empty=False),
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=save_step), checkpoint_handler
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def score_function(engine):
        val_loss = engine.state.metrics[early_stopping_metric]
        return -val_loss

    best_checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(save_dir, create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix='best',
        score_function=score_function,
        score_name='val_loss',
        global_step_transform=global_step_from_engine(trainer),
    )
    val_evaluator.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    if early_stopping:
        print(f'Early Stopping active based on validation {early_stopping_metric}')
        earlystop_handler = EarlyStopping(
            patience=patience, score_function=score_function, trainer=trainer
        )
        val_evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)
    else:
        print('Early Stopping disabled.')

    # Maybe load model
    checkpoint_files = glob(join(save_dir, 'checkpoint_*.pt'))
    if len(checkpoint_files) > 0:
        # latest_checkpoint_file = sorted(checkpoint_files)[-1]
        epoch_list = [int(c.split('.')[0].split('_')[-1]) for c in checkpoint_files]
        last_epoch = sorted(epoch_list)[-1]
        latest_checkpoint_file = join(save_dir, f'checkpoint_{last_epoch}.pt')
        print('Loading last checkpoint', latest_checkpoint_file)
        last_epoch = int(latest_checkpoint_file.split('.')[0].split('_')[-1])
        if last_epoch >= epochs:
            print('Training was already completed')
            eval_only = True
            # return

        checkpoint = torch.load(latest_checkpoint_file, map_location=device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_rmse = metrics['rmse']
        avg_mae = metrics['mae']
        avg_mse = metrics['mse']
        avg_extreme_mse = metrics['extreme_mse']
        print(
            "Training Results - Epoch: {}  Avg RMSE: {:.2e} Avg loss: {:.2e} Avg extreme MSE: {:.2e}".format(
                engine.state.epoch, avg_rmse, avg_mse, avg_extreme_mse
            )
        )
        writer.add_scalar("training/avg_loss", avg_mse, engine.state.epoch)
        writer.add_scalar("training/avg_rmse", avg_rmse, engine.state.epoch)
        writer.add_scalar("training/avg_mae", avg_mae, engine.state.epoch)
        writer.add_scalar("training/avg_extreme_mse", avg_extreme_mse, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=val_step))
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        avg_rmse = metrics['rmse']
        avg_mae = metrics['mae']
        avg_mse = metrics['mse']
        avg_extreme_mse = metrics['extreme_mse']
        print(
            "Validation Results - Epoch: {}  Avg RMSE: {:.2e} Avg loss: {:.2e} Avg extreme MSE: {:.2e}".format(
                engine.state.epoch, avg_rmse, avg_mse, avg_extreme_mse
            )
        )
        writer.add_scalar("validation/avg_loss", avg_mse, engine.state.epoch)
        writer.add_scalar("validation/avg_rmse", avg_rmse, engine.state.epoch)
        writer.add_scalar("validation/avg_mae", avg_mae, engine.state.epoch)
        writer.add_scalar("validation/avg_extreme_mse", avg_extreme_mse, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=save_step))
    def log_model_weights(engine):
        for name, param in model.named_parameters():
            writer.add_histogram(f"model/weights_{name}", param, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=save_step))
    def regularly_predict_val_data(engine):
        predict_data(engine.state.epoch, val_loader)

    def predict_data(epoch: int, data_loader) -> xr.Dataset:
        # Predict all test data points and write the predictions
        print(f'Predicting {data_loader.dataset.mode} data...')
        data_loader_iter = iter(data_loader)
        pred_np = None
        for i in range(len(data_loader)):
            x, y = next(data_loader_iter)
            # print(x)
            pred = (
                model.forward(x.to(device=device))
                .to(device='cpu')
                .detach()
                .numpy()[:, 0, :, :]
            )
            # print('=======================================')
            # print(pred)
            if pred_np is None:
                pred_np = pred
            else:
                pred_np = np.concatenate((pred_np, pred), axis=0)

        preds = xr.Dataset(
            {
                'pred': (['time', 'lat', 'lon'], pred_np),
                'input': (['time', 'lat', 'lon'], data_loader.dataset.X),
                'target': (['time', 'lat', 'lon'], data_loader.dataset.Y[:, :, :, 0]),
            },
            coords={
                'time': data_loader.dataset.times,  # list(range(len(val_loader.dataset))),
                'lon_var': (
                    ('lat', 'lon'),
                    data_loader.dataset.lons[0],
                ),  # list(range(x.shape[-2])),
                'lat_var': (('lat', 'lon'), data_loader.dataset.lats[0]),
            },  # list(range(x.shape[-1]))}
        )

        preds.to_netcdf(
            join(save_dir, f'predictions_{data_loader.dataset.mode}_{epoch}.nc')
        )
        return preds

    # kick everything off
    if not eval_only:
        trainer.run(train_loader, max_epochs=epochs)

    # Load best model
    best_checkpoint = best_checkpoint_handler.last_checkpoint
    train_epochs = best_checkpoint.split('_')[2]
    print('Loading best checkpoint from', best_checkpoint)
    checkpoint = torch.load(
        join(save_dir, best_checkpoint_handler.last_checkpoint), map_location=device
    )
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    writer.close()

    # Manually calculate metrics like in the evaluation script
    val_outputs = predict_data(trainer.state.epoch, val_loader)
    test_outputs = predict_data(trainer.state.epoch, test_loader)
    val_res = mean_metrics(calculate_metrics(val_outputs.pred, val_outputs.target))
    test_res = mean_metrics(calculate_metrics(test_outputs.pred, test_outputs.target))

    # Calculate metrics specifically for events which the DWD would consider at least "Ergiebiger Dauerregen" meaning
    # precipitation >= 50 l/m^2 = 50 mm/m^2
    # We calculate these extreme metrics not per cell but over all cells, since such events are too rare per cell to
    # get reliable results.
    if land_mask_np is not None:
        val_targets = val_outputs.target.values[:, land_mask_np].ravel()
        val_preds = val_outputs.pred.values[:, land_mask_np].ravel()
        test_targets = test_outputs.target.values[:, land_mask_np].ravel()
        test_preds = test_outputs.pred.values[:, land_mask_np].ravel()
    else:
        val_targets = val_outputs.target.values.ravel()
        val_preds = val_outputs.pred.values.ravel()
        test_targets = test_outputs.target.values.ravel()
        test_preds = test_outputs.pred.values.ravel()

    # Filter samples corresponding to extreme precipitation
    extreme_val_targets = val_targets[val_targets >= 50.0]
    extreme_val_preds = val_preds[
        val_targets >= 50.0
    ]  # use only preds where the corresponding target is extreme
    extreme_test_targets = test_targets[test_targets >= 50.0]
    extreme_test_preds = test_preds[
        test_targets >= 50.0
    ]  # use only preds where the corresponding target is extreme

    extreme_val_res = calculate_metrics_flat(extreme_val_preds, extreme_val_targets)
    extreme_test_res = calculate_metrics_flat(extreme_test_preds, extreme_test_targets)

    # val_evaluator.run(val_loader)
    results = {}
    # Store the config, ...
    results.update(
        {section_name: dict(config[section_name]) for section_name in config.sections()}
    )
    # ... how many epochs we have trained,
    results['train_epochs'] = train_epochs
    # ... the last training metrics,
    results.update({f'train_{k}': v for k, v in train_evaluator.state.metrics.items()})
    # ... the last validation metrics from torch,
    results.update(
        {f'val_torch_{k}': v for k, v in val_evaluator.state.metrics.items()}
    )
    # ... the validation metrics that I calculate,
    results.update({f'val_{k}': v for k, v in val_res.items()})
    # ... the test metrics that I calculate,
    results.update({f'test_{k}': v for k, v in test_res.items()})
    # ... the validation metrics for extreme samples that I calculate,
    results.update({f'extreme_val_{k}': v for k, v in extreme_val_res.items()})
    # ... and the test metrics for extreme samples that I calculate
    results.update({f'extreme_test_{k}': v for k, v in extreme_test_res.items()})
    write_results_file(join('results', 'results.json'), pd.json_normalize(results))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.getint('NN', 'batch_size'),
        help='input batch size for training (default: 64)',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=config.getint('NN', 'batch_size'),
        help='input batch size for validation (default: 1000)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.getint('NN', 'training_epochs'),
        help='number of epochs to train (default: 1000)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=config.getfloat('NN', 'learning_rate'),
        help='learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--early_stopping',
        type=str,
        default=config.get('NN', 'early_stopping'),
        help='Stop early after validation metrics stop improving',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=config.get('NN', 'model'),
        help='Which model to use',
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default=config.get('NN', 'architecture'),
        help='Which architecture to use',
    )
    parser.add_argument(
        '--global_module',
        type=str,
        default=config.get('NN', 'global_module'),
        help='Which type of global module to use (ConvMOS)',
    )
    parser.add_argument(
        '--local_module',
        type=str,
        default=config.get('NN', 'local_module'),
        help='Which type of local module to use (ConvMOS)',
    )
    parser.add_argument(
        '--output_activation',
        type=str,
        default=config.get('NN', 'output_activation'),
        help='Which type of local module to use (ConvMOS)',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers dataloaders (default: 0)',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=40,
        help='Patience for early stopping (default: 40)',
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=join(config.get('NN', 'scratch'), config.get('SD', 'model_name')),
        help="log directory for Tensorboard log output",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(config.get('NN', 'scratch'), config.get('SD', 'model_name')),
        help="save directory for checkpoint output",
    )
    parser.add_argument(
        '--save_step',
        type=int,
        default=config.getint('NN', 'save_step'),
        help='After how many epochs is the model supposed to be saved',
    )
    parser.add_argument(
        '--val_step',
        type=int,
        default=config.getint('NN', 'val_step'),
        help='After how many epochs is the model supposed to be validated',
    )
    parser.add_argument(
        '--overfit_on_few_samples',
        action='store_true',
        help='Overfit on few samples for debugging purposes',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Use the last weights and evaluate them without training',
    )
    parser.add_argument(
        '--land_mask',
        type=str,
        default=config.get('Paths', 'land_mask'),
        help='Calculate loss only for land cells using the provided land mask',
    )
    parser.add_argument(
        '--weighted_loss',
        type=str,
        default=config.get('NN', 'weighted_loss'),
        help='Use DenseLoss for training',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=config.getfloat('NN', 'alpha'),
        help='DenseLoss alpha',
    )

    args = parser.parse_args()

    run(
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.model,
        args.architecture,
        args.global_module,
        args.local_module,
        args.output_activation,
        args.momentum,
        args.log_interval,
        args.log_dir,
        args.save_dir,
        args.save_step,
        args.val_step,
        args.num_workers,
        args.patience,
        early_stopping=args.early_stopping == 'True',
        land_mask=args.land_mask,
        eval_only=args.eval_only,
        overfit_on_few_samples=args.overfit_on_few_samples,
        weighted_loss=args.weighted_loss == 'True',
        alpha=args.alpha,
    )
