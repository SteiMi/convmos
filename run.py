from __future__ import print_function
from argparse import ArgumentParser
from functools import partial
from glob import glob
from os.path import join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor
import xarray as xr

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
from evaluate import calculate_metrics, mean_metrics

from model import ConvMOS
from utils import write_results_file


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
    inp: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE loss which ignores parts of the tensor. In our case, we ignore sea cells, as there are no labels there."""
    se = (inp - target) ** 2
    mse_per_cell = torch.mean(se, 0)
    masked = torch.mul(mse_per_cell, mask)
    return masked.sum() / mask.sum()


def run(
    train_batch_size: int,
    val_batch_size: int,
    epochs: int,
    lr: float,
    model_name: str,
    architecture: str,
    momentum: float,
    log_interval: int,
    log_dir: str,
    save_dir: str,
    save_step: int,
    val_step: int,
    num_workers: int,
    patience: int,
    eval_only: bool = False,
    overfit_on_few_samples: bool = False,
):
    train_loader, val_loader, test_loader = get_data_loaders(
        train_batch_size,
        val_batch_size,
        num_workers=num_workers,
        overfit_on_few_samples=overfit_on_few_samples,
    )

    models_available = {
        'convmos': ConvMOS
    }

    model = models_available[model_name](architecture=architecture)
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device=device)

    # E-OBS only provides observational data for land so we need to use a mask to avoid fitting on the sea
    land_mask_np = np.load('remo_eobs_land_mask.npy')
    # Convert booleans to 1 and 0, and convert numpy array to torch Tensor
    land_mask = torch.from_numpy(1 * land_mask_np).to(device)
    print('Land mask:')
    print(land_mask)
    loss_fn = partial(masked_mse_loss, mask=land_mask)

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    metrics = {
        'rmse': RootMeanSquaredError(),
        'mae': MeanAbsoluteError(),
        'mse': Loss(loss_fn),
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
        val_loss = engine.state.metrics['mse']
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

    earlystop_handler = EarlyStopping(
        patience=patience, score_function=score_function, trainer=trainer
    )
    val_evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

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

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                "".format(
                    engine.state.epoch, iter, len(train_loader), engine.state.output
                )
            )
            writer.add_scalar(
                "training/loss", engine.state.output, engine.state.iteration
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_rmse = metrics['rmse']
        avg_mae = metrics['mae']
        avg_mse = metrics['mse']
        print(
            "Training Results - Epoch: {}  Avg RMSE: {:.2f} Avg loss: {:.2f} Avg MAE: {:.2f}".format(
                engine.state.epoch, avg_rmse, avg_mse, avg_mae
            )
        )
        writer.add_scalar("training/avg_loss", avg_mse, engine.state.epoch)
        writer.add_scalar("training/avg_rmse", avg_rmse, engine.state.epoch)
        writer.add_scalar("training/avg_mae", avg_mae, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=val_step))
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        avg_rmse = metrics['rmse']
        avg_mae = metrics['mae']
        avg_mse = metrics['mse']
        print(
            "Validation Results - Epoch: {}  Avg RMSE: {:.2f} Avg loss: {:.2f} Avg MAE: {:.2f}".format(
                engine.state.epoch, avg_rmse, avg_mse, avg_mae
            )
        )
        writer.add_scalar("validation/avg_loss", avg_mse, engine.state.epoch)
        writer.add_scalar("validation/avg_rmse", avg_rmse, engine.state.epoch)
        writer.add_scalar("validation/avg_mae", avg_mae, engine.state.epoch)

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
    print('Loading best checkpoint from', best_checkpoint)
    checkpoint = torch.load(
        join(save_dir, best_checkpoint_handler.last_checkpoint), map_location=device
    )
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    writer.close()

    val_preds = predict_data(trainer.state.epoch, val_loader)
    test_preds = predict_data(trainer.state.epoch, test_loader)
    val_res = mean_metrics(calculate_metrics(val_preds.pred, val_preds.target))
    test_res = mean_metrics(calculate_metrics(test_preds.pred, test_preds.target))

    # val_evaluator.run(val_loader)
    results = {}
    # Store the config, ...
    results.update(
        {section_name: dict(config[section_name]) for section_name in config.sections()}
    )
    # ... the last training metrics,
    results.update({f'train_{k}': v for k, v in train_evaluator.state.metrics.items()})
    # ... the last validation metrics from torch,
    results.update(
        {f'val_torch_{k}': v for k, v in val_evaluator.state.metrics.items()}
    )
    # ... the validation metrics that I calculate,
    results.update({f'val_{k}': v for k, v in val_res.items()})
    # ... asnd the test metrics that I calculate
    results.update({f'test_{k}': v for k, v in test_res.items()})
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

    args = parser.parse_args()

    run(
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.model,
        args.architecture,
        args.momentum,
        args.log_interval,
        args.log_dir,
        args.save_dir,
        args.save_step,
        args.val_step,
        args.num_workers,
        args.patience,
        eval_only=args.eval_only,
        overfit_on_few_samples=args.overfit_on_few_samples,
    )
