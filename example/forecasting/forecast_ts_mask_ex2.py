#!/usr/bin/env python
# encoding: utf-8

"""

    Examples on incomplete (sparse_fusion) time series forecasting (ITSF).
    The incomplete (sparse_fusion) time series data is represented by a dense tensor and a mask (indicator) tensor.

    (1) Single-source single/multivariate time series forecasting with missing values.

    (2) Multi-source single/multivariate time series forecasting with missing values.

    (3) Global static mask and training dynamic mask.


    Some tips for benefiting from the codes:

    (1) For incomplete time series datasets, this supports models working on both fixed-length and
        vary-length input windows.

        The codes avoid the two problems:
        (1.a) Relative time steps/points mis-alignment, such as irregular time series data.
        (1.b) Randomly dynamic padding on collected vary-length sliding windows, such as irregular time series data.

    (2) Computation efficiency: ITSF can work well on personal computers (PC).
        If dataset is large, it is recommended to load on main memory (RAM).
        If model is large, it is recommended to train on GPU(s).
        If both dataset and model are large, the datasets can be loaded on main memory (RAM) and the model
         can be trained on GPU(s).

"""

import logging, os

import torch
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform
from fast.data import RandomMasker, BlockMasker, VariableMasker, masker_generate
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MAE, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.sparse_fusion import TPatchGNN

from dataset.prepare_xmcdc import load_xmcdc_as_sst, load_xmcdc_as_smt
from dataset.manage_sst_datasets import prepare_sst_datasets
from dataset.manage_smt_datasets import prepare_smt_datasets
from dataset.manage_smc_datasets import prepare_smc_datasets


def ts_mask():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'G:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'cpu'

    """
        Sparse long-sequence time series forecasting problems: sparse_fusion decomposition, shapelet representation
    """

    # xmcdc_filename = '../../dataset/xmcdc/outpatients_2011_2020_1week.csv'  # Built-in dataset
    # train_ds, val_ds, test_ds = load_xmcdc_as_sst(xmcdc_filename, None, True, None, False, 10, 1, 1, 1, (0.7, 0.1, 0.2), ds_device)

    task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex2': True, 'shuffle': True}
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'ETTh1', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'HumanActivity', 3000, 1000, 1, 1000, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'USHCN', 745, 31, 1, 31, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'PhysioNet', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'MIMIC-III-Ext-tPatchGNN', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'MIMIC-III-v1.4', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'MIMIC-IV-v3.1', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'SouthPT_1day', 10, 1, 1, 1, (0.7, 0.1, 0.2), 'inter', ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smc_datasets(data_root, 'PhysioNet', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device)

    """
        Global **static mask**. This simulates the missing mechanism of real world.
        chose: RandomMasker(0.8) | BlockMasker(12, 0.8) | VariableMasker(0.8)
    """

    # masker_generate(RandomMasker(0.8), train_ds.ts_mask, inplace=True)
    # if val_ds is not None:
    #     masker_generate(RandomMasker(0.8), val_ds.ts_mask, inplace=True)
    # if test_ds is not None:
    #     masker_generate(RandomMasker(0.8), test_ds.ts_mask, inplace=True)
    #
    # dynamic_masker = RandomMasker(0.95)

    """
        Overwritable scalers and dynamic scalers. 
    """
    overwrite_scaler = scaler_fit(MinMaxScale(), train_ds.ts, train_ds.ts_mask)
    scaler_transform(overwrite_scaler, train_ds.ts, train_ds.ts_mask, inplace=True, show_progress=True)
    if val_ds is not None:
        scaler_transform(overwrite_scaler, val_ds.ts, val_ds.ts_mask, inplace=True, show_progress=True)
    if test_ds is not None:
        scaler_transform(overwrite_scaler, test_ds.ts, test_ds.ts_mask, inplace=True, show_progress=True)

    # Dynamic scaling while training or evaluation.
    scaler = None # scaler_fit(StandardScale(), train_ds.ts, train_ds.ts_mask)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_modeler = {
        'tPatchGNN': [TPatchGNN, {
            "patch_len": 240, "patch_stride": 240,
            "time_embedding_dim": 20, "hidden_dim": 64,
            "num_layers": 1, "transformer_nhead": 2, "transformer_num_layers": 2,
            "node_vector_dim": 10, "dropout_rate": 0.,
            "supports": None, "gcn_hop": 1, "aggregation": 'cnn'
        }],
    }

    model_cls, user_args = ts_modeler['tPatchGNN']

    common_ds_args = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = model_cls(**combined_args)

    loger = logging.getLogger()
    model = covert_weight_types(model, torch_float_type)
    loger.info(get_model_info(model))
    # loger.info(str(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0005, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel')
    criterion = MAE()
    evaluator = Evaluator(['MAE', 'MSE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler)
    loger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True, # forecast_masker=dynamic_masker,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, 32, None, False, is_online=False)
        loger.info('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, 32, None, False, is_online=False)
        loger.info('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()

    ts_mask()
