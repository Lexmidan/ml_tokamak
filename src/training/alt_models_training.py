import os
from pathlib import Path
import re
import time 
from datetime import datetime
import json

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pytorch_lightning as pl

from ..models import alt_models as am
from ..utils import confinement_mode_classifier as cmc
from LHmode_classifier import get_project_root


def train_and_test_alt_model(signal_name = 'divlp',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            dpoints_in_future = 160,
                            sampling_freq = 300,
                            batch_size = 512,
                            num_workers = 32,
                            num_epochs = 1,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = '',
                            random_seed = 42,
                            exponential_elm_decay = False,
                            num_classes = 3,
                            weight_decay = 0.01,
                            use_ELMs = True,
                            no_L_mode = False,
                            only_ELMs = False,
                            test_df_contains_val_df=True,
                            test_run=False):
    """ 
    Trains and tests alternative model on given signal.
    """
    # Set up number of input channels. If mc, then we use all mirnov coils, else just one signal.
    if signal_name == 'mc' or signal_name == 'mc_h_alpha':
        in_channels = 4
    else:
        in_channels = 1

    pl.seed_everything(random_seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    path = get_project_root()
    comment_for_model_name = architecture  + comment_for_model_name
    
    # Load data
    shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
    shot_for_alt = shot_usage[shot_usage['used_for_alt']]
    shot_numbers = shot_for_alt['shot']
    shots_for_testing = shot_for_alt[shot_for_alt['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_alt[shot_for_alt['used_as'] == 'val']['shot']
    shots_for_training = shot_for_alt[shot_for_alt['used_as'] == 'train']['shot']
    
    if test_run:
        shots_for_testing = shots_for_testing[:5]
        shots_for_validation = shots_for_validation[:5]
        shots_for_training = shots_for_training[:5]

    if test_df_contains_val_df:
        shots_for_testing = pd.concat([shots_for_testing, shots_for_validation])

    shot_df, test_df, val_df, train_df = am.split_df(path, shot_numbers, shots_for_training, shots_for_testing, 
                                                     shots_for_validation, use_ELMs=use_ELMs, 
                                                     signal_name=signal_name, sampling_freq=sampling_freq, 
                                                     exponential_elm_decay=exponential_elm_decay, no_L_mode=no_L_mode,
                                                     only_ELMs=only_ELMs)


    # Create dataloaders
    train_dataloader = am.get_dloader(train_df, batch_size=batch_size, 
                               balance_data=True, shuffle=False, 
                               signal_window=signal_window,
                               signal_name=signal_name,
                               num_workers=num_workers)

    val_dataloader = am.get_dloader(val_df, batch_size=batch_size, 
                                balance_data=True, shuffle=False, 
                                signal_window=signal_window,
                                signal_name=signal_name,
                                num_workers=num_workers)

    test_dataloader = am.get_dloader(test_df, batch_size=batch_size, 
                                balance_data=False, shuffle=False, 
                                signal_window=signal_window,
                                signal_name=signal_name,
                                num_workers=num_workers)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # Create tensorboard writer
    timestamp =  datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name
    writer = SummaryWriter(f'runs/{timestamp}')
    model_path = Path(f'{path}/runs/{timestamp}/model.pt')

    # Create model
    untrained_model = am.select_model_architecture(architecture=architecture, window=signal_window, 
                                                   num_classes=num_classes, in_channels=in_channels)
    untrained_model = untrained_model.to(device)

    # Write model graph to tensorboard
    sample_input = next(iter(train_dataloader))[str(signal_name)].to(device).float()
    writer.add_graph(untrained_model, sample_input)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(untrained_model.parameters(), lr=learning_rate_min, weight_decay=weight_decay)

    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs) #!!!

    hyperparameters = {
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': torch.tensor(shots_for_testing.values.tolist()),
    'shots_for_validation': torch.tensor(shots_for_validation.values.tolist()),
    'shots_for_training': torch.tensor(shots_for_training.values.tolist()),
    'signal_name': signal_name,
    'num_classes': num_classes,
    'sampling_frequency':sampling_freq,
    'random_seed': random_seed,
    'architecture': architecture,
    'signal_window': signal_window,
    'dpoints_in_future': dpoints_in_future,
    'exponential_elm_decay': exponential_elm_decay,
    'only_ELMS': 'True' if only_ELMs else 'False',
    }

    # Train model
    model = cmc.train_model(untrained_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs, 
                        chkpt_path = model_path.with_name(f'{model_path.stem}_best_val_acc{model_path.suffix}'),
                        signal_name=signal_name)

    torch.save(model.state_dict(), model_path)

    # Test model
    metrics = cmc.test_model(f'{path}/runs/{timestamp}', model, 
                             test_dataloader, 
                             comment ='', 
                             signal_name=signal_name, 
                             writer=writer, 
                             num_classes=num_classes)
    
    metrics['prediction_df'].to_csv(f'{path}/runs/{timestamp}/prediction_df.csv')
    metrics_per_shot = cmc.per_shot_test(f'{path}/runs/{timestamp}', 
                      shots_for_testing, 
                      metrics['prediction_df'], 
                      writer=writer, 
                      num_classes=num_classes)
    
    metrics_per_shot = pd.DataFrame(metrics_per_shot)
    metrics_per_shot.to_csv(f'{path}/runs/{timestamp}/metrics_per_shot.csv')

    one_digit_metrics = {'Accuracy on test_dataset': metrics['accuracy'], 
                        'F1 metric on test_dataset':metrics['f1'].tolist(), 
                        'Precision on test_dataset':metrics['precision'].tolist(), 
                        'Recall on test_dataset':metrics['recall'].tolist()}
    if num_classes == 3:
        one_digit_metrics['PR AUC L-mode on test_dataset'] = metrics['pr_roc_curves']['pr_auc'][0].tolist()
        one_digit_metrics['PR AUC H-mode on test_dataset'] = metrics['pr_roc_curves']['pr_auc'][1].tolist()
        one_digit_metrics['PR AUC ELM on test_dataset'] = metrics['pr_roc_curves']['pr_auc'][2].tolist()
    

    writer.add_hparams(hyperparameters, one_digit_metrics)
    writer.close()
    
    # Save hyperparameters and metrics to a JSON file
    for key in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
        hyperparameters[key] = hyperparameters[key].tolist()  # Convert tensors to lists
    all_hparams = {**hyperparameters, **one_digit_metrics}
    # Convert to JSON
    json_str = json.dumps(all_hparams, indent=4)
    with open(f'{path}/runs/{timestamp}/hparams.json', 'w') as f:
        f.write(json_str)

if __name__ == "__main__":
    train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            batch_size = 512,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = '3 classes')
    

    print('Done')