import os
import re
import time 
from pathlib import Path
import json
from datetime import datetime

import matplotlib.pyplot as plt
from torch import cuda
import torchvision
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights, ResNet18_Weights

import confinement_mode_classifier as cmc

def train_and_test_ris_model(ris_option = 'both',
                            pretrained_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
                            num_workers = 32,
                            num_epochs_for_fc = 10,
                            num_epochs_for_all_layers = 10,
                            num_classes = 3,
                            batch_size = 32,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f', 3 output classes',
                            random_seed = 42,
                            augmentation = False,
                            test_df_contains_val_df=True,
                            test_run = False,
                            exponential_elm_decay=True,
                            grayscale=False,
                            weight_decay=1e-4,
                            data_frac=1.0):
    
    """
    Trains a one ris model. The model is trained on RIS1 images or RIS2 images.
    """

    comment_for_model_name = ris_option + comment_for_model_name
    pl.seed_everything(random_seed)

    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    #### Create dataloaders ########################################
    shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_ris2'] if ris_option == 'RIS2' else shot_usage['used_for_ris1']]
    shot_numbers = shot_for_ris['shot']
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']
    shots_for_training = shot_for_ris[shot_for_ris['used_as'] == 'train']['shot']

    if test_df_contains_val_df:
        shots_for_testing = pd.concat([shots_for_testing, shots_for_validation])

    if test_run:
        shots_for_testing = shots_for_testing[:3]
        shots_for_validation = shots_for_validation[:3]
        shots_for_training = shots_for_training[:3]

    # Use only a fraction of the data
    shots_for_training = shots_for_training.sample(frac=data_frac, random_state=random_seed)

    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_training, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=num_classes==3, ris_option=ris_option,
                                                                    exponential_elm_decay=exponential_elm_decay)
    
    if ris_option == 'both':
        shot_for_ris2 = shot_usage[shot_usage['used_for_ris2']]
        shot_numbers_ris2 = shot_for_ris2['shot']
        shots_for_testing_ris2 = shot_for_ris2[shot_for_ris2['used_as'] == 'test']['shot']
        shots_for_validation_ris2 = shot_for_ris2[shot_for_ris2['used_as'] == 'val']['shot']
        shots_for_training_ris2 = shot_for_ris2[shot_for_ris2['used_as'] == 'train']['shot']

        if test_df_contains_val_df:
            shots_for_testing_ris2 = pd.concat([shots_for_testing_ris2, shots_for_validation_ris2])

        if test_run:
            shots_for_testing_ris2 = shots_for_testing_ris2[:3]
            shots_for_validation_ris2 = shots_for_validation_ris2[:3]
            shots_for_training_ris2 = shots_for_training_ris2[:3]

        shots_for_training_ris2 = shots_for_training_ris2.sample(frac=data_frac, random_state=random_seed)

        shot_df_ris2, test_df_ris2, val_df_ris2, train_df_ris2 = cmc.load_and_split_dataframes(path,shot_numbers_ris2, shots_for_training_ris2, shots_for_testing_ris2, 
                                                                        shots_for_validation_ris2, use_ELMS=num_classes==3, ris_option='RIS2',
                                                                        exponential_elm_decay=exponential_elm_decay)

        test_df = pd.concat([test_df, test_df_ris2]).reset_index(drop=True)
        val_df = pd.concat([val_df, val_df_ris2]).reset_index(drop=True)
        train_df = pd.concat([train_df, train_df_ris2]).reset_index(drop=True)

        shots_for_testing = pd.concat([shots_for_testing, shots_for_testing_ris2]).reset_index(drop=True)
        shots_for_validation = pd.concat([shots_for_validation, shots_for_validation_ris2]).reset_index(drop=True)
        shots_for_training = pd.concat([shots_for_training, shots_for_training_ris2]).reset_index(drop=True)

    test_dataloader = cmc.get_dloader(test_df, path, batch_size, balance_data=False, 
                                      shuffle=False, num_workers=num_workers, 
                                      augmentation=False, grayscale=grayscale)

    val_dataloader = cmc.get_dloader(val_df, path, batch_size, balance_data=True, 
                                     shuffle=False, num_workers=num_workers, 
                                     augmentation=False, grayscale=grayscale)

    train_dataloader = cmc.get_dloader(train_df, path, batch_size, balance_data=True, 
                                       shuffle=False, num_workers=num_workers, 
                                       augmentation=augmentation, grayscale=grayscale)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    #### Create model and train it #################################
    timestamp =  datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name
    writer = SummaryWriter(f'runs/{timestamp}_last_fc')

    # Load a pretrained model and reset final fully connected layer.
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes) #3 classes: L-mode, H-mode, ELM
    pretrained_model = pretrained_model.to(device)

    if grayscale:
        # Luminance weights for RGB to grayscale conversion
        weights_rgb_to_gray = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(device)

        # Get the original weights of the first conv layer
        original_weights = pretrained_model.conv1.weight.data

        # Compute the weighted sum of the RGB channels
        grayscale_weights = (original_weights * weights_rgb_to_gray).sum(dim=1, keepdim=True)

        # Update the first convolutional layer
        pretrained_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=pretrained_model.conv1.out_channels,
            kernel_size=pretrained_model.conv1.kernel_size,
            stride=pretrained_model.conv1.stride,
            padding=pretrained_model.conv1.padding,
            bias=pretrained_model.conv1.bias is not None)

        # Assign the new grayscale weights to the first conv layer
        pretrained_model.conv1.weight = nn.Parameter(grayscale_weights)

        # If there is a bias term, keep it unchanged
        if pretrained_model.conv1.bias is not None:
            pretrained_model.conv1.bias = nn.Parameter(pretrained_model.conv1.bias.data)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=learning_rate_min, weight_decay=weight_decay)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs_for_fc) #!!!

    # Model will be saved to this folder along with metrics and tensorboard scalars
    model_path = Path(f'{path}/runs/{timestamp}_last_fc/model.pt')

    #### Train the last fully connected layer########################
    model = cmc.train_model(pretrained_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_fc, 
                        chkpt_path=model_path.with_name(f'{model_path.stem}_best_val_acc{model_path.suffix}'),
                        return_best_model=False)


    hyperparameters = {
    'model': model.__class__.__name__,
    'batch_size': batch_size,
    'num_epochs': num_epochs_for_fc,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': torch.tensor(shots_for_testing.values.tolist()),
    'shots_for_validation': torch.tensor(shots_for_validation.values.tolist()),
    'shots_for_training': torch.tensor(shots_for_training.values.tolist()),
    'ris_option': ris_option,
    'num_classes': num_classes,
    'second_image': 'None',
    'augmentation': "applied" if augmentation else "no augmentation",
    'random_seed': random_seed,
    'weight_decay':weight_decay
    }
    
    
    torch.save(model.state_dict(), model_path)

    #### Test the model############################################
    metrics = cmc.test_model(f'runs/{timestamp}_last_fc', model, test_dataloader, comment='', 
                             writer=writer, num_classes=num_classes, signal_name='img')

    metrics['prediction_df'].to_csv(f'{path}/runs/{timestamp}_last_fc/prediction_df.csv')

    metrics_per_shot = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_last_fc/', 
                                shots=shots_for_testing.values.tolist(), 
                                results_df=metrics['prediction_df'], 
                                writer=writer,
                                num_classes=num_classes,
                                two_images=ris_option=='both')

    pd.DataFrame(metrics_per_shot).to_csv(f'{path}/runs/{timestamp}_last_fc/metrics_per_shot.csv')

    one_digit_metrics = {'Accuracy on test_dataset': metrics['accuracy'], 
                        'F1 metric on test_dataset':metrics['f1'].tolist(), 
                        'Precision on test_dataset':metrics['precision'].tolist(), 
                        'Recall on test_dataset':metrics['recall'].tolist()}

    writer.add_hparams(hyperparameters, one_digit_metrics)
    writer.close()
    
    # Save hyperparameters and metrics to a JSON file
    for key in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
        hyperparameters[key] = hyperparameters[key].tolist()  # Convert tensors to lists
    all_hparams = {**hyperparameters, **one_digit_metrics}
    # Convert to JSON
    json_str = json.dumps(all_hparams, indent=4)
    with open(f'{path}/runs/{timestamp}_last_fc/hparams.json', 'w') as f:
        f.write(json_str)

    #### Train the whole model######################################
    torch.cuda.empty_cache()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=learning_rate_min, weight_decay=1e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs_for_all_layers) #!!!

    writer = SummaryWriter(f'runs/{timestamp}_all_layers')
    for param in model.parameters():
        param.requires_grad = True

    model_path = Path(f'{path}/runs/{timestamp}_all_layers/model.pt')

    model = cmc.train_model(model, criterion, optimizer, exp_lr_scheduler, 
                            dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_all_layers,
                            chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))
    
    torch.save(model.state_dict(), model_path)

    hyperparameters = {
    'batch_size': batch_size,
    'num_epochs': num_epochs_for_all_layers,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': torch.tensor(shots_for_testing.values.tolist()),
    'shots_for_validation': torch.tensor(shots_for_validation.values.tolist()),
    'shots_for_training': torch.tensor(shots_for_training.values.tolist()),
    'ris_option': ris_option,
    'num_classes': num_classes,
    'second_image': 'None',
    'augmentation': "applied" if augmentation else "no augmentation",
    'random_seed': random_seed
    }

    #### Test the model############################################
    metrics = cmc.test_model(f'runs/{timestamp}_all_layers', model, test_dataloader,
                              comment='', writer=writer, signal_name='img', num_classes=num_classes)
    
    metrics['prediction_df'].to_csv(f'{path}/runs/{timestamp}_all_layers/prediction_df.csv')

    metrics_per_shot = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_all_layers/', 
                                shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
                                writer=writer, num_classes=num_classes,
                                two_images=ris_option=='both')
    
    metrics_per_shot = pd.DataFrame(metrics_per_shot)
    metrics_per_shot.to_csv(f'{path}/runs/{timestamp}_all_layers/metrics_per_shot.csv')

    one_digit_metrics = {'Accuracy on test_dataset': metrics['accuracy'], 
                        'F1 metric on test_dataset':metrics['f1'].tolist(), 
                        'Precision on test_dataset':metrics['precision'].tolist(), 
                        'Recall on test_dataset':metrics['recall'].tolist()}

    writer.add_hparams(hyperparameters, one_digit_metrics)
    writer.close()
    
    # Save hyperparameters and metrics to a JSON file
    for key in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
        hyperparameters[key] = hyperparameters[key].tolist()  # Convert tensors to lists
    all_hparams = {**hyperparameters, **one_digit_metrics}
    # Convert to JSON
    json_str = json.dumps(all_hparams, indent=4)
    with open(f'{path}/runs/{timestamp}_all_layers/hparams.json', 'w') as f:
        f.write(json_str)

    return model, model_path

if __name__ == '__main__':
    train_and_test_ris_model()
    print('Done')