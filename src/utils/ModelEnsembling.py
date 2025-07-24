import os
import re
import time 
from pathlib import Path
import copy
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

from . import confinement_mode_classifier as cmc

def train_and_test_ensembled_model(ris_option = 'RIS1',
                                    second_img_opt = 'RIS2',
                                    num_workers = 32,
                                    num_epochs_for_fc = 10,
                                    num_epochs_for_all_layers = 10,
                                    batch_size = 16,
                                    learning_rate_min = 0.001,
                                    learning_rate_max = 0.01,
                                    comment_for_model_name = ' no comment.',
                                    one_ris_models_paths = None,
                                    random_seed = 42):
    """
    Trains a model that ensembles two RIS models. The first model is trained on RIS1 images and the second model is trained on RIS2 images.
    """

    if one_ris_models_paths is None:
        raise ValueError('one_ris_model_paths must be a list of two strings. Each string must contain the path to the model trained on RIS1 and RIS2 images, respectively.')   
    pl.seed_everything(random_seed)

    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    comment_for_model_name = ris_option + 'x' + second_img_opt  + comment_for_model_name


    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_ris1'] & shot_usage['used_for_ris2']]
    shot_numbers = shot_for_ris['shot']
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']


    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path, shot_numbers, 
                                                                    shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=False)

    #Get dataloaders. second_img_opt='RIS1' indicates that two RIS1 models will be ensembled
    test_dataloader = cmc.get_dloader(test_df, path=path, batch_size=batch_size,
                                    shuffle=False, balance_data=False, 
                                    ris_option=ris_option, second_img_opt=second_img_opt, 
                                    num_workers=num_workers)

    val_dataloader = cmc.get_dloader(val_df, path=path, batch_size=batch_size,
                                    shuffle=False, balance_data=True, 
                                    ris_option=ris_option, second_img_opt=second_img_opt, 
                                    num_workers=num_workers)

    train_dataloader = cmc.get_dloader(train_df, path=path, batch_size=batch_size,
                                    shuffle=False, balance_data=True, 
                                    ris_option=ris_option, second_img_opt=second_img_opt, 
                                    num_workers=num_workers)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}


    pretrained_model = torchvision.models.resnet18(weights='IMAGENET1K_V1', )
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 2) #3 classes: L-mode, H-mode, ELM
    pretrained_model = pretrained_model.to(device)

    #Load pretrained model. RIS1 in this case
    pretrained_model.load_state_dict(torch.load(one_ris_models_paths['RIS1']))


    #Load pretrained RIS2 model
    ris2_model = copy.deepcopy(pretrained_model)
    ris2_model.load_state_dict(torch.load(one_ris_models_paths['RIS2']))


    untrained_ensembled_model = cmc.TwoImagesModel(modelA=pretrained_model, modelB=ris2_model, hidden_units=30).to(device)



    for name, param in untrained_ensembled_model.named_parameters():
        # Check if the current parameter is part of the MLP
        if 'classifier' in name or 'fc' in name or 'last_fully_connected' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Verify that only the MLP parameters have requires_grad set to True
    for name, param in untrained_ensembled_model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")


    timestamp =  datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name
    writer = SummaryWriter(f'runs/{timestamp}_classifier_training')

    sample_input = next(iter(train_dataloader))['img'].to(device).float()
    writer.add_graph(untrained_ensembled_model, sample_input)

    #
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(untrained_ensembled_model.parameters(), lr=learning_rate_min) #pouzit adam

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=50) #!!!

    model_path = Path(f'{path}/runs/{timestamp}_classifier_training/model.pt')

    ensembled_model = cmc.train_model(untrained_ensembled_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_fc, 
                        chkpt_path = model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))

    torch.save(ensembled_model.state_dict(), model_path)

    metrics = cmc.test_model(f'runs/{timestamp}_classifier_training/', ensembled_model, test_dataloader, comment='', writer=writer, max_batch=0)
    shots_for_testing = shots_for_testing.values.tolist()
    img_path = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_classifier_training/', 
                                shots=shots_for_testing, results_df=metrics['prediction_df'], writer=writer)
    writer.add_scalar(f'Accuracy on test_dataset', metrics['accuracy'])
    writer.add_scalar(f'F1 metric on test_dataset', metrics['f1'])
    writer.add_scalar(f'Precision on test_dataset', metrics['precision'])
    writer.add_scalar(f'Recall on test_dataset', metrics['recall'])
    writer.close()


    # Clear cash
    if cuda.is_available():
        # Do i have a single GPU?
        cuda.empty_cache()
        
        # Do i have multiple GPUs?
        for i in range(cuda.device_count()):
            cuda.reset_max_memory_allocated(i)
            cuda.empty_cache()

    writer = SummaryWriter(f'runs/{timestamp}_all_layers')

    #Unfreeze all layers
    for name, param in ensembled_model.named_parameters():
        param.requires_grad = True

    #Check that all parameters are being optimized
    for name, param in ensembled_model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")

    model_path = Path(f'{path}/runs/{timestamp}_all_layers/model.pt')

    ensembled_model = cmc.train_model(ensembled_model, criterion, optimizer, exp_lr_scheduler, 
                                    dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_all_layers,
                                    chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))


    torch.save(ensembled_model.state_dict(), model_path)

    metrics = cmc.test_model(f'{path}/runs/{timestamp}_all_layers/', ensembled_model, test_dataloader, max_batch=0, writer=writer)
    img_path = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_all_layers/', 
                                shots=shots_for_testing, results_df=metrics['prediction_df'], writer=writer)
    writer.add_scalar(f'Accuracy on test_dataset', metrics['accuracy'])
    writer.add_scalar(f'F1 metric on test_dataset', metrics['f1'])
    writer.add_scalar(f'Precision on test_dataset', metrics['precision'])
    writer.add_scalar(f'Recall on test_dataset', metrics['recall'])
    writer.close()

if __name__ == "__main__":
    train_and_test_ensembled_model(one_ris_models_paths = {'RIS1':'/compass/Shared/Users/bogdanov/ml_tokamak/runs/24-02-25, 09-47-13 RIS1, 2 output classes_all_layers/model.pt', 
                                                            'RIS2':'/compass/Shared/Users/bogdanov/ml_tokamak/runs/24-02-25, 11-15-07 RIS2, 2 output classes_all_layers/model.pt'})
    print('Training finished.')