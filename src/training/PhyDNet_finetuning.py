from skimage.metrics import structural_similarity as ssim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryPrecision, BinaryRecall, F1Score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from PIL import Image
import random
import json
import copy

import time
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall
from ..models.PhyDNet_models import ConvLSTM, PhyCell, ClassifierRNN
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
from PhyDNet.constrain_moments import K2M
import torch.multiprocessing as mp
import pytorch_lightning as pl

import argparse
from tqdm import tqdm
import os
import re
import pandas as pd
from torchvision.io import read_image
from datetime import datetime
from pathlib import Path
from ..utils import confinement_mode_classifier as cmc
from torch.utils.tensorboard import SummaryWriter


from . import PhyDNet_COMPASS as pdnt



def finetune_phydnet(path_to_model, test_run=False, test_df_contains_val_df=True, batch_size=10, 
                          num_workers=6, n_frames_input=4, save_name='PhyDNet, changed loss, reduceLRonPlateau', 
                          learning_rate_max=1e-3, weight_decay=1e-5, num_epochs_cnn=5, num_epochs_all_layers=10, ris_option='RIS1'):
    
    pl.seed_everything(42)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S")
    save_name = timestamp + save_name
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


    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_training, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=True, ris_option=ris_option,
                                                                    exponential_elm_decay=False)

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


        shot_df_ris2, test_df_ris2, val_df_ris2, train_df_ris2 = cmc.load_and_split_dataframes(path,shot_numbers_ris2, shots_for_training_ris2, shots_for_testing_ris2, 
                                                                        shots_for_validation_ris2, use_ELMS=True, ris_option='RIS2',
                                                                        exponential_elm_decay=False)

        test_df = pd.concat([test_df, test_df_ris2]).reset_index(drop=True)
        val_df = pd.concat([val_df, val_df_ris2]).reset_index(drop=True)
        train_df = pd.concat([train_df, train_df_ris2]).reset_index(drop=True)

        shots_for_testing = pd.concat([shots_for_testing, shots_for_testing_ris2]).reset_index(drop=True)
        shots_for_validation = pd.concat([shots_for_validation, shots_for_validation_ris2]).reset_index(drop=True)
        shots_for_training = pd.concat([shots_for_training, shots_for_training_ris2]).reset_index(drop=True)
    
    #Read article, see PhyDNet/constrain_moments.py
    constraints = torch.zeros((49,7,7)).to(device)
    ind = 0
    for i in range(0,7):
        for j in range(0,7):
            constraints[ind,i,j] = 1
            ind +=1   



    train_loader = pdnt.get_loader(train_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=True)
    val_loader = pdnt.get_loader(val_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=True)
    test_loader = pdnt.get_loader(test_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=False)

    dataloaders = {'train':train_loader, 'val':val_loader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    phycell  =  PhyCell(input_shape=(88,88), input_dim=352, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
    convcell =  ConvLSTM(input_shape=(88,88), input_dim=352, hidden_dims=[8,352], n_layers=2, kernel_size=(3,3), device=device)   
    classifier = ClassifierRNN(phycell, convcell, device)

    classifier.load_state_dict(torch.load(f'{path}/{path_to_model}'))
    classifier = classifier.to(device)

    #writer = SummaryWriter(f'PhyDNet/runs/{save_name}_last_conv')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate_max, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs_cnn) # lr_scheduler.ReduceLROnPlateau(optimizer, patience=1) #!!!
    
    

    hyperparameters = {
    'original_model_path': path_to_model,
    'model': classifier.__class__.__name__,
    'exponential_elm_decay':False,
    'n_frames_input': n_frames_input,
    'batch_size': batch_size,
    'num_epochs_cnn': num_epochs_cnn,
    'num_epochs_all_layers':num_epochs_all_layers,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': shots_for_testing.values.tolist(),
    'shots_for_validation': shots_for_validation.values.tolist(),
    'shots_for_training': shots_for_training.values.tolist(),
    'ris_option':'RIS1',
    'num_classes': 3,
    'signal_name': 'imgs_input'
    
    }

    # json_str = json.dumps(hyperparameters, indent=4)
    # with open(f'PhyDNet/runs/{save_name}_last_conv/hparams.json', 'w') as f:
    #     f.write(json_str)


    #######Train only last conv classifier######################

    classifier_CNN = copy.deepcopy(classifier)
    #model_last_cnn_path = Path(f'PhyDNet/runs/{save_name}_last_conv/model.pt')

    #Freeze all the layers except last conv classifier
    # for name, child in classifier.named_children():
    #     if name == 'classifier':
    #         # Skip freezing the classifier module
    #         for param in child.parameters():
    #             param.requires_grad = True  # Just to be sure, explicitly setting it to True
    #     else:
    #         # Freeze parameters in other modules
    #         for param in child.parameters():
    #             param.requires_grad = False

    
    # classifier_CNN = pdnt.train_model(classifier, criterion, optimizer, exp_lr_scheduler, dataloaders, 
    #                                   writer, dataset_sizes, num_epochs_all_layers, 
    #                                   chkpt_path=model_last_cnn_path.with_name(f'{model_last_cnn_path.stem}_chkpt{model_last_cnn_path.suffix}'), 
    #                                   signal_name='imgs_input', device=device, 
    #                                   constraints=constraints, return_best_model=False)
    

    #torch.save(classifier_CNN.state_dict(), model_last_cnn_path)

    # metrics = pdnt.test_model(f'PhyDNet/runs/{save_name}_last_conv', classifier_CNN, test_loader,
    #                           comment='', writer=writer, signal_name='imgs_input', num_classes=3, 
    #                           constraints=constraints, criterion=criterion)
    
    # metrics['prediction_df'].to_csv(f'PhyDNet/runs/{save_name}_last_conv/prediction_df.csv')

    # metrics_per_shot = cmc.per_shot_test(path=f'PhyDNet/runs/{save_name}_last_conv', 
    #                         shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
    #                         writer=writer, num_classes=3,
    #                         two_images=False)
    
    # metrics_per_shot = pd.DataFrame(metrics_per_shot)
    # metrics_per_shot.to_csv(f'PhyDNet/runs/{save_name}_last_conv/metrics_per_shot.csv')

    ################## Train the whole model ############################
    model_path = Path(f'PhyDNet/runs/{save_name}_all_layers/model.pt')

    writer = SummaryWriter(f'PhyDNet/runs/{save_name}_all_layers')

    for param in classifier_CNN.parameters():
        param.requires_grad = True


    json_str = json.dumps(hyperparameters, indent=4)
    with open(f'PhyDNet/runs/{save_name}_all_layers/hparams.json', 'w') as f:
        f.write(json_str)

    optimizer = torch.optim.AdamW(classifier_CNN.parameters(), lr=learning_rate_max, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs_all_layers)

    classifier_all_layers = pdnt.train_model(classifier_CNN, criterion, optimizer, 
                                             exp_lr_scheduler, dataloaders,writer, dataset_sizes, 
                                             num_epochs_all_layers, 
                                             chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'), 
                                             signal_name='imgs_input', device=device, 
                                             constraints=constraints, return_best_model=False)
    

    torch.save(classifier.state_dict(), model_path)

    metrics = pdnt.test_model(f'PhyDNet/runs/{save_name}_all_layers', classifier_all_layers, test_loader,
                              comment='', writer=writer, signal_name='imgs_input', num_classes=3, 
                              constraints=constraints, criterion=criterion)
    
    metrics['prediction_df'].to_csv(f'PhyDNet/runs/{save_name}_all_layers/prediction_df.csv')

    metrics_per_shot = cmc.per_shot_test(path=f'PhyDNet/runs/{save_name}_all_layers', 
                            shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
                            writer=writer, num_classes=3,
                            two_images=False)
    
    metrics_per_shot = pd.DataFrame(metrics_per_shot)
    metrics_per_shot.to_csv(f'PhyDNet/runs/{save_name}_all_layers/metrics_per_shot.csv')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    finetune_phydnet(path_to_model='PhyDNet/runs/24-05-15, 12-53-51 PhyDNet_finetuning_last_conv/model_chkpt_best_acc.pt',
                     test_run=False, test_df_contains_val_df=True,
                     num_epochs_cnn=10, num_epochs_all_layers=10)
    print('Done')