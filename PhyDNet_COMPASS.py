from skimage.metrics import structural_similarity as ssim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.classification import BinaryPrecision, BinaryRecall, F1Score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import xarray as xr
from PIL import Image
import json
import time
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall
from PhyDNet_models import ConvLSTM, PhyCell, ClassifierRNN
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
import confinement_mode_classifier as cmc
from torch.utils.tensorboard import SummaryWriter


class ImagesDataset(Dataset):
    '''
    takes annotations, img_dir and index, returns sequence of 10 images. label and path correspond to the last image in the sequence
    '''
    def __init__(self, annotations, img_dir, n_frames_input=10,
                 device = torch.device("cuda:0"), gray_scale = False):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.n_frames_input = n_frames_input
        self.device = device
        self.gray_scale = gray_scale
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        prev_imgs = torch.tensor([]).to(self.device) #images from the past - input for the model
        for i in range(self.n_frames_input):

            if idx-2*i<0 or idx-2*i>len(self.img_labels)-1:
                img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
            else:
                img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-2*i, 'filename']) #shift by 50 to avoid KeyError. They're black anyway.
            
            image = read_image(img_path).float().to(self.device)
            image = image[:,74:-74,144:-144] # crop the image 640x500 -> 352x352
            if self.gray_scale:
                image = image.mean(dim=0, keepdim=True)

            normalized_image = (image/255)
            prev_imgs = torch.cat((prev_imgs, normalized_image.unsqueeze(0)), dim=0)

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        path = self.img_labels.iloc[idx]['filename']
  
        labeled_imgs_dict = {'imgs_input': prev_imgs,
                             'label':label, 'path':path, 'time':time}


        return labeled_imgs_dict
    
def train_on_batch(input_tensor, labels_tensor, classifier, criterion, 
                   device = torch.device("cuda:0"), constraints = None, save_name = None, save_weights = False, step=0):                
    
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    loss = 0
    for ei in range(input_length): 
        outputs = classifier(input_tensor[:,ei,:,:,:], (ei==0))
        loss += 2**((ei-input_length)/input_length)*criterion(outputs, labels_tensor.long() if len(labels_tensor.size())==1 else labels_tensor)

    #Errors should be more or less equal
    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    constraints_loss = 0
    k2m = K2M([7,7]).to(device)
    for b in range(0,classifier.phycell.cell_list[0].input_dim):
        filters = classifier.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()  

        if constraints is not None and save_weights:
            filters_data = filters.detach().cpu().numpy()
            m_data = m.detach().cpu().numpy()
            constraints_data = constraints.detach().cpu().numpy()

            # Create a DataFrame with the filter, m, k2m, and constraints data
            data = {'filters': (['iter', 'in_channels', 'rows', 'cols'], np.expand_dims(filters_data, axis=0)),
                    'm': (['iter', 'in_channels', 'rows', 'cols'], np.expand_dims(m_data, axis=0)),
                    'constraints': (['iter', 'in_channels', 'rows', 'cols'], np.expand_dims(constraints_data, axis=0)),
                    'b': (['iter', 'value'], np.expand_dims(np.array([b]), axis=0)),
                    'step': (['iter', 'value'], np.expand_dims(np.array([step]), axis=0))}

            if step < 9:
                data_df = xr.Dataset(data_vars=data)
            else:
                old_df = xr.open_dataset(f'{save_name}_filter_data.nc')
                
                data_df = xr.concat([old_df, xr.Dataset(data_vars=data)], dim='iter')

            save_path = f'{save_name}_filter_data.nc'
            save_dir = os.path.dirname(save_path)

            # Ensure the directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


            # Save the dataset to a NetCDF file
            data_df.to_netcdf(save_path, mode='w')


        constraints_loss += nn.MSELoss()(m, constraints) # constrains is a precomputed matrix 

    loss += constraints_loss

    return outputs, loss, constraints_loss


def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict,
                 writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25,
                 chkpt_path=os.getcwd(), signal_name='img', device = torch.device("cuda:0"),
                 return_best_model = False, constraints = None):

    since = time.time()

    best_acc = 0.0

    total_loss = {'train': 0.0, 'val': 0.0}
    total_constraints_loss = {'train': 0.0, 'val': 0.0}
    total_batch = {'train': 0, 'val': 0}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                inputs = batch[signal_name].to(device).float()
                labels = batch['label'].to(device)

                if len(labels.size()) > 1: #If soft labels are used
                    _, ground_truth = torch.max(labels, axis=1)
                else: 
                    ground_truth = labels

                running_batch += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, loss, constraints_loss = train_on_batch(inputs, labels, model, criterion, constraints=constraints, 
                                                                     device=device, save_weights=running_batch%4==0, save_name=chkpt_path.with_name(f'{chkpt_path.stem}'),
                                                                     step=running_batch)

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        total_batch['train'] += 1
                        if total_batch['train'] == 1:
                            total_loss['train'] = loss.item()
                            total_constraints_loss['train'] = constraints_loss.item()
                        else:
                            total_loss['train'] = (0.85*(total_loss['train']) + 0.15*loss.item())/(1-0.85**total_batch['train'])
                            total_constraints_loss['train'] = (0.85*(total_constraints_loss['train']) + 0.15*constraints_loss.item())/(1-0.85**total_batch['train'])
                    else:
                        total_batch['val'] += 1
                        if total_batch['val'] == 1:
                            total_loss['val'] = loss.item()
                            total_constraints_loss['val'] = constraints_loss.item()
                        else:
                            total_loss['val'] = (0.85*(total_loss['val']) + 0.15*loss.item())/(1-0.85**total_batch['val'])        
                            total_constraints_loss['val'] = (0.85*(total_constraints_loss['val']) + 0.15*constraints_loss.item())/(1-0.85**total_batch['val'])              
                    
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == ground_truth.data) #How many correct answers
                
                
                #tensorboard part
                
                if running_batch % int(len(dataloaders[phase])/400)==int(len(dataloaders[phase])/400)-1: 
                    # ...log the running loss
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing loss', total_loss[phase],
                                    total_batch[phase])
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing constr loss', total_constraints_loss[phase],
                                    total_batch[phase])
                    
                    #F1 metric
                    writer.add_scalar(f'{phase}ing F1 metric',
                                    F1Score(task="multiclass", num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                    epoch * len(dataloaders[phase]) + running_batch)
                    
                    #Precision recall
                    writer.add_scalar(f'{phase}ing macro Precision', 
                                        MulticlassPrecision(num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                    writer.add_scalar(f'{phase}ing macro Recall', 
                                        MulticlassRecall(num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                if phase == 'train':
                    scheduler.step() #scheduler.step(total_loss['val']) 

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar(f'accuracy', epoch_acc, epoch)
                writer.close()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), chkpt_path.with_name(f'{chkpt_path.stem}_best_acc{chkpt_path.suffix}'))

            torch.save(model.state_dict(), chkpt_path)
            
        time_elapsed = time.time() - since

        # load best model weights
    if return_best_model:
        model.load_state_dict(torch.load(chkpt_path.with_name(f'{chkpt_path.stem}_best_acc{chkpt_path.suffix}')))
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model



def images_to_probs(imgs_tensor, labels_tensor, model, criterion, constraints, device):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    ''' 
    with torch.no_grad():
        output, loss, constraints_loss = train_on_batch(imgs_tensor, labels_tensor, model, criterion, constraints=constraints, device=device)
    # convert output probabilities to predicted class
    max_logit, class_prediction = torch.max(output, 1) 
    preds = np.squeeze(class_prediction.cpu().numpy())
    return output, preds, loss, constraints_loss


def test_model(run_path,
               model, test_dataloader: DataLoader,
               max_batch: int = 0, return_metrics: bool = True, num_classes: int = 3,
               comment: str ='', signal_name: str = 'divlp', writer: SummaryWriter = None,
               device = torch.device("cuda:0"), constraints=None, criterion = nn.CrossEntropyLoss()):
    '''
    Takes model and dataloader and returns figure with confusion matrix, 
    dataframe with predictions, F1 metric value, precision, recall and accuracy

    Args:
        model: ResNet model
        test_dataloader: DataLoader used for testing
        max_batch: maximum number of bathces to use for testing. Set = 0 to use all batches in DataLoader
        return_metrics: if True returns confusion matrix, F1, precision, recall and accuracy 
    
    Returns: 
        preds: pd.DataFrame() pd.DataFrame with columns of predicted class, true class, frame time and confidence of the prediction
        precision: MulticlassPrecision(num_classes=3)
        recall: MulticlassRecall(num_classes=3)
        accuracy: (TP+TN)/(TP+TN+FN+FP)
        fig_confusion_matrix: MulticlassConfusionMatrix(num_classes=3)
    '''
    y_df = torch.tensor([])
    y_hat_df = torch.tensor([])
    softmax_out_df = torch.tensor([])
    preds = pd.DataFrame([])
    pattern = re.compile(r'RIS[12]_(\d+)_t=')        
    batch_index = 0 #iterator
    avg_loss = 0
    avg_constr_loss = 0
    for batch in tqdm(test_dataloader, desc='Processing batches'):
        batch_index +=1
        softmax_out, y_hat, loss, constr_loss = images_to_probs(batch[signal_name].to(device).float(), batch['label'].to(device), model, criterion, constraints, device)
        avg_loss += loss
        avg_constr_loss += constr_loss
        softmax_out = softmax_out.cpu()
        y_hat = torch.tensor(y_hat)
        #Here I use if statement, because sometimes the model returns 2D tensor with soft labels
        y_df = torch.cat((y_df.int(), batch['label'] if len(batch['label'].size())==1 else batch['label'].max(axis=1)[1]), dim=0)
        try:
            y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
        except:
            y_hat_df = torch.cat((y_hat_df, torch.tensor([y_hat])), dim=0)

        softmax_out_df = torch.cat((softmax_out_df, softmax_out), dim=0)

        #That's simply mean  if shot is not explicitly given, then that's a RIS model, and it has shot number in the filename
        if 'shot' not in batch.keys():
            shot_numbers = [int(pattern.search(path).group(1)) for path in batch['path']]
        else:
            shot_numbers = batch['shot']

        pred = pd.DataFrame({'shot': shot_numbers, 'prediction': y_hat.data, 
                            'label': batch['label'] if len(batch['label'].size())==1 else batch['label'].max(axis=1)[1], 
                            'time':batch['time'], 
                            'prob_0': softmax_out[:,0].cpu(), 
                            'prob_1': softmax_out[:,1].cpu()})

        if num_classes==3:
            pred['prob_2'] = softmax_out[:,2].cpu()

        preds = pd.concat([preds, pred], axis=0, ignore_index=True)  


        if max_batch!=0 and batch_index>max_batch:
            break

    avg_loss/=batch_index
    avg_constr_loss/=batch_index
    
    if return_metrics:
        print('Processing metrics...')
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes)
        confusion_matrix_metric.update(y_hat_df, y_df)
        conf_matrix_fig, conf_matrix_ax  = confusion_matrix_metric.plot()
        #F1
        f1 = f1_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'micro')

        #Precision and recall
        precision = precision_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'micro')
        recall = recall_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'micro')


        #Precision_recall and ROC curves are generated using the pr_roc_auc()
        pr_roc = cmc.pr_roc_auc(y_df, softmax_out_df, task="binary" if num_classes==2 else 'ternary')
        pr_fig = pr_roc['pr_curve'][0]
        roc_fig = pr_roc['roc_curve'][0]
        roc_ax = pr_roc['roc_curve'][1]

        #Accuracy
        accuracy = len(preds[preds['prediction']==preds['label']])/len(preds)

        textstr = '\n'.join((
            f'Whole test dset',
            r'f1=%.2f' % (f1.item(), ),
            r'precision=%.2f' % (precision.item(), ),
            r'recall=%.2f' % (recall.item(), ),
            r'accuracy=%.2f' % (accuracy, ),
            r'avg CE loss=%.2f' % (avg_loss, ),
            r'avg constr CE loss=%.2f' % (avg_constr_loss, )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        conf_matrix_ax.set_title(f'confusion matrix for whole test dset')
        roc_ax.text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)

        # Open the saved images using Pillow
        roc_img = cmc.matplotlib_figure_to_pil_image(roc_fig)
        pr_img = cmc.matplotlib_figure_to_pil_image(pr_fig)

        if num_classes==3: #otherwise the images are too wide (problem with 3 colorbars)
            roc_img = roc_img.crop([int(0.06*roc_img.width), 0, int(0.8*roc_img.width), roc_img.height])
            pr_img = pr_img.crop([int(0.06*pr_img.width), 0, int(0.8*pr_img.width), pr_img.height])
        conf_matrix_img = cmc.matplotlib_figure_to_pil_image(conf_matrix_fig)
        
        # Resize the images to have the same height
        new_conf_width = int(conf_matrix_img.width/conf_matrix_img.height * pr_img.height)
        conf_matrix_img = conf_matrix_img.resize((new_conf_width, pr_img.height))

        # Create a new image with a white background
        combined_image = Image.new('RGB', (conf_matrix_img.width + pr_img.width + roc_img.width,
                                            roc_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(conf_matrix_img, (0, 0))
        combined_image.paste(roc_img, (conf_matrix_img.width, 0))
        combined_image.paste(pr_img, (roc_img.width+conf_matrix_img.width, 0))
        
        # Save the combined image
        combined_image.save(f'{run_path}/metrics_for_whole_test_dset_{comment}.png')

        # Save the images to tensorboard
        if writer:
            combined_image_tensor = torchvision.transforms.ToTensor()(combined_image)
            writer.add_image('metrics_for_whole_test_dset', combined_image_tensor)

        return {'prediction_df': preds, 'confusion_matrix': (conf_matrix_fig, conf_matrix_ax), 'f1': f1,
                'precision': precision, 'recall': recall, 'accuracy': accuracy, 'pr_roc_curves': pr_roc, 
                'avg_loss':avg_loss, 'avg_constr_loss':avg_constr_loss}
                
    else:
        return {'prediction_df': preds}


def get_loader(df, batch_size=8, num_workers=16, n_frames_input=4, path=os.getcwd(), balance=True):
    dataset = ImagesDataset(df, path, gray_scale=True, n_frames_input=n_frames_input)
    if balance:
        mode_weight = (1/df['mode'].value_counts()).values
        sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
        sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                sampler=sampler, 
                                num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers)
    return dataloader


def train_and_eval_PhyDNet(batch_size=8, learning_rate_min=0.0001, learning_rate_max=0.01, num_epochs=16,
                           test_run=False, test_df_contains_val_df=False, n_frames_input=4, num_workers=3,
                           weight_decay=1e-2, ris_option='RIS1', save_name='PhyDNet'):
     # data range 0 to 1 - images normalized this way

    pl.seed_everything(42)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ")
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

    train_loader = get_loader(train_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=True)
    val_loader = get_loader(val_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=True)
    test_loader = get_loader(test_df, batch_size=batch_size, num_workers=num_workers, n_frames_input=n_frames_input, path=path, balance=False)

    dataloaders = {'train':train_loader, 'val':val_loader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    phycell  =  PhyCell(input_shape=(88,88), input_dim=352, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
    convcell =  ConvLSTM(input_shape=(88,88), input_dim=352, hidden_dims=[8,352], n_layers=2, kernel_size=(3,3), device=device)   
    classifier = ClassifierRNN(phycell, convcell, device)

    writer = SummaryWriter(f'PhyDNet/runs/{save_name}')
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate_max, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=dataset_sizes['train'], epochs=num_epochs) 
    
    model_path = Path(f'PhyDNet/runs/{save_name}/model.pt')

    hyperparameters = {
    'model': classifier.__class__.__name__,
    'exponential_elm_decay':False,
    'n_frames_input': n_frames_input,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
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

    json_str = json.dumps(hyperparameters, indent=4)
    with open(f'PhyDNet/runs/{save_name}/hparams.json', 'w') as f:
        f.write(json_str)

    trained_model = train_model(classifier, criterion, optimizer, exp_lr_scheduler, {'train':train_loader, 'val':val_loader}, writer, 
                                dataset_sizes, num_epochs=num_epochs, 
                                chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'), 
                                signal_name='imgs_input', device=device, constraints=constraints, return_best_model=True)
    
    torch.save(trained_model.state_dict(), model_path)

    metrics = test_model(f'PhyDNet/runs/{save_name}', trained_model, test_loader,
                              comment='', writer=writer, signal_name='imgs_input', num_classes=3, 
                              constraints=constraints, criterion=criterion)
    
    metrics['prediction_df'].to_csv(f'PhyDNet/runs/{save_name}/prediction_df.csv')

    metrics_per_shot = cmc.per_shot_test(path=f'PhyDNet/runs/{save_name}', 
                            shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
                            writer=writer, num_classes=3,
                            two_images=False)
    
    metrics_per_shot = pd.DataFrame(metrics_per_shot)
    metrics_per_shot.to_csv(f'PhyDNet/runs/{save_name}/metrics_per_shot.csv')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    train_and_eval_PhyDNet(batch_size=8, learning_rate_max=0.001, num_epochs=12,
                           test_run=True, test_df_contains_val_df=True, n_frames_input=5, num_workers=2,
                           weight_decay=1e-5, save_name='phydnet, smaller time-diff, 5 images, onecycleLR')
    print('Done')