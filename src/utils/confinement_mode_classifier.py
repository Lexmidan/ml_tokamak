import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import re
import torch
import pandas as pd
import torchvision
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall
from torch.optim import lr_scheduler
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import copy
from torch.utils.tensorboard import SummaryWriter
import time 





# # Setting the seed
# pl.seed_everything(42)
# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



####################### Create datasets and dataloaders #######################

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, annotations, img_dir, mean, std, augmentation = False, grayscale = False):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        if augmentation:
            self.transformations = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),  # Example for additional augmentation
            transforms.RandomAffine(12, translate=(0.1, 0.1)),  # Random rotation between -12 and 12 degrees + 10% translation
            AddRandomNoise(0., 0.05),  # Add random noise
            ])
        elif grayscale:
            self.transformations = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            #transforms.ToTensor()
            ])
        else:
            self.transformations = transforms.Lambda(lambda x: x) #Identity transformation

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
        #load image
        image = read_image(img_path).float()
        #Normalize and then augment the image
        augmented_image = self.transformations((image - self.mean[:, None, None])/(255 * self.std[:, None, None]))
        #Get label and time
        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        return {'img': augmented_image,'label': label, 'path': img_path, 'time': time}



class TwoImagesDataset(Dataset):
    """DEPRECATED Dataset for two images, either RIS1 and RIS2 or two subsequential images from RIS1."""
    def __init__(self, annotations, img_dir, mean, std, second_img_opt: str = 'RIS2', augmentation = False):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.option = second_img_opt
        if augmentation:
            if np.random.rand() > 0.67: #1/3 of the time we augment the data
                self.transformations = transforms.Compose([
                transforms.RandomAffine(12, translate=(0.1, 0.1)),  # Random rotation between -12 and 12 degrees + 10% translation
                AddRandomNoise(0., 0.05),  # Add random noise
                ])
            else:
                self.transformations = transforms.Lambda(lambda x: x) #Identity transformation
        else:
            self.transformations = transforms.Lambda(lambda x: x) #Identity transformation
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        first_img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
        first_image = read_image(first_img_path).float()
        normalized_first_image = (first_image/255 - self.mean[:, None, None])/(self.std[:, None, None])
        
        #Dealing second image
        if self.option == 'RIS2': 
            second_img_path = first_img_path.replace('RIS1', 'RIS2')
        # Then we have a second image from the same RIS, but taken at previous time
        elif  idx-1 < 0: #If previous img doesn't exist, use the same image twice
            second_img_path = first_img_path
        else:
            second_img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-1, 'filename'])
        second_image = read_image(second_img_path).float()
        normalized_second_image = (second_image/255 - self.mean[:, None, None])/(self.std[:, None, None])

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
  
        labeled_imgs_dict = {'img': torch.cat((normalized_first_image.unsqueeze(0), 
                                               normalized_second_image.unsqueeze(0)), dim=0),
                             'label':label, 'path':first_img_path, 'time':time}

        return labeled_imgs_dict

    
class TwoImagesModel(nn.Module):
    """
    DEPRECATED
    Initializes the TwoImagesModel composed of two pretrained resnet.
    Removes last fc layer and connects the logits with Sequential.
    Can be used for models trained RIS1 and RIS2 respectively, or for two subsequential images from RIS1

    Parameters:
    - model (torch.nn.Module): Pre-trained base model.

    - hidden_units (int): Number of hidden units in the classifier layer.
    """
    def __init__(self, modelA, modelB, hidden_units):
        super(TwoImagesModel, self).__init__()

        num_ftrs_A = modelA.fc.in_features
        num_ftrs_B = modelB.fc.in_features

        self.modelA = copy.deepcopy(modelA)
        self.modelA.fc = nn.Linear(num_ftrs_A, 256)

        self.modelB = copy.deepcopy(modelB)
        self.modelB.fc = nn.Linear(num_ftrs_B, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, hidden_units),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_units, 3) #3 for L, H, ELM
        )

    def forward(self, imgs):
        xA = self.modelA(imgs[:, 0])
        xB = self.modelB(imgs[:, 1])

        x = torch.cat((xA, xB), dim=1)

        x = self.classifier(x)
        
        
        return x   
    

def _apply_exponential_elm_decay(df):
    """
    Helper function to apply exponential ELM decay logic to a dataframe.
    This creates soft labels for ELM transitions.
    """
    df['soft_label'] = df.apply(lambda x: [0, 1, 0] if x['mode'] == 'H-mode' else [1, 0, 0], axis=1)            
    #Pre peak and post peak time
    pre_time = 1
    post_time = 2
    if df['mode'].str.contains('ELM-peak').any():
        for elm_peak in df[df['mode'] == 'ELM-peak']['time']:
            
            # Pre-ELM probabilities
            pre_indices = df.loc[df['time'].between(elm_peak-pre_time, elm_peak)].index
            if not pre_indices.empty:
                pre_elm_prob = np.exp(-5 * np.linspace(pre_time, 0, len(pre_indices)))
                for i, prob in zip(pre_indices, pre_elm_prob):
                    df.at[i, 'soft_label'] = [0, 1 - np.max([prob, df.at[i, 'soft_label'][2]]), 
                                            np.max([prob, df.at[i, 'soft_label'][2]])]
            
            # Post-ELM probabilities
            post_indices = df.loc[df['time'].between(elm_peak, elm_peak+post_time)].index
            if not post_indices.empty:
                post_elm_prob = np.exp(-3 * np.linspace(0, post_time, len(post_indices)))
                for i, prob in zip(post_indices, post_elm_prob):
                    df.at[i, 'soft_label'] = [0, 1 - np.max([prob, df.at[i, 'soft_label'][2]]), 
                                            np.max([prob, df.at[i, 'soft_label'][2]])]
    return df


def load_and_split_dataframes(path:Path, shots:list, shots_for_training:list, shots_for_testing:list, shots_for_validation:list,
                            use_ELMS: bool = True, ris_option: str = 'RIS1', exponential_elm_decay: bool=True):
    '''
    Takes path and lists of shots. Shots not specified in shots_for_testing
    and shots_for_validation will be used for training. Returns test_df, val_df, train_df 
    'mode' columns is then transformed to [0,1,2] notation, where 0 stands for L-mode, 1 for H-mode and 2 for ELM

    use_for_PhyDNet - if True, then simply cuts the first 11 rows to avoid NaNs in PhyDNet
    '''
    if ris_option not in ['RIS1', 'RIS2', 'both']:
        raise Exception("Invalid ris_option. Choose from 'RIS1', 'RIS2', 'both'")
    
    shot_df = pd.DataFrame([])

    # Load shot usage information for 'both' cameras option
    if ris_option == 'both':
        shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
        shot_usage_dict = shot_usage.set_index('shot')[['used_for_ris1', 'used_for_ris2']].to_dict('index')

    for shot in shots:
        if ris_option == 'both':
            # Handle both cameras case - load data from available cameras
            if shot not in shot_usage_dict:
                print(f"Warning: Shot {shot} not found in shot_usageNEW.csv, skipping...")
                continue
                
            shot_info = shot_usage_dict[shot]
            
            # Load RIS1 data if available
            if shot_info['used_for_ris1']:
                try:
                    df_ris1 = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
                    df_ris1['shot'] = shot
                    df_ris1 = df_ris1.iloc[:-100]  # Drop last 100 rows
                    
                    if exponential_elm_decay and use_ELMS:
                        df_ris1 = _apply_exponential_elm_decay(df_ris1)
                            
                    shot_df = pd.concat([shot_df, df_ris1], axis=0)
                except FileNotFoundError:
                    print(f"Warning: RIS1 data file not found for shot {shot}")
            
            # Load RIS2 data if available
            if shot_info['used_for_ris2']:
                try:
                    df_ris2 = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
                    df_ris2['shot'] = shot
                    df_ris2 = df_ris2.iloc[:-100]  # Drop last 100 rows
                    # Convert RIS1 filenames to RIS2
                    df_ris2['filename'] = df_ris2['filename'].str.replace('RIS1', 'RIS2')
                    
                    # Filter out rows where RIS2 image doesn't exist
                    def image_exists(filename):
                        return os.path.exists(os.path.join(path, filename))
                    
                    exists_mask = df_ris2['filename'].apply(image_exists)
                    df_ris2 = df_ris2[exists_mask]
                    
                    if exponential_elm_decay and use_ELMS:
                        df_ris2 = _apply_exponential_elm_decay(df_ris2)
                    
                    shot_df = pd.concat([shot_df, df_ris2], axis=0)
                except FileNotFoundError:
                    print(f"Warning: RIS2 data file not found for shot {shot}")
        else:
            # Original single camera logic
            df = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
            df['shot'] = shot
            df = df.iloc[:-100] #Drop last 100 rows, because sometimes RIS cameras don't end at the same time :C
            
            if exponential_elm_decay and use_ELMS:
                df = _apply_exponential_elm_decay(df)
                        
            shot_df = pd.concat([shot_df, df], axis=0)


    df_mode = shot_df['mode'].copy()
    df_mode[shot_df['mode']=='L-mode'] = 0
    df_mode[shot_df['mode']=='H-mode'] = 1
    df_mode[shot_df['mode']=='ELM'] = 2 if use_ELMS else 1

    shot_df['mode'] = df_mode
    shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

    # Only apply RIS2 filename replacement for single-camera RIS2 mode
    # For 'both' mode, filenames are already handled per camera in the loop above
    if ris_option == 'RIS2':
        shot_df['filename'] = shot_df['filename'].str.replace('RIS1', 'RIS2')

                            
    test_df = shot_df[shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
    val_df = shot_df[shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = shot_df[shot_df['shot'].isin(shots_for_training)].reset_index(drop=True)

    return shot_df, test_df, val_df, train_df


def get_dloader(df: pd.DataFrame, path: Path, batch_size: int = 32, 
                balance_data: bool = True, shuffle: bool = True,
                second_img_opt: str = None, num_workers: int = 0, 
                augmentation: bool = False, grayscale: bool = False):
    """
    Gets dataframe, path and batch size, returns "equiprobable" dataloader

    Args:
        df: should contain columns with time, confinement mode in [0,1,2] notation and filename of the images
        path: path where images are located
        batch_size: batch size
        balance_data: uses sampler if True
        shuffle: shuffles data if True
        second_img_opt: Either RIS2 image taken in the same time as RIS1, or RIS1 taken in time t-dt
        only_halpha: returns dataset with time label and h_alpha
        h_alpha_window: how many datapoints of h_alpha from previous times will be used
    Returns:  
        dataloader: dloader, which returns each class with the same probability

    Example:
    >>> get_dloader(df, path, batch_size=32, balance_data=True, shuffle=True, only_halpha=False, second_img_opt=None, h_alpha_window=0)

    """

    if shuffle and balance_data:
        raise Exception("Can't use data shuffling and balancing simultaneously")
    
    if second_img_opt == None:
        dataset = ImageDataset(annotations=df, img_dir=path, mean=mean, 
                               std=std, augmentation=augmentation, grayscale=grayscale)
    elif second_img_opt == 'RIS2':
        dataset = TwoImagesDataset(annotations=df, img_dir=path, mean=mean, std=std, 
                                    second_img_opt='RIS2', augmentation=augmentation)
    elif second_img_opt == 'RIS1':
        dataset = TwoImagesDataset(annotations=df, img_dir=path, mean=mean, std=std, 
                                    second_img_opt='RIS1', augmentation=augmentation)
    else:
        raise Exception("Can't initiate a dataset with given kwargs")

    if balance_data:
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
                                shuffle=shuffle, 
                                num_workers=num_workers)

    return dataloader


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(inp)
    ax.grid(False)
    if title is not None:
        plt.title(title)

def check_nan_on_row(row_data):
    """
    Check if any container on the row contains NaN values.
    
    Args:
        row_data: List or array representing the data on the row.
        
    Returns:
        bool: True if NaN is present, False otherwise.
    """
    for container in row_data:
        if isinstance(container, (list, np.ndarray)):
            if np.isnan(container).any():
                return True
    return False

################### Helping functions to track the model training #############

def images_to_probs(net, batch_of_imgs):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    ''' 
    with torch.no_grad():
        output = net(batch_of_imgs)
    # convert output probabilities to predicted class
    max_logit, class_prediction = torch.max(output, 1) 
    preds = np.squeeze(class_prediction.cpu().numpy())
    return output, preds, torch.nn.functional.softmax(output, dim=1)


def plot_classes_preds(net, images, img_paths, labels, identificator):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along   
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    modes = ['L-mode', 'H-mode', 'ELM']
    _, preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(16,9))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        image = read_image(img_paths[idx]).numpy()
        plt.grid(False)
        plt.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("Phase {3}, Prediction: {0}, {1:.1f}%\n(Label: {2})".format(
            modes[preds[idx]],
            probs[idx] * 100.0,
            modes[labels[idx]],
            identificator),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    #plt.savefig(f'{path}/preds_vs_actuals/preds_vs_actuals_{timestamp}_{identificator}.jpg')
    return fig


##################### Define model training function ##########################

def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict,
                 writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25,
                 chkpt_path=os.getcwd(), signal_name='img', device = torch.device("cuda:0"),
                 return_best_model = False):

    since = time.time()
    
    # Get logger from parent module if it exists, otherwise create a simple one
    logger = logging.getLogger('LHmode_classifier') if logging.getLogger('LHmode_classifier').handlers else logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    best_acc = 0.0

    total_loss = {'train': 0.0, 'val': 0.0}
    total_batch = {'train': 0, 'val': 0}

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
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
                    outputs = model(inputs) #2D tensor with shape Batchsize*len(modes)
                    
                    if torch.isnan(outputs).any():
                        logger.warning(f'NaN values found in outputs in {phase} phase. Skipping this batch.')
                        continue

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long() if len(labels.size())==1 else labels)

                    if torch.isnan(loss):
                        logger.warning(f'Loss is NaN in {phase} phase. Skipping this batch.')
                        continue

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if total_batch['train'] == 0:
                            total_loss['train'] = 0
                        else:
                            total_loss['train'] = (0.75*(total_loss['train']) + 0.25*loss.item())/(1-0.75**total_batch['train'])
                        total_batch['train'] += 1
                    else:
                        if total_batch['val'] == 0:
                            total_loss['val'] = 0
                        else:
                            total_loss['val'] = (0.75*(total_loss['val']) + 0.25*loss.item())/(1-0.75**total_batch['val'])                      
                        total_batch['val'] += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == ground_truth.data) #How many correct answers
                
                
                #tensorboard part
                
                if running_batch % int(len(dataloaders[phase])/10)==int(len(dataloaders[phase])/10)-1: 
                    # ...log the running loss
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing loss', total_loss[phase] / total_batch[phase],
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
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar(f'accuracy', epoch_acc, epoch)
                writer.close()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), chkpt_path)

        time_elapsed = time.time() - since

        # load best model weights
    if return_best_model:
        model.load_state_dict(torch.load(chkpt_path))
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')
    return model


def test_model(run_path,
               model, test_dataloader: DataLoader,
               max_batch: int = 0, return_metrics: bool = True, num_classes: int = 3,
               comment: str ='', signal_name: str = 'divlp', writer: SummaryWriter = None,
               device = torch.device("cuda:0")):
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
    for batch in tqdm(test_dataloader, desc='Processing batches'):
        batch_index +=1
        outputs, y_hat, softmax_out = images_to_probs(model, batch[signal_name].to(device).float())
        softmax_out = softmax_out.cpu()
        y_hat = torch.tensor(y_hat)
        #Here I use if statement, because sometimes the model returns 2D tensor with soft labels
        y_df = torch.cat((y_df.int(), batch['label'] if len(batch['label'].size())==1 else batch['label'].max(axis=1)[1]), dim=0)
        y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
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
        
    if return_metrics:
        # Get logger from parent module if it exists, otherwise create a simple one
        logger = logging.getLogger('LHmode_classifier') if logging.getLogger('LHmode_classifier').handlers else logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
        logger.info('Processing metrics...')
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes)
        confusion_matrix_metric.update(y_hat_df, y_df)
        conf_matrix_fig, conf_matrix_ax  = confusion_matrix_metric.plot()
        #F1
        f1 = f1_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'macro')

        #Precision and recall
        precision = precision_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'macro')
        recall = recall_score(y_df, y_hat_df, average='binary' if num_classes==2 else 'macro')

        #Precision_recall and ROC curves are generated using the pr_roc_auc()
        pr_roc = pr_roc_auc(y_df, softmax_out_df, task="binary" if num_classes==2 else 'ternary')
        pr_fig = pr_roc['pr_curve'][0]
        roc_fig = pr_roc['roc_curve'][0]
        roc_ax = pr_roc['roc_curve'][1]

        #Accuracy
        accuracy = len(preds[preds['prediction']==preds['label']])/len(preds)

        textstr = '\n'.join((
            f'Whole test dset',
            r'threshhold = 0.5:',
            r'f1=%.2f' % (f1, ),
            r'precision=%.2f' % (precision, ),
            r'recall=%.2f' % (recall, ),
            r'accuracy=%.2f' % (accuracy, )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        conf_matrix_ax.set_title(f'confusion matrix for whole test dset')
        roc_ax.text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)

        # Open the saved images using Pillow
        roc_img = matplotlib_figure_to_pil_image(roc_fig)
        pr_img = matplotlib_figure_to_pil_image(pr_fig)

        if num_classes==3: #otherwise the images are too wide (problem with 3 colorbars)
            roc_img = roc_img.crop([int(0.06*roc_img.width), 0, int(0.8*roc_img.width), roc_img.height])
            pr_img = pr_img.crop([int(0.06*pr_img.width), 0, int(0.8*pr_img.width), pr_img.height])
        conf_matrix_img = matplotlib_figure_to_pil_image(conf_matrix_fig)
        
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
                'precision': precision, 'recall': recall, 'accuracy': accuracy, 'pr_roc_curves': pr_roc}
                
    else:
        return {'prediction_df': preds}
    

def per_shot_test(path, shots: list, results_df: pd.DataFrame, 
                  writer: SummaryWriter = None, 
                  save_metrics_imgs: bool = True, 
                  num_classes: int = 3, two_images: bool = False):
    '''
    Takes model's results dataframe from confinement_mode_classifier.test_model() and shot numbers.
    Returns metrics of model for each shot separately

    Args: 
        path: path where to save the results
        shots: list with numbers of shot to be tested on.
        model: ResNet model
        results_df: pd.DataFrame from confinement_mode_classifier.test_model().
        time_confidence_img: Image with model confidence on separate shot
        roc_img: Image with ROC 
        conf_matrix_img: Image with confusion matrix
        combined_image: Combined image with three previous returns
    Returns:
        metrics: Path where images are saved
    '''
    metrics = {'shot':[], 'f1':[], 'precision':[], 'recall':[], 'kappa':[]}
    
    # Get logger from parent module if it exists, otherwise create a simple one
    logger = logging.getLogger('LHmode_classifier') if logging.getLogger('LHmode_classifier').handlers else logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    for shot in tqdm(shots):
        pred_for_shot = results_df[results_df['shot']==shot]

        if len(pred_for_shot)==0:
            logger.warning(f'Shot {shot} not found in the results_df. Probably L-mode was removed from the dataset, and shot {shot} was L-mode only shot.')
            continue

        # Sort by time to ensure proper line plotting
        pred_for_shot = pred_for_shot.reset_index(drop=True)

        metrics['shot'].append(shot)
        preds_tensor = torch.tensor(pred_for_shot['prediction'].values.astype(float))
        labels_tensor = torch.tensor(pred_for_shot['label'].values.astype(int))
        
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes)
        confusion_matrix_metric.update(preds_tensor, labels_tensor)
        conf_matrix_fig, conf_matrix_ax = confusion_matrix_metric.plot()

        conf_time_fig, conf_time_ax = plt.subplots(figsize=(10,6))

        pred_for_shot_ris1, pred_for_shot_ris2 = split_into_two_monotonic_dfs(pred_for_shot)
        if two_images and len(pred_for_shot_ris2)!=0: #If the model takes data from the shot twice (one for each RIS), then split the data. 
            #I assume, that the first half of the data is from RIS1 and the second half from RIS2. 
            #It should be by how the split_df() function works
            
            conf_time_ax.plot(pred_for_shot_ris1['time'],pred_for_shot_ris1['prob_1'],
                              label='1st class Confidence RIS1', alpha=0.5)
            conf_time_ax.plot(pred_for_shot_ris2['time'],pred_for_shot_ris2['prob_1'], 
                              label='1st class Confidence RIS2', alpha=0.5)

            conf_time_ax.scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==1]['time'], 
                          len(pred_for_shot_ris1[pred_for_shot_ris1['label']==1])*[1], 
                          s=10, alpha=1, label='1st class Truth', color='maroon')
            
            kappa1 = cohen_kappa_score(pred_for_shot_ris1['label'], pred_for_shot_ris1['prediction'])
            kappa2 = cohen_kappa_score(pred_for_shot_ris2['label'], pred_for_shot_ris2['prediction'])
            
            #Convert to tensors (otherwise torchmetrics won't work)
            y_hat_ris1 = torch.tensor(pred_for_shot_ris1['prediction'].values.astype(float))
            y_ris1 = torch.tensor(pred_for_shot_ris1['label'].values.astype(int))

            y_hat_ris2 = torch.tensor(pred_for_shot_ris2['prediction'].values.astype(float))
            y_ris2 = torch.tensor(pred_for_shot_ris2['label'].values.astype(int))

            #Metrics
            f1_ris1 = f1_score(y_ris1, y_hat_ris1, average='binary' if num_classes==2 else 'macro')
            f1_ris2 = f1_score(y_ris2, y_hat_ris2, average='binary' if num_classes==2 else 'macro')

            precision_ris1 = precision_score(y_ris1, y_hat_ris1, average='binary' if num_classes==2 else 'macro')
            precision_ris2 = precision_score(y_ris2, y_hat_ris2, average='binary' if num_classes==2 else 'macro')

            recall_ris1 = recall_score(y_ris1, y_hat_ris1, average='binary' if num_classes==2 else 'macro')
            recall_ris2 = recall_score(y_ris2, y_hat_ris2, average='binary' if num_classes==2 else 'macro')

            conf_time_ax.set_title(f'Shot {shot}, RIS1/RIS2: kappa = {kappa1:.2f}/{kappa2:.2f}, F1 = {f1_ris1:.2f}/{f1_ris2:.2f}, Precision = {precision_ris1:.2f}/{precision_ris2:.2f}, Recall = {recall_ris1:.2f}/{recall_ris2:.2f}')
            
            kappa = (kappa1 + kappa2)/2
            f1 = (f1_ris1 + f1_ris2)/2
            precision = (precision_ris1 + precision_ris2)/2
            recall = (recall_ris1 + recall_ris2)/2

            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['kappa'].append(kappa)

            if num_classes==3:
                conf_time_ax.plot(pred_for_shot_ris1['time'],-pred_for_shot_ris1['prob_2'], 
                                  label='2d class Confidence RIS1', alpha=0.5)
                conf_time_ax.plot(pred_for_shot_ris2['time'],-pred_for_shot_ris2['prob_2'], 
                                  label='2d class Confidence RIS2', alpha=0.5)
                conf_time_ax.scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==2]['time'], 
                    len(pred_for_shot_ris1[pred_for_shot_ris1['label']==2])*[-1], 
                    s=10, alpha=1, label='ELM Truth', color='royalblue')
                
        else:
            conf_time_ax.plot(pred_for_shot['time'],pred_for_shot['prob_1'], label='Confidence')

            kappa = cohen_kappa_score(pred_for_shot['prediction'].astype(int), pred_for_shot['label'])
            f1 = f1_score(labels_tensor, preds_tensor, average='binary' if num_classes==2 else 'micro')
            precision = precision_score(labels_tensor, preds_tensor, average='binary' if num_classes==2 else 'micro')
            recall = recall_score(labels_tensor, preds_tensor, average='binary' if num_classes==2 else 'micro')

            conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==1]['time'], 
                            len(pred_for_shot[pred_for_shot['label']==1])*[1], 
                            s=10, alpha=1, label='1st class Truth', color='maroon')
            
            conf_time_ax.set_title(f'Shot {shot}, kappa = {kappa:.2f}, F1 = {f1:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}')
            
            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['kappa'].append(kappa)

            if num_classes==3:
                conf_time_ax.plot(pred_for_shot['time'],-pred_for_shot['prob_2'], label='2d class Confidence')
                conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==2]['time'], 
                                len(pred_for_shot[pred_for_shot['label']==2])*[-1], 
                                s=10, alpha=1, label='ELM Truth', color='royalblue')

        conf_time_ax.set_xlabel('t [ms]')
        conf_time_ax.set_ylabel('Confidence')

        
        conf_time_ax.legend()

        conf_matrix_ax.set_title(f'confusion matrix for shot {shot}')
        conf_matrix_fig.set_figheight(conf_time_fig.get_size_inches()[1])

        # Open the saved images using Pillow
        time_confidence_img = matplotlib_figure_to_pil_image(conf_time_fig)
        conf_matrix_img = matplotlib_figure_to_pil_image(conf_matrix_fig)

        combined_image = Image.new('RGB', (time_confidence_img.width + conf_matrix_img.width,
                                            time_confidence_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(time_confidence_img, (0, 0))
        combined_image.paste(conf_matrix_img, (time_confidence_img.width, 0))

        # Save the combined image
        if save_metrics_imgs:
            combined_image.save(f'{path}/metrics_for_shot_{shot}.png')

                # Save the images to tensorboard
        if writer:
            combined_image_tensor = torchvision.transforms.ToTensor()(combined_image)
            writer.add_image(f'metrics_for_test_shot_{shot}', combined_image_tensor)

    return metrics


def matplotlib_figure_to_pil_image(fig):
    """
    Convert a Matplotlib figure to a PIL Image.

    Parameters:
    - fig (matplotlib.figure.Figure): The Matplotlib figure to be converted.

    Returns:
    - PIL.Image.Image: The corresponding PIL Image.

    Example:
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3, 4], [10, 5, 20, 15])
    >>> pil_image = matplotlib_figure_to_pil_image(fig)
    >>> pil_image.save("output_image.png")
    >>> pil_image.show()
    """
    # Create a FigureCanvasAgg to render the figure
    canvas = FigureCanvasAgg(fig)

    # Render the figure to a bitmap
    canvas.draw()

    # Get the RGB buffer from the bitmap
    buf = canvas.buffer_rgba()

    # Convert the buffer to a PIL Image
    image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)

    return image


def pr_roc_auc(y_true, y_pred, cmap='viridis', task='binary'):
    """
    Calculate the PR (Precision-Recall) and ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve)
    for a binary and ternary classification problem.

    Args:
        y_true (torch.Tensor): True labels of the binary classification problem.
        y_pred (torch.Tensor): Predicted probabilities of the positive class.

    Returns:
        dict: A dictionary containing the PR curve, ROC curve, PR AUC, and ROC AUC.

    """

    #ensure both tensors are on cpu
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    def binary_pr_roc_auc(y_true, y_pred):
        # Sort predictions and corresponding labels
        sorted_indices = torch.argsort(y_pred, descending=True)
        sorted_pred = y_pred[sorted_indices]
        sorted_true = y_true[sorted_indices]

        # Calculate TP, FP, FN for each threshold
        true_positives = torch.cumsum(sorted_true, dim=0)
        false_positives = torch.cumsum(1 - sorted_true, dim=0)
        total_positives = true_positives[-1]
        total_negatives = false_positives[-1]

        # Precision and Recall and false positive rate calculations
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_positives
        fpr = false_positives / total_negatives

        # Calculate AUC
        auc_pr = calculate_auc(recall, precision)
        auc_roc = calculate_auc(fpr, recall)

        return precision, recall, fpr, auc_roc, auc_pr, sorted_pred

    if task == 'binary':
        precision, recall, fpr, auc_roc, auc_pr, sorted_pred = binary_pr_roc_auc(y_true, y_pred[:,1])

        # Create the PR and ROC curves
        fig_pr, ax_pr = plt.subplots()
        scatter_pr = ax_pr.scatter(recall, precision, c=sorted_pred, cmap=cmap, s=2)
        ax_pr.set_xlim(-0.07, 1.07)
        ax_pr.set_ylim(-0.07, 1.07)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR AUC: {auc_pr:.4f}')
        ax_pr.grid(True)

        fig_roc, ax_roc = plt.subplots()
        scatter_roc = ax_roc.scatter(fpr, recall, c=sorted_pred, cmap=cmap, s=2)
        ax_roc.set_xlim(-0.07, 1.07)
        ax_roc.set_ylim(-0.07, 1.07)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f'ROC AUC: {auc_roc:.4f}')
        ax_roc.grid(True)

        # Add colorbar
        cbar_pr = fig_pr.colorbar(scatter_pr)
        cbar_pr.set_label('Threshold')

        cbar_roc = fig_roc.colorbar(scatter_roc)
        cbar_roc.set_label('Threshold')

        return {'pr_curve': (fig_pr, ax_pr), 'roc_curve': (fig_roc, ax_roc), 'pr_auc': auc_pr, 'roc_auc': auc_roc}

    elif task == 'ternary':
        cmaps = ['autumn', 'winter', 'cool']
        # Create the PR and ROC curves for L-mode
        L_precision, L_recall, L_fpr, L_auc_roc, L_auc_pr, L_sorted_pred =  binary_pr_roc_auc((y_true==0).long(), y_pred[:,0])
        # Create the PR and ROC curves for H-mode
        H_precision, H_recall, H_fpr, H_auc_roc, H_auc_pr, H_sorted_pred =  binary_pr_roc_auc((y_true==1).long(), y_pred[:,1])
        # Create the PR and ROC curves for ELM
        E_precision, E_recall, E_fpr, E_auc_roc, E_auc_pr, E_sorted_pred =  binary_pr_roc_auc((y_true==2).long(), y_pred[:,2])

        # Create the PR image
        fig_pr, ax_pr = plt.subplots(figsize=(10, 5))
        scatter_pr_L = ax_pr.scatter(L_recall, L_precision, c=L_sorted_pred, cmap=cmaps[0], s=2)
        scatter_pr_H = ax_pr.scatter(H_recall, H_precision, c=H_sorted_pred, cmap=cmaps[1], s=2)
        scatter_pr_E = ax_pr.scatter(E_recall, E_precision, c=E_sorted_pred, cmap=cmaps[2], s=2)
        ax_pr.set_xlim(-0.07, 1.07)
        ax_pr.set_ylim(-0.07, 1.07)
        ax_pr.set_xlabel('Recall', fontsize=16)
        ax_pr.set_ylabel('Precision', fontsize=16)
        ax_pr.set_title(f'PR AUC L-mode: {L_auc_pr:.4f}, H-mode: {H_auc_pr:.4f}, ELM: {E_auc_pr:.4f}')
        cbar_pr = [fig_pr.colorbar(scatter_pr_L), fig_pr.colorbar(scatter_pr_H), fig_pr.colorbar(scatter_pr_E)]
        cbar_pr[0].set_label('Threshold for L-mode')
        cbar_pr[1].set_label('Threshold for H-mode')
        cbar_pr[2].set_label('Threshold for ELM')
        cbar_pr[1].set_ticks([])
        cbar_pr[2].set_ticks([])
        # Adjust positions of colorbars in order to avoid uselessly large white space
        for i, cb in enumerate(cbar_pr):
            pos = cb.ax.get_position()
            cb.ax.set_position([0.68 - i*0.05, pos.y0, 0.02, pos.height])

        #Create the ROC image

        fig_roc, ax_roc = plt.subplots(figsize=(10, 5))
        scatter_roc_L = ax_roc.scatter(L_fpr, L_recall, c=L_sorted_pred, cmap=cmaps[0], s=2)
        scatter_roc_H = ax_roc.scatter(H_fpr, H_recall, c=H_sorted_pred, cmap=cmaps[1], s=2)
        scatter_roc_E = ax_roc.scatter(E_fpr, E_recall, c=E_sorted_pred, cmap=cmaps[2], s=2)
        ax_roc.set_xlim(-0.07, 1.07)
        ax_roc.set_ylim(-0.07, 1.07)
        ax_roc.set_xlabel("False Positive Rate", fontsize=16)
        ax_roc.set_ylabel("True Positive Rate", fontsize=16)
        ax_roc.set_title(f'ROC AUC L-mode: {L_auc_roc:.4f}, H-mode: {H_auc_roc:.4f}, ELM: {E_auc_roc:.4f}')
        cbar_roc = [fig_roc.colorbar(scatter_roc_L), fig_roc.colorbar(scatter_roc_H), fig_roc.colorbar(scatter_roc_E)]
        cbar_roc[0].set_label('Threshold for L-mode')
        cbar_roc[1].set_label('Threshold for H-mode') 
        cbar_roc[2].set_label('Threshold for ELM')
        cbar_roc[1].set_ticks([])
        cbar_roc[2].set_ticks([])
        # Adjust positions of colorbars in order to avoid uselessly large white space
        for i, cb in enumerate(cbar_roc):
            pos = cb.ax.get_position()
            cb.ax.set_position([0.68 - i*0.05, pos.y0, 0.02, pos.height])

        return {'pr_curve': (fig_pr, ax_pr), 'roc_curve': (fig_roc, ax_roc), 'pr_auc': [L_auc_pr, H_auc_pr, E_auc_pr], 'roc_auc': [L_auc_roc, H_auc_roc, E_auc_roc]}
    
    else:
        raise Exception("Task should be either 'binary' or 'ternary'")

        



def calculate_auc(x, y):
    # Convert to numpy arrays for integration
    x_np = x.numpy()
    y_np = y.numpy()

    # Use numpy's trapezoidal rule integration
    auc = np.trapezoid(y_np, x_np)
    return auc

class AddRandomNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

def split_into_two_monotonic_dfs(df):
    """
    We have a df with non monotonic time column.
    Split it into two monotonic dfs.

    Parameters:
    df (pd.DataFrame): DataFrame with a 'time' column.

    Returns:
    tuple of pd.DataFrame: Two DataFrames split at the first non-monotonic point.
    """
    # Find the first non-monotonic increase point
    for i in range(1, len(df)):
        if df['time'].iloc[i] <= df['time'].iloc[i - 1]:
            # Split the DataFrame at the found index
            df1 = df.iloc[:i]
            df2 = df.iloc[i:]
            return df1, df2

    # If no non-monotonic point found, return the entire DataFrame as one part, and an empty DataFrame as the other
    return df, pd.DataFrame(columns=df.columns)