import os
import time 
from PIL import Image
import re

import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import torch.nn.functional as F
import pandas as pd
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ..utils import confinement_mode_classifier as cmc

####### Stolen from https://github.com/TheMrGhostman/InceptionTime-Pytorch/###########
def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def pass_through(X):
	return X


class Inception(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], 
              bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
		"""
		super(Inception, self).__init__()
		self.return_indices=return_indices
		if in_channels > 1:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = 1

		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=n_filters, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		Z4 = self.conv_from_maxpool(Z_maxpool)
		# step 3 
		Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, 
              use_residual=True, activation=nn.ReLU(), return_indices=False):
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.return_indices = return_indices
		self.activation = activation
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_2 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_3 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=4*n_filters
									)
								)

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
			Z, i2 = self.inception_2(Z)
			Z, i3 = self.inception_3(Z)
		else:
			Z = self.inception_1(X)
			Z = self.inception_2(Z)
			Z = self.inception_3(Z)
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		if self.return_indices:
			return Z,[i1, i2, i3]
		else:
			return Z



class InceptionTranspose(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		"""
		super(InceptionTranspose, self).__init__()
		self.activation = activation
		self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.conv_to_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
		self.bottleneck = nn.Conv1d(
								in_channels=3*bottleneck_channels, 
								out_channels=out_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

		def forward(self, X, indices):
			Z1 = self.conv_to_bottleneck_1(X)
			Z2 = self.conv_to_bottleneck_2(X)
			Z3 = self.conv_to_bottleneck_3(X)
			Z4 = self.conv_to_maxpool(X)

			Z = torch.cat([Z1, Z2, Z3], axis=1)
			MUP = self.max_unpool(Z4, indices)
			BN = self.bottleneck(Z)
			# another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution
			
			return self.activation(self.batch_norm(BN + MUP))


class InceptionTransposeBlock(nn.Module):
	def __init__(self, in_channels, out_channels=32, kernel_sizes=[9,19,39], bottleneck_channels=32,
               use_residual=True, activation=nn.ReLU()):
		super(InceptionTransposeBlock, self).__init__()
		self.use_residual = use_residual
		self.activation = activation
		self.inception_1 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_2 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_3 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=out_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.ConvTranspose1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=out_channels
									)
								)

	def forward(self, X, indices):
		assert len(indices)==3
		Z = self.inception_1(X, indices[2])
		Z = self.inception_2(Z, indices[1])
		Z = self.inception_3(Z, indices[0])
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		return Z
      
class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)
##############################################################################	


def split_df(df, shots, shots_for_training, shots_for_testing, 
             shots_for_validation, signal_name, use_ELMs=True, no_L_mode = False,
             path=os.getcwd(), sampling_freq=300, exponential_elm_decay=False, only_ELMs=False):

    """
    Splits the dataframe into train, test, and validation sets and scales the data.

    Parameters:
    df (pandas.DataFrame): The input dataframe. Should contain.
    shots (list): A list of shot numbers.
    shots_for_training (list): A list of shot numbers to be used for training.
    shots_for_testing (list): A list of shot numbers to be used for testing.
    shots_for_validation (list): A list of shot numbers to be used for validation.
    signal_name (str): The name of the signal, e.g. 'mc', 'divlp'.
    use_ELMs (bool, optional): Whether to label ELMs as 3d class. Defaults to True.
    only_ELMs (bool, optional): Whether to label H-mode and L-mode as one class, and ELMs as another. Defaults to False.
    no_L_mode (bool, optional): Whether to exclude L-mode from whole dataset. Defaults to False.
    path (str, optional): The path to the data. Defaults to the current working directory.
    sampling_freq (int, optional): The sampling frequency. Defaults to 300.
    exponential_elm_decay (bool, optional): Whether to use soft labels for ELMs. Defaults to False.
    test_df_contains_val_df (bool, optional): Whether include validation dataframe to test dataframe. Defaults to True.

    Returns:
    tuple: A tuple containing the following dataframes:
        - shot_df (pandas.DataFrame): The dataframe with all shots.
        - test_df (pandas.DataFrame): The dataframe for testing.
        - val_df (pandas.DataFrame): The dataframe for validation.
        - train_df (pandas.DataFrame): The dataframe for training.
    """
    if only_ELMs: #yeaaah... this is beacause use_ELMs make ELM a 3rd class, but only_ELMs should make it 2nd class
          if use_ELMs:
                print('Warning: use_ELMs and only_ELMs are both set to True.\
                        This is not a valid combination. Setting only_ELMs to False.')
          use_ELMs = False

    signal_paths_dict = {'h_alpha': f'{path}/data/h_alpha_signal_{sampling_freq}kHz', 
                         'mc': f'{path}/data/mirnov_coil_signal_{sampling_freq}kHz',
                         'mcDIV': f'{path}/data/mirnov_coil_signal_{sampling_freq}kHz', 
                         'mcHFS': f'{path}/data/mirnov_coil_signal_{sampling_freq}kHz', 
                         'mcLFS': f'{path}/data/mirnov_coil_signal_{sampling_freq}kHz', 
                         'mcTOP': f'{path}/data/mirnov_coil_signal_{sampling_freq}kHz', 
                         'divlp': f'{path}/data/langmuir_probe_signal_{sampling_freq}kHz',
                         'mc_h_alpha': f'{path}/data/mirnov_h_alpha_signal_{sampling_freq}kHz'}
    
    if signal_name not in ['divlp', 'mc', 'mcDIV', 'mcHFS', 'mcLFS', 'mcTOP', 'h_alpha', 'mc_h_alpha']:
            raise ValueError(f'{signal_name} is not a valid signal name. Please use one of the following: divlp, mc, h_alpha')
    
    #Create a dataframe with all the shots
    shot_df = pd.DataFrame()
    for shot in shots:
        df = pd.read_csv(f'{signal_paths_dict[signal_name]}/shot_{shot}.csv')
        df['shot'] = shot

        # Load the dataset
        if exponential_elm_decay and no_L_mode: #This creates soft labels for 2 classes
            df['soft_label'] = df.apply(lambda x: [1, 0] if x['mode'] == 'H-mode' else [1, 0], axis=1)            
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
                            df.at[i, 'soft_label'] = [1 - np.max([prob, df.at[i, 'soft_label'][1]]), 
                                                    np.max([prob, df.at[i, 'soft_label'][1]])]
                    
                    # Post-ELM probabilities
                    post_indices = df.loc[df['time'].between(elm_peak, elm_peak+post_time)].index
                    if not post_indices.empty:
                        post_elm_prob = np.exp(-3 * np.linspace(0, post_time, len(post_indices)))
                        for i, prob in zip(post_indices, post_elm_prob):
                            df.at[i, 'soft_label'] = [1 - np.max([prob, df.at[i, 'soft_label'][1]]), 
                                                    np.max([prob, df.at[i, 'soft_label'][1]])]
                            
        elif exponential_elm_decay and not no_L_mode: #This creates soft labels for 2 classes
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
                            
            # OLD IMPLEMENTATION OF SOFT LABELS FOR 3 CLASSES - TAKES TOO LONG
            # df['mode'] = df['soft_label']
            # df.drop('soft_label', axis=1, inplace=True)

            #     # Identify ELM-peak events and their times
            #     elm_peak_times = df[df['mode'] == 'ELM-peak']['time']

            #     # Calculate the minimum time difference from ELM-peak for all rows
            #     min_time_diff = df['time'].apply(lambda row_time: np.min(np.abs(elm_peak_times - row_time)))

            #     # Apply exponential decay to the time difference to calculate the probability
            #     df['prob'] = np.exp(-3 * min_time_diff)
            # else:
            #     df['prob'] = 0

            # # One-hot encoding mapping for modes
            # mode_to_one_hot = {
            #     'L-mode': [1., 0, 0],
            #     'H-mode': [0, 0., 0],
            #     'ELM': [0, 0, .0],
            #     'ELM-peak': [0, 0, .0]
            # }
            # df['probs']=df.apply(lambda x: [apply_mode_prob_adjusted(x, 
            #                  mode_to_one_hot=mode_to_one_hot)], axis=1, result_type='expand')
                     

        shot_df = pd.concat([shot_df, df], axis=0)

    #Replace the mode with a number
    df_mode = shot_df['mode'].copy()
    df_mode[shot_df['mode']=='L-mode'] = 0
    df_mode[shot_df['mode']=='H-mode'] = 1 if not only_ELMs else 0
    df_mode[shot_df['mode'].isin(['ELM', 'ELM-peak'])] = 2 if use_ELMs else 1
    shot_df['mode'] = df_mode

    if no_L_mode:
        shot_df = shot_df[shot_df['mode'] != 0]
        shot_df['mode'] = shot_df['mode'].map({1: 0, 2: 1}) #map H-mode to 0 and ELM to 1

    shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

    
    val_df = shot_df[shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = shot_df[shot_df['shot'].isin(shots_for_training)].reset_index(drop=True)
    test_df = shot_df[shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)

    return shot_df, test_df, val_df, train_df

# Apply probabilities to one-hot encoding for modes
def apply_mode_prob_adjusted(row, mode_to_one_hot):
    one_hot = np.array(mode_to_one_hot[row['mode']])
    # Apply probability only to the last element (ELM/ELM-peak)
    one_hot[2] += row['prob'] #Probability of ELM
    one_hot[1] += 0 if row['mode'] == 'L-mode' else 1 - row['prob'] #Probability of H-mode
    return one_hot

class SignalDataset(Dataset):
    '''
        Parameters:
            df (DataFrame): The DataFrame containing annotations for the dataset.
            window (int): The size of the time window for fetching sequential data.

    '''

    def __init__(self, df, window, signal_name='divlp', dpoints_in_future=160):
        self.df = df
        self.window = window
        self.signal_name = signal_name
        self.dpoints_in_future = dpoints_in_future

        if self.signal_name not in ['divlp', 'mcDIV', 'mcHFS', 'mcLFS', 'mcTOP', 'h_alpha']:
            raise ValueError(f'{self.signal_name} is not a valid signal name. Please use one of the following:\
                              divlp, mcDIV, mcHFS, mcLFS, mcTOP, h_alpha')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        signal_window = torch.tensor([])

        if idx < self.window - self.dpoints_in_future or idx > len(self.df) - self.dpoints_in_future: 
            signal_window = torch.full((self.window,), self.df.iloc[0][f'{self.signal_name}'])

        else:
            signal_window = torch.tensor(self.df.iloc[idx-(self.window-self.dpoints_in_future) \
                                                      : idx+self.dpoints_in_future]
                                         [f'{self.signal_name}'].to_numpy())

        label = self.df.iloc[idx]['mode']
        time = self.df.iloc[idx]['time']
        shot_num = self.df.iloc[idx]['shot']
        return {'label': label, 
                'time': time, 
                f'{self.signal_name}': signal_window, 
                'shot': shot_num.astype(int)}


class MultipleMirnovCoilsDataset(Dataset):
    '''
        Parameters:
            df (DataFrame): The DataFrame containing annotations for the dataset.
            window (int): The size of the time window for fetching sequential data.

    '''

    def __init__(self, df, window, dpoints_in_future=160, h_alpha_instead_mcLFS=False):
        
        self.df = df
        self.columns = ['mcDIV', 'mcHFS', 'mcLFS', 'mcTOP']
        self.window = window
        self.dpoints_in_future = dpoints_in_future
        self.label_column = 'soft_label' if 'soft_label' in self.df.columns else 'mode'
        self.signal_name = 'mc'

        if h_alpha_instead_mcLFS:
            self.signal_name = 'mc_h_alpha'
            self.columns = ['mcDIV', 'mcHFS', 'h_alpha', 'mcTOP']
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        signal_window = torch.tensor([])

        if idx < self.window - self.dpoints_in_future or idx > len(self.df) - self.dpoints_in_future: 
            #TODO: This is a bit ugly, but I don't know how to do it better
            signal_window = torch.full((self.window, 4), 0.0)
        else:
            signal_window = torch.tensor(self.df.iloc[idx-(self.window-self.dpoints_in_future) : \
                                                       idx+self.dpoints_in_future][self.columns].to_numpy())
        signal_window = signal_window.transpose(0, 1) #so batch shape corresponds to [batch_size, n_channels, window]

        label = torch.tensor(self.df.iloc[idx][self.label_column], dtype=torch.float) 
        time = self.df.iloc[idx]['time']
        shot_num = self.df.iloc[idx]['shot']
        return {'label': label, 'time': time, self.signal_name: signal_window, 'shot': shot_num.astype(int)}

	
def get_dloader(df: pd.DataFrame(), batch_size: int = 32, 
                balance_data: bool = True, shuffle: bool = False,
                signal_window: int = 160, signal_name: str = 'divlp',
                num_workers: int = 8, persistent_workers: bool = True, dpoints_in_future: int = 160):
    """
    Gets dataframe, path and batch size, returns "equiprobable" dataloader

    Args:
        df: should contain columns with time, confinement mode in [0,1,2] notation and filename of the images
        batch_size: batch size
        balance_data: uses sampler if True
        shuffle: shuffles data if True
        signal_window: how many datapoints of signal from previous times will be used
    Returns:  
        dataloader: dloader, which returns each class with the same probability
    """
    if shuffle and balance_data:
        raise Exception("Can't use data shuffling and balancing together")
    
    #If we plan to use all mirnov coils, then we need to use MultipleMirnovCoilsDataset
    if signal_name == 'mc':
        dataset = MultipleMirnovCoilsDataset(df, 
                                             window=signal_window, 
                                             dpoints_in_future=dpoints_in_future)
    elif signal_name == 'mc_h_alpha':
        dataset = MultipleMirnovCoilsDataset(df, 
                                             window=signal_window, 
                                             dpoints_in_future=dpoints_in_future,
                                             h_alpha_instead_mcLFS=True)
    else:
        dataset = SignalDataset(df, 
                                window=signal_window, 
                                signal_name=signal_name, 
                                dpoints_in_future=dpoints_in_future)

    #Balance the data
    if balance_data:
        df['mode'] = df['mode'].map(lambda x: 'ELM' if x =='ELM-peak' else x)
        mode_weight = (1/df['mode'].value_counts()).values
        sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
        sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                sampler=sampler, 
                                num_workers=num_workers, 
                                persistent_workers=persistent_workers)
    else: 
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=num_workers, 
                                persistent_workers=persistent_workers)

    return dataloader

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=3, window=320, in_channels=4):
        super(Simple1DCNN, self).__init__()
        # Define the 1D convolutional layers

        self.window = window # Length of the input signal

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128,\
                               kernel_size=32, stride=1, padding=1, dilation=1)
        self.batch_norm1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=32, stride=1, padding=1, dilation=2)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.avg_pool = nn.AvgPool1d(32, stride=8, padding=0, ceil_mode=False)
        self.max_pool = nn.MaxPool1d(32, stride=8, padding=0, ceil_mode=False)

        ## calculate input size for the fully connected layer
        def output_size_of_1d_conv_layer(input_length, kernel_size, padding, dilation, stride):
            output_length = np.floor((input_length + 2*padding[0] -\
                                      dilation[0]*(kernel_size[0]-1) - 1) // stride[0]) + 1
            return output_length
        
        first_layer_out_length = output_size_of_1d_conv_layer(window, self.conv1.kernel_size, 
                                                              self.conv1.padding, 
                                                              self.conv1.dilation, 
                                                              self.conv1.stride)
        
        second_layer_out_length = output_size_of_1d_conv_layer(first_layer_out_length, 
                                                               self.conv2.kernel_size, 
                                                               self.conv2.padding, 
                                                               self.conv2.dilation, 
                                                               self.conv2.stride)
        
        pool_layer_out_length = output_size_of_1d_conv_layer(second_layer_out_length, 
                                                             self.avg_pool.kernel_size, 
                                                             self.avg_pool.padding, 
                                                             (1,), 
                                                             self.avg_pool.stride)
        
        input_length_for_fc = int(pool_layer_out_length*2*self.batch_norm2.num_features)

        # Define a fully connected layer for classification
        self.fc1 = nn.Linear(in_features=input_length_for_fc, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        if len(x.size()) == 2:  # if the shape is [batch_size, window] corresponding to 1-channel
            x = x.unsqueeze(1)  # Now shape is [batch_size, 1, window]
            
        #First Convolution Activation BatchNorm
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)

        #Second Convolution Activation BatchNorm
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)

        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat((x1, x2), dim=1)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # # Apply the fully connected layer and return the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def select_model_architecture(architecture: str, num_classes: int, window: int, in_channels: int = 1):
    """
    Selects and returns a model architecture based on the specified architecture name.

    Args:
        architecture (str): The name of the architecture to select.\
              Valid options are 'InceptionTime' and 'Simple1DCNN'.
        num_classes (int): The number of output classes for the model.
        window (int): The size of the input window of given signal.

    Returns:
        torch.nn.Module: The selected model architecture.

    Raises:
        ValueError: If the specified architecture name is not valid.
    """

    if architecture == 'InceptionTime':
        model = nn.Sequential(
                    Reshape(out_shape=(in_channels, window)),
                    InceptionBlock(
                        in_channels=in_channels, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=num_classes)
        )


    elif architecture == 'Simple1DCNN':
        model = Simple1DCNN(num_classes=num_classes, window=window, in_channels=in_channels)

    else:
        raise ValueError(f'{architecture} is not a valid architecture.\
                          Please use one of the following: InceptionTime, Simple1DCNN')
    return model

class RobustScalerNumpy:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        """
        Compute the median and IQR of X to later use for scaling.
        X should be a NumPy array.
        """
        self.median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr = q3 - q1

    def transform(self, X):
        """
        Scale features of X according to the median and IQR.
        """
        if self.median is None or self.iqr is None:
            raise RuntimeError("Must fit the scaler before transforming data.")

        # Avoid division by zero
        iqr_nonzero = np.where(self.iqr == 0, 1, self.iqr)
        return (X - self.median) / iqr_nonzero

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)