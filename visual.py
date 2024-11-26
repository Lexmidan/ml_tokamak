
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import torch
from matplotlib.patches import Circle, Arc
from IPython.display import display, clear_output
from matplotlib.image import imread
from functools import lru_cache
from torchvision.io import read_image

@lru_cache
def cached_read_csv(filepath, **kwargs):
    return pd.read_csv(filepath, **kwargs)



def visualize(path_to_run, shot,  figure_vertical_size, figure_horizontal_size, zoom_signal, zoom_time, time_for_signal):

    try:
        with open(f'/compass/Shared/Users/bogdanov/ml_tokamak/runs/{path_to_run}/hparams.json', 'r') as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print(f"Can't find the hparams. Chosen run is either irrelevant or the model is still training.")
        fig = shot_not_present(color='red')
        plt.close(fig)
        display(fig)
        return

    preds_csv = cached_read_csv(f'/compass/Shared/Users/bogdanov/ml_tokamak/runs/{path_to_run}/prediction_df.csv')
    


    if not shot in hparams['shots_for_testing']:
        print(f'Shot {shot} was not (yet?) processed in this run.')
        print(f'Model is either still training, or shot {shot} was not in the testing set.')
        fig = shot_not_present(shot)
        plt.close(fig)
        display(fig)
        return
    
    #Find the shot in the results_df
    pred_for_shot = preds_csv[preds_csv['shot']==shot]

    if len(pred_for_shot)==0:
        print(f'Shot {shot} not found in the results_df. Probably L-mode was removed from the dataset, and shot {shot} was L-mode only shot.')
        fig = shot_not_present(shot)
        plt.close(fig)
        display(fig)
        return

    #Get the metrics for the shot
    try:
        metrics_per_shot = cached_read_csv(f'/compass/Shared/Users/bogdanov/ml_tokamak/runs/{path_to_run}/metrics_per_shot.csv', index_col=0)
        kappa, f1, precision, recall = metrics_per_shot[metrics_per_shot['shot']==shot][['kappa','f1','precision','recall']].values[0]
        
    except FileNotFoundError:
        print(f"Can't find the metrics_per_shot.csv for run {path_to_run}.")
        kappa, f1, precision, recall = 99, 99, 99, 99

    #If the model takes data from the shot twice (one for each RIS), then split the data. 
    if 'ris_option' in hparams and hparams['ris_option']=='both': 
        #I assume, that the first half of the data is from RIS1 and the second half from RIS2. 
        #It should be by how the split_df() function works
        pred_for_shot_ris1, pred_for_shot_ris2 = split_into_two_monotonic_dfs(pred_for_shot)

        #Create the main figure - confidence over time
        conf_time_fig, conf_time_ax = plt.subplots(2, figsize=(10*figure_horizontal_size,9*figure_vertical_size), gridspec_kw={'height_ratios': [1, 1]})
        conf_time_ax[0].set_xlim(time_for_signal - exp_decaying(zoom_time), time_for_signal + exp_decaying(zoom_time))
        conf_time_ax[0].grid(True)

        #Plot confidence for the first class (it may be H-mode or ELM, depending on the model)
        conf_time_ax[0].plot(pred_for_shot_ris1['time'],pred_for_shot['prob_1'][:len(pred_for_shot_ris1)], 
                          label='1st class Confidence RIS1', alpha=0.5)
        conf_time_ax[0].plot(pred_for_shot_ris2['time'],pred_for_shot['prob_1'][len(pred_for_shot_ris1):], 
                          label='1st class Confidence RIS2', alpha=0.5)
        
        conf_time_ax[0].scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==1]['time'], 
                      len(pred_for_shot_ris1[pred_for_shot_ris1['label']==1])*[1], 
                      s=10, alpha=1, label='1st class Truth', color='maroon')
        
        conf_time_ax[0].vlines(time_for_signal, 0, 1, color='black', linestyle='--')
        conf_time_ax[0].set_title(f'Shot {extract_float(shot)}, RIS1/RIS2: kappa = {extract_float(kappa):.2f}, F1 = {extract_float(f1):.2f}, Precision = {extract_float(precision):.2f}, Recall = {extract_float(recall):.2f}')


        #Plot confidence for the second class (it's alway ELM)
        if 'prob_2' in pred_for_shot.columns:
            conf_time_ax[0].plot(pred_for_shot_ris1['time'],-pred_for_shot['prob_2'][:len(pred_for_shot_ris1)], 
                              label='2d class Confidence RIS1', alpha=0.5)
            
            conf_time_ax[0].plot(pred_for_shot_ris2['time'],-pred_for_shot['prob_2'][len(pred_for_shot_ris1):], 
                              label='2d class Confidence RIS2', alpha=0.5)
            
            conf_time_ax[0].scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==2]['time'], 
                len(pred_for_shot_ris1[pred_for_shot_ris1['label']==2])*[-1], 
                s=10, alpha=1, label='ELM Truth', color='royalblue')
        conf_time_ax[0].legend()

    #If the model takes data from the shot once, then plot the confidence directly
    else:
        #If the model takes image or just 1 signal, then conf_time should have 2 rows. Else 5 rows
        if 'ris_option' in hparams or not ('mc' in hparams['signal_name']):
            num_graph_rows = 2
            gridspec_kw = {'height_ratios': [1, 1]}
        else:
            num_graph_rows = 5
            gridspec_kw = {'height_ratios': [4, 1, 1, 1, 1]}

        conf_time_fig, conf_time_ax = plt.subplots(num_graph_rows, figsize=(10*figure_horizontal_size,9*figure_vertical_size), 
                                                   gridspec_kw=gridspec_kw, sharex=True if hparams["signal_name"]!='imgs_input' else False)
        conf_time_ax[0].set_xlim(time_for_signal - exp_decaying(zoom_time), time_for_signal + exp_decaying(zoom_time))


        conf_time_ax[0].plot(pred_for_shot['time'],pred_for_shot['prob_1'], label='1st class Confidence')

        conf_time_ax[0].scatter(pred_for_shot[pred_for_shot['label']==1]['time'], 
                        len(pred_for_shot[pred_for_shot['label']==1])*[1], 
                        s=10, alpha=1, label='1st class Truth', color='maroon')

        conf_time_ax[0].set_title(f'Cohen kappa = {extract_float(kappa):.2f}, F1 = {extract_float(f1):.2f}, Precision = {extract_float(precision):.2f}, Recall = {extract_float(recall):.2f}')
        conf_time_ax[0].vlines(time_for_signal, 0, 1, color='black', linestyle='--')

        if 'prob_2' in pred_for_shot.columns:
            conf_time_ax[0].plot(pred_for_shot['time'],-pred_for_shot['prob_2'], label='2d class Confidence')

            conf_time_ax[0].scatter(pred_for_shot[pred_for_shot['label']==2]['time'], 
                            len(pred_for_shot[pred_for_shot['label']==2])*[-1], 
                            s=10, alpha=1, label='ELM Truth', color='royalblue')

    ###Handle the second plot. It is either a signal or an image
    #If it's an image, then load both images for RIS1/2 and display it. Hope that user will know which image the given model used
    if 'ris_option' in hparams:
        conf_time_fig.subplots_adjust(hspace=0.5) #Add some space between the plots
        conf_time_ax[0].set_xlabel('Time [ms]')

        try:
            closest_time_str, closest_time = closest_decimal_time(time_for_signal)
            # Create an inset axis for image1 on the left half of the second axes
            

            # Create an inset axis for image2 on the right half of the second axes
            if 'model' in hparams.keys() and 'ClassifierRNN' in hparams['model']:
                
                conf_time_ax[0].vlines(time_for_signal - 4*4*0.2, 0, 1, color='green', linestyle='--', label='Signal window')
                conf_time_ax[0].set_xlim(time_for_signal - exp_decaying(zoom_time), time_for_signal + exp_decaying(zoom_time))
                image_combined = torch.tensor([])
                for i in range(4, 0, -1):
                    closest_time_i_str, closest_time_i = closest_decimal_time(time_for_signal - i*0.2)
                    image = read_image(f'/compass/Shared/Users/bogdanov/ml_tokamak/imgs/{shot}/RIS1_{shot}_t={closest_time_i_str}.png').float()
                    image_combined = torch.cat((image_combined, image[:,74:-74,144:-144].mean(dim=0, keepdim=True)), dim=2)

                conf_time_ax[1].imshow(image_combined.permute(1,2,0).cpu().numpy())
                conf_time_ax[1].axis('tight')  # Tight fit around the data
                conf_time_ax[1].axis('off')

            else:
                ax1 = conf_time_ax[1].inset_axes([0, 0, 0.5, 1])  # left, bottom, width, height
                image1 = imread(f'/compass/Shared/Users/bogdanov/ml_tokamak/imgs/{shot}/RIS1_{shot}_t={closest_time_str}.png')
                image2 = imread(f'/compass/Shared/Users/bogdanov/ml_tokamak/imgs/{shot}/RIS2_{shot}_t={closest_time_str}.png')

                ax1.imshow(image1)
                ax1.axis('off')  # Turn off axis for image1

                ax2 = conf_time_ax[1].inset_axes([0.5, 0, 0.5, 1])  # left, bottom, width, height
                ax2.imshow(image2)
                ax2.axis('off')  # Turn off axis for image2

                # Optionally turn off the main axis
                conf_time_ax[1].axis('off')
            if len(pred_for_shot) > 0:

                if len(pred_for_shot[abs(pred_for_shot["time"]-closest_time)<0.01]["label"].values)>1:
                    title = pred_for_shot[abs(pred_for_shot["time"]-closest_time)<0.01]["label"].values[0] 
                else: title = pred_for_shot[abs(pred_for_shot["time"]-closest_time)<0.01]["label"].values

                conf_time_ax[1].set_title(f'Shot {shot}, time {closest_time} ms, GT class: {title}')
            else:
                conf_time_ax[1].set_title(f'Shot {shot}, time {closest_time} ms, GT class: N/A')
            #
            #display the image
            #display(conf_time_fig)

        except FileNotFoundError:
            print(f"Can't find the image for shot {shot} and time {closest_time_str}.")

        #plt.close(img_fig)

    #If it's a signal, then load the signal and display it
    else:
        

        closest_time = pred_for_shot.iloc[(pred_for_shot['time'] - time_for_signal).abs().argmin()]
        signal_paths_dict = {'h_alpha': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/h_alpha_signal_{hparams["sampling_frequency"]}kHz', 
                            'mc': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz',
                            'mcDIV': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcHFS': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcLFS': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcTOP': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'divlp': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/langmuir_probe_signal_{hparams["sampling_frequency"]}kHz',
                            'mc_h_alpha': f'/compass/Shared/Users/bogdanov/ml_tokamak/data/mirnov_h_alpha_signal_{hparams["sampling_frequency"]}kHz'}
        
        signal_df = cached_read_csv(f'{signal_paths_dict[hparams["signal_name"]]}/shot_{shot}.csv')
        #signal_df = signal_df[(signal_df['time'] >= time_for_signal - exp_decaying(zoom_time)) & (signal_df['time'] <= time_for_signal + exp_decaying(zoom_time))]
        signal_columns = [col for col in signal_df.columns if col not in ['time', 'mode']]
        
        #If the model is trained on only one signal, then plot it directly
        if len(signal_columns) == 1:
            percentile = signal_df[signal_columns[0]].quantile(exp_decaying(zoom_signal)/1000)
            conf_time_ax[1].plot(signal_df['time'], signal_df[signal_columns[0]])
            conf_time_ax[1].set_title(signal_columns[0])
            conf_time_ax[1].set_ylabel(f'{signal_columns[0]}')
            conf_time_ax[1].set_xlabel('Time [ms]')
            conf_time_ax[1].set_ylim(0.1*percentile, 3*percentile)
            
            
            #conf_time_ax[1].set_xlim(time_for_signal - exp_decaying(zoom_signal), time_for_signal + exp_decaying(zoom_signal))

            #Plot the signal window on signal figure
            conf_time_ax[1].vlines(time_for_signal+hparams['dpoints_in_future']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--', label='Signal window')
            conf_time_ax[1].vlines(time_for_signal-(hparams['signal_window']/hparams['sampling_frequency']-hparams['dpoints_in_future']/hparams['sampling_frequency']), 0, percentile, color='green', linestyle='--')
            conf_time_ax[1].vlines(time_for_signal, 0, percentile, color='black', linestyle='--')
            #Plot the signal window on main figure
            conf_time_ax[0].vlines(time_for_signal+hparams['dpoints_in_future']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--', label='Signal window')
            conf_time_ax[0].vlines(time_for_signal+hparams['signal_window']/hparams['sampling_frequency']-hparams['dpoints_in_future']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--')
            

        #Else plot all signals (There may be 4 of them)
        else:
            for i, col in enumerate(signal_columns, 1):
                print('col is ', col)
                percentile = signal_df[col].quantile(1-exp_decaying(zoom_signal)/1000)
                #tretile = signal_df[col].quantile(.15)
                # Assuming you have data to plot related to 'col'
                conf_time_ax[i].plot(signal_df['time'], signal_df[col])
                conf_time_ax[i].set_ylabel(f'{col}')
                #Plot the signal window on signal figure
                conf_time_ax[i].vlines(time_for_signal+hparams['dpoints_in_future']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--', label='Signal window')
                conf_time_ax[i].vlines(time_for_signal-(hparams['signal_window']/hparams['sampling_frequency']-hparams['dpoints_in_future']/hparams['sampling_frequency']), 0, percentile, color='green', linestyle='--')
                conf_time_ax[i].set_ylim(-percentile, percentile)
                conf_time_ax[i].vlines(time_for_signal, 0, percentile, color='black', linestyle='--')
                if col=='h_alpha':
                    print('col is h_alpha, percentile is', percentile)
                    conf_time_ax[i].set_ylim(0.1*percentile, 3*percentile)
                else:
                    conf_time_ax[i].set_ylim(-percentile, percentile)
            #Plot the signal window on main figure
            conf_time_ax[0].vlines(time_for_signal+hparams['dpoints_in_future']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--', label='Signal window')
            conf_time_ax[0].vlines(time_for_signal-(hparams['signal_window']/hparams['sampling_frequency']-hparams['dpoints_in_future']/hparams['sampling_frequency']), 0, 1, color='green', linestyle='--')
            conf_time_ax[-1].set_xlabel('Time [ms]')    
        
        conf_time_ax[0].set_ylabel('Confidence')
        conf_time_ax[0].legend()
        plt.close(conf_time_fig)
        #display(conf_time_fig)
    plt.close(conf_time_fig)
    display(conf_time_fig)
    
    try:
        print(metrics_per_shot)
        print(metrics_per_shot[metrics_per_shot['kappa']!=0].describe()[['f1','precision','recall', 'kappa']])
    except:
        print('Metrics per shot not available')

    
    

def exp_decaying(x):
    return 1000*np.exp(-0.06908*x)

def shot_not_present(shot=False, color='yellow'):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Drawing a sad face
    # Face outline
    face = Circle((0.5, 0.5), 0.4, color=color, fill=True)
    ax.add_patch(face)

    # Eyes
    left_eye = Circle((0.35, 0.65), 0.05, color='black', fill=True)
    right_eye = Circle((0.65, 0.65), 0.05, color='black', fill=True)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    # Sad mouth
    mouth = Arc((0.5, 0.45), 0.2, 0.2, angle=0, theta1=0, theta2=180, color='black', linewidth=2)
    ax.add_patch(mouth)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Remove axes
    ax.axis('off')

    # Set title
    if not shot:
        ax.set_title('No results available for this run.')
    else:
        ax.set_title(f'Shot {shot} not present in results for this run.')

    # Show the plot
    return fig


def closest_decimal_time(img_time):
    # Allowed decimal parts
    decimal_parts = [0, 0.2, 0.4, 0.6, 0.8]

    # Extract the integer part
    integer_part = int(img_time)

    # Find the closest decimal part
    closest_decimal = min(decimal_parts, key=lambda x: abs(x - (img_time - integer_part)))

    # Combine the integer and closest decimal part
    closest_time = integer_part + closest_decimal
    return f'{closest_time:.1f}', closest_time

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


def extract_float(tensor_string):
    try:
        # Try to directly convert the string to a float
        return float(tensor_string)
    except ValueError:
        # If direct conversion fails, search for the 'tensor(...)' pattern
        match = re.search(r'tensor\(([^)]+)\)', tensor_string)
        if match:
            # Convert the extracted string to a float
            return float(match.group(1))
        else:
            raise ValueError("Input must be a plain number or in the format 'tensor(...)'.")
        