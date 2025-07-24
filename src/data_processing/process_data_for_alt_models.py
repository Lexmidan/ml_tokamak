import os
from pathlib import Path
import sys
sys.path.append('/compass/Shared/Common/IT/projects/user-libraries/python/cdb_extras/stable/cdb_extras/')
# import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import pandas as pd
import xrscipy.signal as dsp
import numpy as np
from tqdm import tqdm

from . import imgs_processing as imgs

cdb = client.CDBClient()

path = Path('/compass/Shared/Users/bogdanov/ml_tokamak')
os.chdir(path)

def process_data_for_alt_models(shot_numbers, variant = 'seidl_2023', sampling_freq=300):
    #dirs where to save the csv files
    directories = {'h_alpha':'data/h_alpha_signal',  
                    'divlp':'data/langmuir_probe_signal'}
    
    #RobustScaler for scaling the signals
    scaler = imgs.RobustScalerNumpy().fit_transform

    shot = 0 #In order to tqdm print the description
    with tqdm(total=len(shot_numbers), desc='working on shots') as pbar:
        for shot in shot_numbers:
            pbar.set_description(f'working on shot:{shot}')

            
            #Load shot from CDBClient
            shot_from_client = cdbxr.Shot(shot)
            
            # Load signals from CDBClient
            h_alpha_signal = - shot_from_client['H_alpha']
            divlp_signal = shot_from_client['DIVLPB01']

            # Load labels from CDB
            t_ELM_start = cdb.get_signal(f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_ELM_peak = cdb.get_signal(f"t_ELM_peak/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

            #Calculate t_elm_end from t_ELM_peak and t_ELM_start (it should be symetric to t_ELM_peak)
            t_ELM_end = 2 * t_ELM_peak.data - t_ELM_start.data

            t_H_mode_start = cdb.get_signal(f"t_H_mode_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_H_mode_end = cdb.get_signal(f"t_H_mode_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

            #TODO:  To create a DataFrame with only one row, one needs to specify an index, 
            # so if plasma enters H-mode more than once during one shot index have to be passed. Thus crutch with try: except:
            try:
                len(t_ELM_start.data)
            except:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end}, index=[0])
            else:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end})

            try:
                len(t_H_mode_start.data)
            except:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data}, index=[0])
            else:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data})


            for signal, signal_name in zip([h_alpha_signal, divlp_signal], ['h_alpha', 'divlp']):
                # Decimate signal to 300 kHz
                signal = dsp.decimate(signal, target_fs=sampling_freq) 

                #Scale signal using RobustScaler
                signal = scaler(signal)

                # Create a DataFrame from the decimated signal xarray
                signal_df = pd.DataFrame({'time':signal.time.values, signal_name:signal.data})
                signal_df = signal_df.set_index('time')

                # Remove data with no plasma
                discharge_start, discharge_end = imgs.discharge_duration(shot, 4e4)
                signal_df = signal_df[np.logical_and(signal_df.index>discharge_start, signal_df.index<discharge_end)]

                # Create a column with mode labels. These are all L-mode by default.
                signal_df['mode'] = 'L-mode'

                for H_mode in t_H_mode.values:
                    signal_df.loc[H_mode[0]:H_mode[1], 'mode'] = 'H-mode'

                for elm in t_ELM.values:
                    signal_df.loc[elm[0]:elm[1], 'mode'] = 'ELM'

                if len(t_ELM_peak.data.shape)!=0:
                    for elm in t_ELM_peak.data:
                        closest_idx = signal_df.index.get_indexer([elm], method='nearest')[0]
                        # Update the 'mode' column for the closest index
                        signal_df.iloc[closest_idx, signal_df.columns.get_loc('mode')] = 'ELM-peak'
                else:
                    closest_idx = signal_df.index.get_indexer([float(t_ELM_peak.data)], method='nearest')[0]
                    # Update the 'mode' column for the closest index
                    signal_df.iloc[closest_idx, signal_df.columns.get_loc('mode')] = 'ELM-peak'

                # Save data
                signal_df.to_csv(f'{directories[signal_name]}_{sampling_freq}kHz/shot_{shot}.csv')
            pbar.update(1)


def process_data_for_multiple_mirnov_coils(shot_numbers, variant = 'seidl_2023', sampling_freq=300):
    
    #RobustScaler for scaling the signals
    scaler = imgs.RobustScalerNumpy().fit_transform
    shot = 0 #In order to tqdm print the description
    with tqdm(total=len(shot_numbers), desc='working on shots') as pbar:
        for shot in shot_numbers:
            pbar.set_description(f'working on shot:{shot}')

            #Load shot from CDBClient
            shot_from_client = cdbxr.Shot(shot)
            
            # Load signals from CDBClient
            mcHFS_signal = shot_from_client['Mirnov_coil_A_theta_13_RAW']
            mcLFS_signal = shot_from_client['Mirnov_coil_A_theta_02_RAW']
            mcDIV_signal = shot_from_client['Mirnov_coil_A_theta_19_RAW']
            mcTOP_signal = shot_from_client['Mirnov_coil_A_theta_07_RAW']

            # Load labels from CDB
            t_ELM_start = cdb.get_signal(f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_ELM_peak = cdb.get_signal(f"t_ELM_peak/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

            #Calculate t_elm_end from t_ELM_peak and t_ELM_start (it should be symetric to t_ELM_peak)
            t_ELM_end = 2 * t_ELM_peak.data - t_ELM_start.data

            t_H_mode_start = cdb.get_signal(f"t_H_mode_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_H_mode_end = cdb.get_signal(f"t_H_mode_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

            #TODO:  To create a DataFrame with only one row, one needs to specify an index, 
            # so if plasma enters H-mode more than once during one shot index have to be passed. Thus crutch with try: except:
            try:
                len(t_ELM_start.data)
            except:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end}, index=[0])
            else:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end})

            try:
                len(t_H_mode_start.data)
            except:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data}, index=[0])
            else:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data})

            # Decimate signal to 300 kHz and scale it using RobustScaler
            mcHFS_signal, mcLFS_signal, mcDIV_signal, mcTOP_signal = map(lambda signal: scaler(dsp.decimate(signal, target_fs=sampling_freq)),
                                                                [mcHFS_signal, mcLFS_signal, mcDIV_signal, mcTOP_signal])

            # Create a DataFrame from the decimated signal xarray
            signal_df = pd.DataFrame({'time': mcHFS_signal.time.values, 
                                    'mcHFS': mcHFS_signal.data, 
                                    'mcLFS': mcLFS_signal.data, 
                                    'mcDIV': mcDIV_signal.data, 
                                    'mcTOP': mcTOP_signal.data})
            
            signal_df = signal_df.set_index('time')

            # Remove data with no plasma
            discharge_start, discharge_end = imgs.discharge_duration(shot, 4e4)
            signal_df = signal_df[np.logical_and(signal_df.index>discharge_start, signal_df.index<discharge_end)]

            # Create a column with mode labels. These are all L-mode by default.
            signal_df['mode'] = 'L-mode'

            for H_mode in t_H_mode.values:
                signal_df.loc[H_mode[0]:H_mode[1], 'mode'] = 'H-mode'

            for elm in t_ELM.values:
                signal_df.loc[elm[0]:elm[1], 'mode'] = 'ELM'

            if len(t_ELM_peak.data.shape)!=0:
                for elm in t_ELM_peak.data:
                    closest_idx = signal_df.index.get_indexer([elm], method='nearest')[0]
                    # Update the 'mode' column for the closest index
                    signal_df.iloc[closest_idx, signal_df.columns.get_loc('mode')] = 'ELM-peak'
            else:
                closest_idx = signal_df.index.get_indexer([float(t_ELM_peak.data)], method='nearest')[0]
                # Update the 'mode' column for the closest index
                signal_df.iloc[closest_idx, signal_df.columns.get_loc('mode')] = 'ELM-peak'

            # Save data
            signal_df.to_csv(f'data/mirnov_coil_signal_{sampling_freq}kHz/shot_{shot}.csv')
            pbar.update(1)

if __name__ == "__main__":
    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_alt = shot_usage[shot_usage['used_for_alt']]
    shot_numbers = shot_for_alt['shot']
    
    process_data_for_alt_models(shot_numbers, variant='seidl_2023', sampling_freq=300)

    process_data_for_alt_models(shot_numbers, variant='seidl_2023', sampling_freq=150)
    
    process_data_for_alt_models(shot_numbers, variant='seidl_2023', sampling_freq=30)
    