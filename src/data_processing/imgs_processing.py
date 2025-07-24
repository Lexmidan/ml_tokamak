import os
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import xarray as xr     
import numpy as np         # práce s numerickými poli
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple  
from tqdm import tqdm
cdb = client.CDBClient()
os.chdir("/compass/Shared/Users/bogdanov/ml_tokamak")

def load_RIS_data(shot: int, ris: int) -> xr.DataArray:
    """Load RAW camera data from database

    Args:
        shot: Shot number.
        ris: Which camrea to load? 1 or 2.

    Returns:
        Xarray DataArray of shape (time, 2*nx, 2*ny).

    Raises:
        KeyError: When the data is missing for the shot.
        CDBException: Thrown by the database client.
        ValueError: When the camera number is incorrect.
    """
    if ris not in [1, 2]:
        raise ValueError(f'RIS camera nr. {ris} does not exist')
    s = cdbxr.Shot(shot, cache=False)

    return s[f'RIS.RISEye_{ris}.RAW']


def demosaic(data: xr.DataArray) -> np.array:
    """Demosaic RAW camera data

    Transforms RAW data of shape (time, 2*nx, 2*ny) to (time, nx, ny, 3) and normalizes them into range <0, 1>.

    Args:
        data: RAW camera data

    Returns:
        Data in RGB format.
    """

    r = data[:,::2,1::2]
    b = data[:,1::2,::2]
    g1 = data[:,1::2,1::2]
    g2 = data[:,::2,::2]
    g = xr.DataArray((g1.data+g2.data)/2)

    xar = xr.DataArray([r,g,b], dims=('color', 'time', 'x', 'y'), 
                       coords={'color': ['r','g','b'], 'time': data.coords['time'],
                       'x': np.arange(r.shape[1]), 'y':np.arange(r.shape[2])})
    axr = xar.transpose('time', 'x', 'y', 'color')

    # kazdy pixel ma 12 bitu (hodnoty 0..4095) => normalizujeme na rozsah 0..1
    axr = axr/(2**12-1)
    
    return axr


def flip_image(data: xr.DataArray, flip_horizontal: bool = True, flip_vertical: bool = True) -> np.array:
    """Flip image data vertically and horizontally.

    Args:
        data: Image data with shape (time, nx, ny, 3)
        flip_horizontal: Flip horizontally?
        flip_vertical: Flip vertically?

    Returns:
        Flipped data with the same shape.
    """

    if flip_horizontal:
        data = data[:,:,::-1]
    if flip_vertical:
        data = data[:,::-1,:]
    return data

def save_frame(path: Path, frame: xr.DataArray, ris: int, 
               shot: int, time: float, just_names: bool = False) -> Path:
    """Save single frame to image file.

    Output file name is f"{path}/RIS{ris}_{shot}_t={time}.png"

    Args:
        path: Where to save the image.
        frame: Array with data, shape (nx, ny, 3), values in the range <0, 1>.
        ris: Number of RIS camera, will be saved as a part of image file name.
        shot: Shot, will be saved as a part of image file name.
        time: Time in [ms]. Will be saved as a part of image file name
        just_names: Bool, function will only return filenames if True. Saves time if images are already saved

    Returns:
        Path to created file name.
    """

    filename = path.joinpath(f"RIS{ris}_{shot}_t={time:.1f}.png") 

    if not(just_names):
        img = Image.fromarray((frame.data*255).astype('uint8')).convert('RGB')
        img.save(filename, format=None)
    
    return filename

def discharge_duration(shot: int, threshold: float = 1e4) -> Tuple[float, float]:
    """Find discharge duration

    Args:
        shot: Queried shot.
        threshold: I_plasma must be higher than this threshold to say there is a plasma.

    Returns:
        Tuple[plasma_start, plasma_end]
    """

    s = cdbxr.Shot(shot)
    ipla = s["I_plasma"]
    plasma_time = ipla[abs(ipla)>threshold].time.data
    start = plasma_time[0]
    end = plasma_time[-1]

    return  start, end

def save_ris_images_to_folder(data: int, shot, path: Path, ris: int, 
                              use_discharge_duration: bool=True, just_names: bool = False):
    """Save all images from RIS camera in a given shot to a given folder.

    Saves only frames from times when there was a plasma.

    Args:
        shot: Shot.
        path: Output path. The image files will be saved to subfolder path / {shot}. The subfolder
            will be created if it does not exist.
        ris: Which camera to use? 1 or 2.
        use_discharge_duration: save only images with plasma on it
        just_names: function will only return filenames if True. Saves time.
    Returns:
        filenames: np.array with all the names of saved images
    """
    filenames = [] #str array with all the names of saved images
    print('Demosaicing images')
    if use_discharge_duration:
        start, end = discharge_duration(shot, 1e4)
        dem_data = flip_image(demosaic(data).sel(time=slice(start, end)))
    else:
        dem_data = flip_image(demosaic(data))

    for frame in tqdm(dem_data, total=len(dem_data), desc='Saving images'):
        filename = save_frame(path=path, frame=frame, ris=ris, shot=shot, 
                              time=frame.time.data, just_names=just_names)
        filenames.append(filename)

    return filenames



def process_shots(shots: list, use_discharge_duration: bool=True, just_names: bool=False,
                  variant: str = 'seidl_2023', save_ris2: bool=True):
    '''
    Processes multiple shots in a row.

    Args:
    use_discharge_duration: save only images with plasma on it
    just_names: function will only return filenames if True. Saves time.
    variant: which variant of diagnostics to use.
    '''
    for shot in tqdm(shots):
        try:
            print('Working on shot ', shot)

            out_path = Path(f'./imgs/{shot}')
            if not os.path.exists(out_path):
                os.mkdir(out_path, mode=0o777)

            ris1_data = load_RIS_data(shot, 1)
            

            ris1_names = save_ris_images_to_folder(ris1_data, path=out_path, ris=1, shot=shot, 
                                                use_discharge_duration=use_discharge_duration, 
                                                just_names=just_names)
            if save_ris2:
                ris2_data = load_RIS_data(shot, 2)
                ris2_names = save_ris_images_to_folder(ris2_data, path=out_path, ris=2, shot=shot, 
                                                    use_discharge_duration=use_discharge_duration, 
                                                    just_names=just_names)

            #contains time and the state of the plasma (L-mode, H-mode, ELM)
            LorH = pd.DataFrame(data={'mode':np.full(len(ris1_data), 'L-mode')}, 
                                index=pd.Index(ris1_data.time, name='time'))
            
            t_ELM_start = cdb.get_signal(f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_ELM_end = cdb.get_signal(f"t_ELM_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_H_mode_start = cdb.get_signal(f"t_H_mode_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
            t_H_mode_end = cdb.get_signal(f"t_H_mode_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

            #TODO:  To create a DataFrame with only one row, one needs to specify an index, 
            # so if plasma enters H-mode more than once during one shot index have to be passed. Thus crutch with try: except:
            try:
                len(t_ELM_start.data)
            except:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end.data}, index=[0])
            else:
                t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end.data})

            try:
                len(t_H_mode_start.data)
            except:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data}, index=[0])
            else:
                t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data})


            for H_mode in t_H_mode.values:
                LorH.loc[H_mode[0]:H_mode[1]] = 'H-mode'

            for elm in t_ELM.values:
                LorH.loc[elm[0]:elm[1]] = 'ELM'

            #Discarding pictures without plasma
            discharge_start, discharge_end = discharge_duration(shot)
            LorH = LorH[discharge_start : discharge_end]

            #Appending columns with paths of the RIS imgs
            LorH['filename'] = np.array(ris1_names)
            LorH.to_csv(f'/compass/Shared/Users/bogdanov/ml_tokamak/data/LH_alpha/LH_alpha_shot_{shot}.csv')
            print(f'csv saved to ./LH_alpha_shot_{shot}.csv')
            
        except Exception as e:
            print(f'Error processing shot {shot}: {e}')
        

def add_average_halpha_column(lh_mode_df, shot):
    """
    Adds a column 'current' to the lh_mode_df dataframe, containing the average current 
    from h_alpha_df for each time step in lh_mode_df.
    
    :param lh_mode_df: DataFrame based on LH_alpha_shot_{shot}.csv
    :param shot: shot from which data for h_alpha diagnostics will be retained
    :return: DataFrame with added 'current' column
    """
    h_alpha_signal = cdb.get_signal(f"H_alpha/SPECTROMETRY_RAW:{shot}")
    h_alpha_df = pd.DataFrame({'time':h_alpha_signal.time_axis.data, 'h_alpha':h_alpha_signal.data}) 
    h_alpha_df = h_alpha_df[np.logical_and(h_alpha_df['time'] > lh_mode_df.index[0] - 1, 
                                           h_alpha_df['time'] < lh_mode_df.index[-1] + 1)]

    def calculate_average_current_for_time_range(start_time, end_time, current_data):
        """Calculate the average current for a given time range."""
        relevant_data = current_data[(current_data['time'] >= start_time) & (current_data['time'] < end_time)]
        if not relevant_data.empty:
            return relevant_data['h_alpha'].mean()
        else:
            return None

    # Calculate the time step in lh_mode_df
    time_step = lh_mode_df.index[11] - lh_mode_df.index[10]

    # Calculate the average current for each time step in lh_mode_df
    average_currents = []
    for time_point in lh_mode_df.index:
        start_time = time_point
        end_time = start_time + time_step
        average_current = calculate_average_current_for_time_range(start_time, end_time, h_alpha_df)
        average_currents.append(-average_current)

    # Add the new column to the lh_mode_df dataframe
    lh_mode_df['h_alpha'] = average_currents

    return lh_mode_df

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




if __name__ == "__main__":
    shots_str = input('input shots numbers separated with commas')
    shots = np.array([int(x) for x in shots_str.split(',')])
    process_shots(shots)
    