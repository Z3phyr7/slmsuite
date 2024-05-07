import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple
from datetime import datetime

def crop(image, ROI):
    """
    Crop an image to a region of interest (ROI).

    Parameters
    ----------
    image : np.ndarray
        2D array of intensity values.
    ROI : tuple
        Region of interest (x, y, width, height).

    Returns
    -------
    np.ndarray
        Cropped image.
    """

    x, width, y, height = ROI
    width = int(width/2)
    height = int(height/2)
    return image[y-height:y+height, x-width:x+width]

def peak_detection(intensity_matrix: np.ndarray, min_distance:int=10, threshold_abs:float=0) -> np.ndarray:
    """
    Detect peaks in an intensity matrix using Gaussian smoothing and peak finding.

    Parameters
    ----------
    intensity_matrix : np.ndarray
        2D array of intensity values.
    min_distance : int, optional
        Minimum number of pixels separating peaks, by default 1.
    threshold_abs : float, optional 
        Minimum intensity of peaks, by default 0.

    Returns
    -------
    np.ndarray
        Array of peak coordinates.
    """
    peaks = peak_local_max(intensity_matrix, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=2)
    return peaks

def gaussian_2d(yx:tuple, y0:float, x0:float, A:float, sigma_x:float, sigma_y:float) -> float:
    """
    2D Gaussian function.
    """
    y,x = yx
    return A * np.exp(-(x-x0)**2/(2*sigma_x**2) - (y-y0)**2/(2*sigma_y**2))

def fit_gaussian_2d(intensity_matrix:np.ndarray, peaks_coords:list, ROI_radius:int=60) -> list:
    """
    Fit 2D Gaussian function to intensity matrix around peak coordinates.

    Parameters
    ----------
    intensity_matrix : np.ndarray
        2D array of intensity values.
    peaks_coords : list
        List of peak coordinates.
    ROI_radius : int, optional
        Radius of region of interest around peak center, by default 50.

    Returns
    -------
    list
        List of Gaussian parameters for each peak.
    """
    gaussian_params = []
    errors = []
    for y0, x0 in peaks_coords:
        # Define ROI around the peak center
        x_roi = np.arange(x0 - ROI_radius, x0 + ROI_radius + 1)
        y_roi = np.arange(y0 - ROI_radius, y0 + ROI_radius + 1)
        x_roi, y_roi = np.meshgrid(x_roi, y_roi)
        z_roi = intensity_matrix[y_roi, x_roi]  # Note: reverse indexing for numpy arrays
        
        # Initial guess for Gaussian parameters
        initial_guess = (y0, x0, z_roi.max(), ROI_radius/2, ROI_radius/2)  # (x0, y0, amplitude, sigma_x, sigma_y)
        bound = ([y0 - ROI_radius, x0 - ROI_radius, 0, 0, 0],[y0 + ROI_radius, x0 + ROI_radius, np.max(intensity_matrix) + 1, ROI_radius, ROI_radius])
        
        # Fit Gaussian function to ROI
        popt, pcov = curve_fit(gaussian_2d, (y_roi.flatten(), x_roi.flatten()), z_roi.flatten(), p0=initial_guess, bounds=bound)
        gaussian_params.append(popt)
        errors.append(np.sqrt(np.diag(pcov)))
    return gaussian_params, errors

def write_gaussian_params_to_file(gaussian_params:list, errors:list, peaks:list, filename:str) -> None:
    """
    Write Gaussian parameters to a text file.

    Parameters
    ----------
    gaussian_params : list
        List of Gaussian parameters for each peak.
    filename : str
        Name of the output file.
    """
    with open(filename, 'a') as f:
        for peak, params, error in zip(peaks, gaussian_params, errors):
            current_time = datetime.now().strftime("%H/%M/%S")
            f.write(f'x_peak: {peak[1]}, y_peak: {peak[0]}, x0: {params[1]}, y0: {params[0]}, A: {params[2]}, sigma_x: {params[3]}, sigma_y: {params[4]}, x0_err: {error[1]}, y0_err: {error[0]}, A_err: {error[2]}, sigma_x_err: {error[3]}, sigma_y_err: {error[4]}, time: {current_time}\n')

def read_gaussian_params_from_file(filename:str) -> Tuple[List[Dict[str, float]], Dict]:
    """
    Read Gaussian parameters from a text file.

    Parameters
    ----------
    filename : str
        Name of the input file.

    Returns
    -------
    list
        List of dictionaries of Gaussian parameters for each peak.
    """
    gaussian_params = []
    with open(filename, 'r') as f:
        first_line = f.readline().strip().split(', ')
        classification_dict = {}
        for element in first_line:
            key, value = element.split(': ')
            if key == 'axis':
                classification_dict[key] = value
            elif key == 'radiant' or key == 'black' or key == 'polaris' or key == 'time':
                classification_dict[key] = float(value)
            else:
                ValueError("The classification is not valid")

        for line in f:
            params = line.strip().split(', ')
            parsed_params = {}
            for param in params:
                key, value = param.split(': ')
                if '/' in value:
                    # Handle time data in HH.MM.SS format
                    hours, minutes, seconds = map(int, value.split('/'))
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    parsed_params[key] = total_seconds
                else:
                    # Convert other values to floats
                    parsed_params[key] = float(value)
            gaussian_params.append(parsed_params)
    return gaussian_params, classification_dict

def sort_spot(gaussian_params: List[Dict], classification_dict: Dict, spot1_name:str, spot2_name:str, vibration_range = 50) -> Tuple[List[Dict]]:
    """
    Sort the spots to two classes based on their x or y coordinate.
    
    Parameters
    ----------
    gaussian_params : 
        List[Dict]
            List of dictionaries of Gaussian parameters for each peak.
        x1 : float
            x or ycoordinate of the first class.
        x2 : float
            x or y coordinate of the second class.
        by_x : bool, optional
            Sort by x coordinate if True, by y coordinate if False, by default True.
        vibration_range : int, optional
            Range of vibration, by default 50.

    Returns
    -------
    Tuple[List[Dict]]
        Two lists of dictionaries of Gaussian parameters for each peak.
    """
    x1 = classification_dict[spot1_name]
    x2 = classification_dict[spot2_name]
    by_x = classification_dict['axis'] == 'x'

    spot1 = []
    spot2 = []
    if by_x:
        for dictionary in gaussian_params:
            if dictionary['x0']>= (x1 - vibration_range) and dictionary['x0']<= (x1 + vibration_range):
                spot1.append(dictionary)
            elif dictionary['x0']>= (x2 - vibration_range) and dictionary['x0'] <= (x2 +vibration_range):
                spot2.append(dictionary)
            else:
                raise ValueError("The spot cannot be sorted to any class, the coordinate is {}".format(dictionary['x0']))
    else:
        for dictionary in gaussian_params:
            if dictionary['y0']>= (x1 - vibration_range) and dictionary['y0']<= (x1 + vibration_range):
                spot1.append(dictionary)
            elif dictionary['y0']>= (x2 - vibration_range) and dictionary['y0'] <= (x2 +vibration_range):
                spot2.append(dictionary)
            else:
                raise ValueError("The spot cannot be sorted to any class")
    return spot1,spot2

def consecutive_anlys_1st(spot_list:List[Dict]) -> float:
    """
    Calculate the deviation of consecutive spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.
    
    Returns
    -------
    float
        Deviation of consecutive spots in the same class.
    """
    x0 = np.array([spot['x0'] for spot in spot_list])
    y0 = np.array([spot['y0'] for spot in spot_list])
    x_diff = np.diff(x0)
    y_diff = np.diff(y0)
    deviation = x_diff**2 + y_diff**2
    return deviation.mean()**0.5

def consecutive_anlys_2nd(spot_list:List[Dict]) -> float:
    """
    Calculate the deviation of 2nd consecutive spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.

    Returns
    -------
    float
        Deviation of 2nd consecutive spots in the same class.    
    """
    x0 = [spot['x0'] for spot in spot_list]
    y0 = [spot['y0'] for spot in spot_list]
    x_diff = np.array([x0[i+2] - x0[i] for i in range(len(x0)-2)])
    y_diff = np.array([y0[i+2] - y0[i] for i in range(len(y0)-2)])
    deviation = x_diff**2 + y_diff**2
    return deviation.mean()**0.5

def consecutive_anlys_3rd(spot_list:List[Dict]) -> float:
    """
    Calculate the deviation of 3rd consecutive spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.

    Returns
    -------
    float
        Deviation of 3rd consecutive spots in the same class.
    """
    x0 = [spot['x0'] for spot in spot_list]
    y0 = [spot['y0'] for spot in spot_list]
    x_diff = np.array([x0[i+3] - x0[i] for i in range(len(x0)-3)])
    y_diff = np.array([y0[i+3] - y0[i] for i in range(len(y0)-3)])
    deviation = x_diff**2 + y_diff**2
    return deviation.mean()**0.5

def spread_anlys(spot_list:List[Dict]) -> float:
    """
    Calculate the spread of spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.

    Returns
    -------
    float
        Spread of spots in the same class.
    """

    x0 = np.array([spot['x0'] for spot in spot_list])
    y0 = np.array([spot['y0'] for spot in spot_list])
    x_mean = np.mean(x0)
    y_mean = np.mean(y0)
    spread = (x0 - x_mean)**2 + (y0 - y_mean)**2
    return spread.mean()**0.5

def drift_visualisation(spot_list:List[Dict], spot_name = None) -> None:
    """
    Visualise the drift of spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.
    """
    color_dict = {'radiant':'red', 'black':'black', 'polaris':'blue'}
    x0 = np.array([spot['x0'] for spot in spot_list]) - spot_list[0]['x0']
    y0 = np.array([spot['y0'] for spot in spot_list]) - spot_list[0]['y0']
    plt.plot(x0, y0, color = color_dict[spot_name], marker='o', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    if spot_name is not None:
        plt.title(f'Drift of {spot_name} spots')
    else: pass
    plt.title('Drift of spots')
    plt.show()

def drift_visualisation_for_two_spots(spot_list_1: List[Dict], spot_list_2: List[Dict], spot1_name: str = None, spot2_name:str = None) -> None:
    color_dict = {'radiant':'red', 'black':'black', 'polaris':'blue'}
    x0_1 = np.array([spot['x0'] for spot in spot_list_1]) - spot_list_1[0]['x0']
    y0_1 = np.array([spot['y0'] for spot in spot_list_1]) - spot_list_1[0]['y0']
    x0_2 = np.array([spot['x0'] for spot in spot_list_2]) - spot_list_2[0]['x0']
    y0_2 = np.array([spot['y0'] for spot in spot_list_2]) - spot_list_2[0]['y0']
    plt.plot(x0_1, y0_1, 'o-', label=spot1_name, color = color_dict[spot1_name])
    plt.plot(x0_2, y0_2, 'o-', label=spot2_name, color = color_dict[spot2_name])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Drift of two spots')
    plt.legend()
    plt.show()


def XY_visualisation(spot_list:List[Dict], spot_name = None) -> None:
    """
    Visualise the XY coordinates of spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.
    """
    color_dict = {'radiant':'red', 'black':'black', 'polaris':'blue'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x0 = np.array([spot['x0'] for spot in spot_list])
    y0 = np.array([spot['y0'] for spot in spot_list])
    time = np.arange(len(x0)) * 3
    ax[0].plot(time, x0, 'o-', color = color_dict[spot_name])
    ax[1].plot(time, y0, 'o-', color = color_dict[spot_name])
    ax[0].set_xlabel('Time (min)')
    ax[1].set_xlabel('Time (min)')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    ax[0].title.set_text('x vs Time')
    ax[1].title.set_text('y vs Time')
    plt.show()

def XY_visualisation_error(spot_list:List[Dict], spot_name = None) -> None:
    """
    Visualise the XY coordinates of spots in the same class.

    Parameters
    ----------
    spot_list : List[Dict]
        List of dictionaries of Gaussian parameters for each peak.
    """
    color_dict = {'radiant':'red', 'black':'black', 'polaris':'blue'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x0 = np.array([spot['x0'] for spot in spot_list])
    y0 = np.array([spot['y0'] for spot in spot_list])
    x0_err = np.array([spot['x0_err'] for spot in spot_list])
    y0_err = np.array([spot['y0_err'] for spot in spot_list])
    time = np.arange(len(x0)) * 3
    ax[0].errorbar(time, x0, yerr = x0_err, fmt='o', color=color_dict[spot_name], ecolor='purple', label='x0', barsabove=True)
    ax[1].errorbar(time, y0, yerr = y0_err, fmt='o', color=color_dict[spot_name], ecolor='purple', label='y0', barsabove=True)
    ax[0].set_xlabel('Time (min)')
    ax[1].set_xlabel('Time (min)')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    ax[0].title.set_text('x vs Time')
    ax[1].title.set_text('y vs Time')
    plt.show()

def XY_visualisation_for_two_spots(spot_list_1: List[Dict], spot_list_2: List[Dict], spot1_name: str = None, spot2_name:str = None) -> None:

    color_dict = {'radiant':'red', 'black':'black', 'polaris':'blue'}

    fig,ax = plt.subplots(1,2, figsize=(10,5))

    x0_1 = np.array([spot['x0'] for spot in spot_list_1]) - spot_list_1[0]['x0']
    y0_1 = np.array([spot['y0'] for spot in spot_list_1]) - spot_list_1[0]['y0']
    x0_2 = np.array([spot['x0'] for spot in spot_list_2]) - spot_list_2[0]['x0']
    y0_2 = np.array([spot['y0'] for spot in spot_list_2]) - spot_list_2[0]['y0']
    time_1 = np.array([spot['time'] for spot in spot_list_1]) - spot_list_1[0]['time']
    time_2 = np.array([spot['time'] for spot in spot_list_2]) - spot_list_2[0]['time']

    if spot1_name is not None and spot2_name is not None:
        ax[0].plot(time_1, x0_1, 'o-', label=spot1_name, color=color_dict[spot1_name])
        ax[0].plot(time_2, x0_2, 'o-', label=spot2_name, color=color_dict[spot2_name])
        ax[1].plot(time_1, y0_1, 'o-', label=spot1_name, color=color_dict[spot1_name])
        ax[1].plot(time_2, y0_2, 'o-', label=spot2_name, color=color_dict[spot2_name])
    else:
        ax[0].plot(time_1, x0_1, 'o-', label='spot1', color='blue')
        ax[0].plot(time_2, x0_2, 'o-', label='spot2', color='red')
        ax[1].plot(time_1, y0_1, 'o-', label='spot1', color='blue')
        ax[1].plot(time_2, y0_2, 'o-', label='spot2', color='red')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    fig.suptitle('Comparison of XY drift of two spots')
    ax[0].title.set_text('x vs Time')
    ax[1].title.set_text('y vs Time')
    ax[0].legend()
    ax[1].legend()
    plt.show()
