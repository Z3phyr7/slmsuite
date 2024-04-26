import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple

def crop(image, ROI):
    x, width, y, height = ROI
    width = int(width/2)
    height = int(height/2)
    return image[y-height:y+height, x-width:x+width]

def peak_detection(intensity_matrix: np.ndarray, min_distance:int=1, threshold_abs:float=0) -> np.ndarray:
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

def fit_gaussian_2d(intensity_matrix:np.ndarray, peaks_coords:list, ROI_radius:int=50) -> list:
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
    return gaussian_params

def write_gaussian_params_to_file(gaussian_params:list, peaks:list, filename:str) -> None:
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
        for peak, params in zip(peaks, gaussian_params):
            f.write(f'x_peak: {peak[1]}, y_peak: {peak[0]}, x0: {params[1]}, y0: {params[0]}, A: {params[2]}, sigma_x: {params[3]}, sigma_y: {params[4]}\n')

def read_gaussian_params_from_file(filename:str) -> List[Dict[str, float]]:
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
        for line in f:
            params = line.strip().split(', ')
            params = {param.split(': ')[0] : float(param.split(': ')[1]) for param in params}
            gaussian_params.append(params)
    return gaussian_params

def sort_spot(gaussian_params: List[Dict], x1, x2, vibration_range = 50) -> Tuple[List[Dict]]:
    spot1 = []
    spot2 = []
    for dictionary in gaussian_params:
        if dictionary['x0']>= (x1 - vibration_range) and dictionary['x0']<= (x1 + vibration_range):
            spot1.append(dictionary)
        elif dictionary['x0']>= (x2 - vibration_range) and dictionary['x0'] <= (x2 +vibration_range):
            spot2.append(dictionary)
        else:
            raise ValueError("The spot cannot be sorted to any class")
    return spot1,spot2

def deviation_anlys(spot_list:List[Dict]) -> List:
    deviation_list =[]
    previous = spot_list[0]
    for spot in spot_list[1:]:
        deviation =  ((previous['x0']-spot['x0'])**2 + (previous['y0']-spot['y0'])**2)**0.5
        deviation_list.append(deviation)
        previous = spot
    return deviation_list
