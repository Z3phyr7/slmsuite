import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from typing import List, Dict


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

def gaussian_2d(xy:tuple, x0:float, y0:float, A:float, sigma_x:float, sigma_y:float) -> float:
    """
    2D Gaussian function.
    """
    x,y = xy
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
    for x0, y0 in peaks_coords:
        # Define ROI around the peak center
        x_roi = np.arange(x0 - ROI_radius, x0 + ROI_radius + 1)
        y_roi = np.arange(y0 - ROI_radius, y0 + ROI_radius + 1)
        x_roi, y_roi = np.meshgrid(x_roi, y_roi)
        z_roi = intensity_matrix[y_roi, x_roi]  # Note: reverse indexing for numpy arrays
        
        # Initial guess for Gaussian parameters
        initial_guess = (x0, y0, z_roi.max(), 1, 1)  # (x0, y0, amplitude, sigma_x, sigma_y)
        
        # Fit Gaussian function to ROI
        popt, pcov = curve_fit(gaussian_2d, (x_roi.flatten(), y_roi.flatten()), z_roi.flatten(), p0=initial_guess)
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
            f.write(f'x_peak: {peak[0]}, y_peak: {peak[1]}, x0: {params[0]}, y0: {params[1]}, A: {params[2]}, sigma_x: {params[3]}, sigma_y: {params[4]}\n')

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