"""CPU excerpt of official SC code; arithmetic bodies unchanged.

Source: gollischlab/SpatiotemporalSCModel
commit 76b733421cc16131c2229e66a7714d8892de39d7
convolutions.py:108-245 and nonlinearities.py.
Only imports/loader annotations are narrowed to the official NumPy fallback.
MAX_FLOAT_SIZE=500 is from the same commit's project_variables.py.
Copyright (c) 2025 Gollisch Lab. MIT: spatial_contrast_LICENSE.txt.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy as cp
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

_HAVE_CUPY = False
MAX_FLOAT_SIZE = 500


def convolve_stimulus_with_kernels_for_sc(
    stimulus: np.ndarray,
    spatial_filter: np.ndarray,
    temporal_filter: np.ndarray,
    total_trials: int,
    stimulus_smoothing: Optional[float] = None,
):
    """
    Convolve a given stimulus with spatial and temporal kernels for the SC model.

    This function performs a convolution operation on the input stimulus using the specified
    spatial and temporal kernels. The convolution is performed in chunks to optimize memory usage
    and speed. The function also applies a Gaussian filter to the stimulus if specified. It uses
    CuPy for GPU-accelerated computation if available, otherwise falls back to NumPy.

    :param stimulus: The input stimulus to be convolved. It can be either a 3D array with dimensions
        (trials, time, space) or the WhitenoiseLoader or NaturalMovieLoader object.
    :type stimulus: np.ndarray, WhitenoiseLoader, NaturalMovieLoader
    :param spatial_filter: The spatial filter to convolve the stimulus with. It should be a 2D array.
    :type spatial_filter: np.ndarray
    :param temporal_filter: The temporal filter to convolve the stimulus with. It should be a 1D array.
    :type temporal_filter: np.ndarray
    :param total_trials: The number of trials in the stimulus to process. If specified, only the first
        `total_trials` trials will be processed. If None, all trials will be processed.
    :type total_trials: int
    :param stimulus_smoothing: The standard deviation for Gaussian kernel. The Gaussian kernel is used
        for smoothing the stimulus. If None, no smoothing is applied. Default is None.
    :type stimulus_smoothing: float, optional

    :return: A tuple containing two arrays:
        - The mean luminosity signal (I_mean) for each trial.
        - The local spatial contrast signal (LSC) for each trial.
        Each array has dimensions (trials, time).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Initialize empty arrays to store the convolved responses
    convolved_response_i_mean = np.zeros(
        (total_trials, stimulus.shape[1] - temporal_filter.size + 1)
    )
    convolved_response_lsc = np.zeros_like(convolved_response_i_mean)

    # Get the spatial crop size
    spatial_crop = spatial_filter.shape[0] // 2

    # Get the non-zero pixels in the spatial kernel
    non_zero_pixels = spatial_filter[spatial_filter != 0.0]
    # Calculate the sum of non-zero pixels
    non_zero_pixels_sum = non_zero_pixels.sum()

    # Convert the spatial and temporal kernels into CuPy arrays for GPU-accelerated computation
    spatial_filter = cp.asarray(spatial_filter.flatten())
    temporal_filter = cp.asarray(temporal_filter)
    k = temporal_filter.size

    # maximum size of the data chunk that can be processed at once based on the available GPU memory
    # value is arbitrary and is defined in `sc_model.utils.project_variables.py` and can be decreased there
    # in case the user runs into memory issues on the GPU or increased to speed up the model
    # current value works well for a GPU with 8GB of memory and still leaves room for other applications in VRAM
    max_float_size = MAX_FLOAT_SIZE

    # Loop over each trial in the stimulus
    for tr, trial in tqdm(enumerate(stimulus[:total_trials]), total=total_trials, desc="Trial", leave=False):
        # Calculate the size of each data chunk based on the available GPU memory
        frame_nbytes = trial[0].nbytes / (1024 ** 2)  # in megabytes
        chunk_size = int(np.floor(max_float_size / frame_nbytes))
        chunks_needed = int(np.ceil(trial.shape[0] / chunk_size))

        chunked_imean = []
        chunked_lsc = []
        # Loop over each chunk in the trial
        for ch in tqdm(range(chunks_needed), desc="Chunk", leave=False):
            # Calculate the start and end indices of the chunk
            chunk_start = (ch * chunk_size) - k + 1
            # Ensure that the start index is not negative
            if chunk_start < 0:
                chunk_start = 0
            # Calculate the end index of the chunk
            chunk_end = (ch + 1) * chunk_size

            chunk = trial[chunk_start:chunk_end]

            # reshape chunk to (Time, X*Y)
            chunk = cp.asarray(chunk.reshape(
                (chunk.shape[0], chunk.shape[1] * chunk.shape[2])
            ))
            # calculate temporal convolution
            temp_conv = convolve2d(
                chunk, cp.expand_dims(temporal_filter, axis=-1), mode="valid"
            )
            # apply Gaussian filter to the temporal convolution
            if stimulus_smoothing is not None:
                filt_temp_conv = gaussian_filter(
                    temp_conv.reshape((
                        temp_conv.shape[0], spatial_crop * 2, spatial_crop * 2
                    )),
                    sigma=(0, stimulus_smoothing, stimulus_smoothing),
                    truncate=3.0,
                ).reshape(temp_conv.shape)
            else:
                # if no smoothing is applied, use the original temporal convolution
                filt_temp_conv = temp_conv.copy()

            # calculate the I_mean and LSC
            # according to Liu and Gollisch, Natural Image Coding
            imean = (spatial_filter * temp_conv).sum(axis=-1) / non_zero_pixels_sum
            # inner_sum = spatial_kernel * temp_conv
            # spat_conv = inner_sum.mean(axis=-1)
            # spat_conv = inner_sum / spatial_kernel.size  # I_mean
            lcl_sptl_cntrst = cp.sqrt(
                (
                        spatial_filter * (filt_temp_conv - cp.expand_dims(imean, 1)) ** 2
                ).sum(axis=-1) / non_zero_pixels_sum
            )  # LSC

            chunked_imean.append(_get(imean))
            chunked_lsc.append(_get(lcl_sptl_cntrst))
        convolved_response_i_mean[tr] = np.concatenate(chunked_imean)
        convolved_response_lsc[tr] = np.concatenate(chunked_lsc)

    return convolved_response_i_mean, convolved_response_lsc


def _get(arr: cp.ndarray) -> np.ndarray:
    if _HAVE_CUPY:
        return arr.get()
    else:
        return arr


def vectorized_softplus(
    x: np.ndarray,
    params: Union[list, np.ndarray],
):
    a = params[0]
    b = params[1]
    w = np.array([params[2:]])
    return a * np.log(1. + np.exp(w @ x + b))[0]


def vectorized_softplus_derivative(
    x: np.ndarray,
    params: Union[list, np.ndarray],
):
    a = params[0]
    b = params[1]
    w = np.array([params[2:]])

    inner_exp = np.exp(w @ x + b)
    inner_exp_p1 = 1. + inner_exp
    inner_log = np.log(inner_exp_p1)

    der_a = inner_log

    der_b = a * inner_exp / inner_exp_p1

    der_w = der_b * x

    return np.vstack([der_a, der_b, der_w]).T
