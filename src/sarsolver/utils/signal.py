import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.signal.windows import hann, tukey


def get_downsample_kernel(downsample_ratio: float) -> NDArray:
    """
    Function which returns a spectral kernel useful for downsampling complex data.
    In this case we use a five-point stencil with a Hann weighting.
    Parameters
    ----------
    downsample_ratio : float
        Ratio by which we will downsample. Note this will be rounded.

    Returns
    -------
    impulse_response : NDArray
        The spectral kernel we want for downsampling, in the normal FFT ordering.
    """
    num_samples = int(np.round(5 * downsample_ratio))
    freq_filter = np.zeros(num_samples, dtype="complex64")
    freq_filter[:2] = 1.0
    freq_filter[-1:] = 1.0
    impulse_response = np.fft.fftshift(np.fft.ifft(freq_filter)) * hann(num_samples)
    return impulse_response


def to_db_abs(x: NDArray) -> NDArray:
    """
    Extremely handy little function which one gets fed up with redefining constantly.
    Parameters
    ----------
    x : NDArray
        (Complex) data array to convert to dBs.

    Returns
    -------
    NDArray
        Real-valued data array in dBs.
    """
    return np.real(10.0 * np.log10(np.abs(x) + 1.0E-30))


def fast_freq_downsample(phase_history: NDArray, new_num_freqs: int, safety_factor: float = 1.2) -> NDArray:
    """
    Function which produces a copy of a phase history which has been reweighted and rephased to downsample in fast
    frequency (or clip in fast time).
    Parameters
    ----------
    phase_history : NDArray
        Phase history we want to resample.
    new_num_freqs : int
        Number of fast frequency samples we want to end up with.
    safety_factor : float
        Degree by which we want to oversample in order to guarantee Nyquist requirements are met. Defaults to 1.2.

    Returns
    -------
    new_phase_history : NDArray
        Resampled copy of the original phase history.
    """
    if safety_factor < 1.0:
        safety_factor = 1.0
    window = np.expand_dims(np.fft.fftshift(tukey(new_num_freqs, alpha=(safety_factor - 1.0))), axis=0)
    new_range_profiles = np.zeros([phase_history.shape[0], new_num_freqs], dtype=phase_history.dtype)
    old_range_profiles = np.fft.ifft(np.fft.ifftshift(phase_history, axes=[1]), axis=1)
    new_range_profiles[:, :new_num_freqs // 2] = old_range_profiles[:, :new_num_freqs // 2]
    new_range_profiles[:, -(new_num_freqs - new_num_freqs // 2):] = old_range_profiles[:, -(new_num_freqs -
                                                                                            new_num_freqs // 2):]
    new_range_profiles = window * new_range_profiles
    new_phase_history = np.fft.fftshift(np.fft.fft(new_range_profiles, axis=1), axes=[1])
    return new_phase_history


def cosine_tapered_bandpass_filter(num_samples: int, first_blank_ends: float, flat_starts: float,
                                   flat_ends: float, second_blank_starts: float) -> NDArray:
    """
    Makes an array which produces a flattop with tapered cosine ends.
    Parameters
    ----------
    num_samples : int
        Number of samples we want in the filter.
    first_blank_ends : float
        Fraction of the way from the start we wish to start the cosine upramp.
    flat_starts : float
        Fraction of the way through we wish to end the cosine upramp.
    flat_ends : float
        Fraction of the way through we wish to start the cosine downramp.
    second_blank_starts : float
        Fraction of the way through we wish to end the cosine downramp.

    Returns
    -------
    final_filter : NDArray
        The filter we want. Complex-valued.
    """
    xs = np.linspace(0.0, 1.0, num_samples)
    output = np.zeros(num_samples, dtype="complex128")
    output[np.logical_and(xs < flat_ends, xs > flat_starts)] = 1.0
    output[np.logical_and(xs > first_blank_ends, xs < flat_starts)] = 0.5 * (
            1.0 - np.cos(pi * (first_blank_ends - xs) / (flat_starts - first_blank_ends)))[
        np.logical_and(xs > first_blank_ends, xs < flat_starts)]
    output[np.logical_and(xs > flat_ends, xs < second_blank_starts)] = 0.5 * (
            1.0 + np.cos(pi * (flat_ends - xs) / (second_blank_starts - flat_ends)))[
        np.logical_and(xs > flat_ends, xs < second_blank_starts)]
    return output


def log_spaced_samples(start: float, end: float, num_samples) -> NDArray:
    """
    Function to generate logarithmically-spaced values between an upper and lower value (inclusive).
    Useful for parameter scans.
    Parameters
    ----------
    start : float
        Lowest value.
    end : float
        Highest value.
    num_samples : int
        Number of values to generate.

    Returns
    -------
    log_spaced_samples : NDArray
        Logarithmically evenly-spaced values between start and end, in an array.
    """
    start_log = np.log(start)
    end_log = np.log(end)
    logged_samples = np.linspace(start_log, end_log, num_samples)
    return np.exp(logged_samples)
