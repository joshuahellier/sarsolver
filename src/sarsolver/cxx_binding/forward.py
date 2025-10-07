from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from .sarsolver_cxx import SarCalculationInfo, forward_evaluate
from ..base import BaseSarScene, BaseSimpleSarAperture, block_hypothesis_vector
from ..utils import default_num_threads


def single_forward_evaluate(scene: BaseSarScene, measurements: BaseSimpleSarAperture, scattering_amplitudes: NDArray,
                            upsample_ratio: float = 128.0) -> NDArray:
    """
        Triggers evaluation of our bound single-threaded C++ adjoint evaluation code.
        Parameters
        ----------
        scene : BaseSarScene
            The BaseSarScene representing the image domain.
        measurements : BaseSimpleSarAperture
            The BaseSimpleSarAperture representing the SAR measurement aperture.
        scattering_amplitudes : NDArray
            1D complex-valued numpy array containing the image scattering amplitudes to be forward projected.
        upsample_ratio : float
            Upsampling ratio used in the calculation. Defaults to 128, which gives a modelling error of around 1%.

        Returns
        -------
        meas_amps : NDArray
            1D complex-valued numpy array containing the scattered amplitudes.
        """
    sar_info = SarCalculationInfo(trans_posns=measurements.transmitter_positions,
                                  rec_posns=measurements.receiver_positions,
                                  srps=measurements.stab_ref_posns, scat_posns=scene.positions,
                                  phase_history=np.zeros([measurements.num_slow_times,
                                                          measurements.num_freqs], dtype="complex128"),
                                  scat_amplitudes=scattering_amplitudes,
                                  waveform_fft=np.fft.ifftshift(measurements.waveform_fft),
                                  centre_freq=measurements.centre_frequency, sample_freq=measurements.sample_bandwidth,
                                  c_eff=measurements.c_eff, sign_multiplier=measurements.sign_multiplier,
                                  slow_time_weighting=measurements.slow_time_weighting, upsample_ratio=upsample_ratio)
    forward_evaluate(sar_info)
    return np.fft.fftshift(sar_info.phase_history_py, axes=1)


def multi_forward_evaluate(scene: BaseSarScene, measurements: BaseSimpleSarAperture, scattering_amplitudes: NDArray,
                           num_threads: Optional[int] = None, upsample_ratio: float = 128.0, verbose: bool = False,
                           num_calculation_blocks: int = 1) -> NDArray:
    """

        Parameters
        ----------
        scene : BaseSarScene
            The BaseSarScene representing the image domain.
        measurements : BaseSimpleSarAperture
            The BaseSimpleSarAperture representing the SAR measurement aperture.
        scattering_amplitudes : NDArray
            1D complex-valued numpy array containing the image scattering amplitudes to be forward projected.
        num_threads : int
            Number of threads to distribute the calculation over.
        upsample_ratio : float
            Upsampling ratio used in the calculation. Defaults to 128, which gives a modelling error of around 1%.
        verbose : bool
            Whether to have tqdm progress bar for calculation. Defaults to false.
        num_calculation_blocks : int
            Number of sequential blocks to chop calculation into. Lowers maximum RAM usage.

        Returns
        -------
        meas_amps : NDArray
            1D complex-valued numpy array containing the scattered amplitudes.
        """
    if num_threads is None:
        num_threads = default_num_threads()

    meas_blocks = measurements.to_blocks(num_calculation_blocks)
    scene_blocks = scene.to_blocks(num_threads)
    h_blocks = block_hypothesis_vector(scattering_amplitudes, num_threads)

    if verbose:
        m_indices = tqdm(range(num_calculation_blocks))
    else:
        m_indices = range(num_calculation_blocks)

    phase_history_chunks = []
    for m_index in m_indices:
        arg_list = []
        for h_index in range(num_threads):
            arg_list.append((scene_blocks[h_index], meas_blocks[m_index], h_blocks[h_index], upsample_ratio))
        partial_sums = (Parallel(n_jobs=num_threads)([
            delayed(single_forward_evaluate)(*arg) for arg in arg_list]))
        ph_chunk = np.sum(np.stack(partial_sums, axis=0), axis=0)
        phase_history_chunks.append(ph_chunk)

    return np.concatenate(phase_history_chunks, axis=0)
