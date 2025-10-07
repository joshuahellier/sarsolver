from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from .sarsolver_cxx import SarCalculationInfo, adjoint_evaluate
from ..base import BaseSarScene, BaseSimpleSarAperture, block_phase_history_array
from ..utils import default_num_threads


def single_adjoint_evaluate(scene: BaseSarScene, measurements: BaseSimpleSarAperture, adjoint_amplitudes: NDArray,
                            upsample_ratio: float = 128.0) -> NDArray:
    """
    Triggers evaluation of our bound single-threaded C++ adjoint evaluation code.
    Parameters
    ----------
    scene : BaseSarScene
        The BaseSarScene representing the image domain.
    measurements : BaseSimpleSarAperture
        The BaseSimpleSarAperture representing the SAR measurement aperture.
    adjoint_amplitudes : NDArray
        1D complex-valued numpy array containing the measurement amplitudes to be backprojected onto the image.
    upsample_ratio : float
        Upsampling ratio used in the calculation. Defaults to 128, which gives a modelling error of around 1%.

    Returns
    -------
    scat_amps : NDArray
        1D complex-valued numpy array containing the backprojected amplitudes.
    """
    sar_info = SarCalculationInfo(trans_posns=measurements.transmitter_positions,
                                  rec_posns=measurements.receiver_positions,
                                  srps=measurements.stab_ref_posns, scat_posns=scene.positions,
                                  phase_history=np.fft.ifftshift(adjoint_amplitudes, axes=1),
                                  scat_amplitudes=np.zeros(scene.num_scatterers, dtype="complex128"),
                                  waveform_fft=np.fft.ifftshift(measurements.waveform_fft),
                                  centre_freq=measurements.centre_frequency, upsample_ratio=upsample_ratio,
                                  sample_freq=measurements.sample_bandwidth, c_eff=measurements.c_eff,
                                  slow_time_weighting=measurements.slow_time_weighting,
                                  sign_multiplier=measurements.sign_multiplier)
    adjoint_evaluate(sar_info)
    return sar_info.scattering_amplitudes_py


def multi_adjoint_evaluate(scene: BaseSarScene, measurements: BaseSimpleSarAperture, adjoint_amplitudes: NDArray,
                           num_threads: Optional[int] = None, upsample_ratio: float = 128.0, verbose: bool = False,
                           num_calculation_blocks: int = 1) -> NDArray:
    """

    Parameters
    ----------
    scene : BaseSarScene
        The BaseSarScene representing the image domain.
    measurements : BaseSimpleSarAperture
        The BaseSimpleSarAperture representing the SAR measurement aperture.
    adjoint_amplitudes : NDArray
        1D complex-valued numpy array containing the measurement amplitudes to be backprojected onto the image.
    num_threads : int
        Number of threads to distribute the calculation over.
    upsample_ratio : float
        Upsampling ratio used in the calculation. Defaults to 128, which gives a modelling error of around 1%.verbose
    num_calculation_blocks : int
        Number of sequential blocks to chop calculation into. Lowers maximum RAM usage.

    Returns
    -------
    scat_amps : NDArray
        1D complex-valued numpy array containing the backprojected amplitudes.
    """
    if num_threads is None:
        num_threads = default_num_threads()

    meas_blocks = measurements.to_blocks(num_calculation_blocks)
    scene_blocks = scene.to_blocks(num_threads)
    m_blocks = block_phase_history_array(adjoint_amplitudes, num_calculation_blocks)

    if verbose:
        m_indices = tqdm(range(num_calculation_blocks))
    else:
        m_indices = range(num_calculation_blocks)

    scattering_chunks = []
    for m_index in m_indices:
        arg_list = []
        for h_index in range(num_threads):
            arg_list.append((scene_blocks[h_index], meas_blocks[m_index], m_blocks[m_index], upsample_ratio))
        backprojections = Parallel(n_jobs=num_threads)([
            delayed(single_adjoint_evaluate)(*arg) for arg in arg_list])
        backproj = np.concatenate(backprojections, axis=0)
        scattering_chunks.append(backproj)

    return np.sum(np.stack(scattering_chunks, axis=0), axis=0)
