import re
from ctypes import cdll, c_size_t, c_double, POINTER, Structure
from os.path import isfile, join
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
from numpy.ctypeslib import as_ctypes
from numpy.typing import NDArray
from scipy.constants import c


def get_dir_files(x: Union[Path, str]) -> List[Path]:
    """
    Get all the files in a directory.
    Parameters
    ----------
    x : Union[Path, str]
        Directory to look in.
    Returns
    -------
    dir_files : List[Path]
        Files (not directories) found under x.
    """
    x = Path(x)
    y = [f for f in x.iterdir() if isfile(join(x, f))]
    return y


# First we work out where the compiled object we're trying to link is using regex. This is slightly hacky,
# but seems to be necessary as we don't seem to be able to control what SetupTools/CiBuildTools ends up calling the binary.

this_file = Path(__file__)
project_root = this_file.parent.parent
candidate_files = get_dir_files(project_root)
so_location = None
re_pattern = re.compile(r".*sarsolver_cxx\..*so")
for f in candidate_files:
    if re_pattern.search(str(f.resolve())):
        so_location = f

if so_location is not None:
    my_lib = cdll.LoadLibrary(str(so_location))
else:
    raise ValueError("Compiled binary sar_data_collect.so (or equivalent) not found! Check build.")


class SarCalculationInfo(Structure):
    """
    Equivalent to SarCalculationInfo the C struct defined in sarsolver_cxx.hpp .
    """
    _fields_ = [("num_fast_times", c_size_t), ("num_slow_times", c_size_t), ("num_scatterers", c_size_t),
                ("transmit_posns", POINTER(c_double)), ("receive_posns", POINTER(c_double)),
                ("stab_ref_posns", POINTER(c_double)),
                ("scat_posns", POINTER(c_double)),
                ("phase_history", POINTER(c_double)), ("scattering_amplitudes", POINTER(c_double)),
                ("waveform_fft", POINTER(c_double)),
                ("slow_time_weighting", POINTER(c_double)),
                ("centre_frequency", c_double), ("sample_frequency", c_double), ("c_eff", c_double),
                ("upsample_ratio", c_double),
                ("sign_multiplier", c_double)]

    def __init__(self, trans_posns: NDArray, rec_posns: NDArray, srps: NDArray, scat_posns: NDArray,
                 phase_history: NDArray, scat_amplitudes: NDArray, waveform_fft: NDArray, slow_time_weighting: NDArray,
                 centre_freq: float, sample_freq: float, c_eff: float = c, upsample_ratio: float = 128.0,
                 sign_multiplier: float = 1.0):
        """
        Constructor. Infers array shapes from NDArrays provided. Will throw errors if shapes are inconsistent.
        Parameters
        ----------
        trans_posns : NDArray
            [num_slow_times, 3] real-valued array of the aperture's transmitter positions.
        rec_posns : NDArray
            [num_slow_times, 3] real-valued array of the aperture's receiver positions.
        srps : NDArray
            [num_slow_times, 3] real-valued array of the aperture's stabilisation reference points.
        scat_posns  : NDArray
            [num_scatterers, 3] real-valued array of the scene's scattering site positions.
        phase_history : NDArray
            [num_slow_times, num_fast_times] complex-valued array of the phase history.
        scat_amplitudes : NDArray
            [num_scatterers] complex-valued array of the scattering amplitudes.
        waveform_fft : NDArray
            [num_fast_times] complex-valued array of the waveform's Fourier transform.
        slow_time_weighting : NDArray
            [num_slow_times] complex-valued array of any slow time weighting we want to use.
        centre_freq : float
            Centre frequency of the physical-band waveform.
        sample_freq : float
            Sampling rate used in the calculation.
        c_eff : float
            Speed of light used in calculation. Defaults to the vacuum value.
        upsample_ratio : float
            Upsample ratio used in calculation. Calculations for individual slow times are carried out at a sample rate
            of upsample_ratio*sample_freq, allowing us to use zeroth-order interpolation in the fast time domain (and
            therefore enabling us to do a cheap calculation). Defaults to 128, which is a good trade in terms of
            modelling error from interpolation (around 1%) and RAM usage/access time.
        sign_multiplier : float
            Sign convention multiplier, which should be -1.0 or +1.0 (defaults to 1.0). This corresponds to which
            way around the quadrature demodulation circuit for the measurement was wired. Both are equally valid,
            measurement data (e.g. CPHD) should indicate which convention was used.
        """
        self.num_fast_times = phase_history.shape[1]
        self.num_slow_times = phase_history.shape[0]
        self.num_scatterers = scat_posns.shape[0]
        self.centre_frequency = centre_freq
        self.sample_frequency = sample_freq
        self.c_eff = c_eff
        self.upsample_ratio = upsample_ratio
        self.sign_multiplier = sign_multiplier

        self.trans_posns_py = trans_posns.astype("float64")
        x = self.trans_posns_py.reshape([-1])
        if len(x) != 3 * self.num_slow_times:
            raise ValueError("Check transmitter positions.")
        self.transmit_posns = as_ctypes(x)

        self.rec_posns_py = rec_posns.astype("float64")
        x = self.rec_posns_py.reshape([-1])
        if len(x) != 3 * self.num_slow_times:
            raise ValueError("Check receiver positions.")
        self.receive_posns = as_ctypes(x)

        self.stab_ref_posns_py = srps.astype("float64")
        x = self.stab_ref_posns_py.reshape([-1])
        if len(x) != 3 * self.num_slow_times:
            raise ValueError("Check stab ref positions.")
        self.stab_ref_posns = as_ctypes(x)

        self.scat_posns_py = scat_posns.astype("float64")
        x = self.scat_posns_py.reshape([-1])
        if len(x) != 3 * self.num_scatterers:
            raise ValueError("Check transmitter positions.")
        self.scat_posns = as_ctypes(x)

        self._phase_history_py = np.zeros(shape=[self.num_slow_times, self.num_fast_times, 2], dtype="float64")
        self._phase_history_py[:, :, 0] = np.real(phase_history).astype("float64")
        self._phase_history_py[:, :, 1] = np.imag(phase_history).astype("float64")
        x = self._phase_history_py.reshape([-1])
        if len(x) != 2 * self.num_slow_times * self.num_fast_times:
            raise ValueError("Check phase history.")
        self.phase_history = as_ctypes(x)

        self._scattering_amplitudes_py = np.zeros(shape=[self.num_scatterers, 2], dtype="float64")
        self._scattering_amplitudes_py[:, 0] = np.real(scat_amplitudes).astype("float64")
        self._scattering_amplitudes_py[:, 1] = np.imag(scat_amplitudes).astype("float64")
        x = self._scattering_amplitudes_py.reshape([-1])
        if len(x) != 2 * self.num_scatterers:
            raise ValueError("Check scattering amplitudes.")
        self.scattering_amplitudes = as_ctypes(x)

        self._waveform_fft_py = np.zeros(shape=[self.num_fast_times, 2], dtype="float64")
        self._waveform_fft_py[:, 0] = np.real(waveform_fft).astype("float64")
        self._waveform_fft_py[:, 1] = np.imag(waveform_fft).astype("float64")
        x = self._waveform_fft_py.reshape([-1])
        if len(x) != 2 * self.num_fast_times:
            raise ValueError("Check waveform.")
        self.waveform_fft = as_ctypes(x)

        self._slow_time_weighting_py = np.zeros(shape=[self.num_slow_times, 2], dtype="float64")
        self._slow_time_weighting_py[:, 0] = np.real(slow_time_weighting).astype("float64")
        self._slow_time_weighting_py[:, 1] = np.imag(slow_time_weighting).astype("float64")
        x = self._slow_time_weighting_py.reshape([-1])
        if len(x) != 2 * self.num_slow_times:
            raise ValueError("Check slow time weighting.")
        self.slow_time_weighting = as_ctypes(x)

    @property
    def phase_history_py(self) -> NDArray:
        """
        Gives access to the phase history array used for measurement.
        Returns
        -------
`       phase_history : NDArray
            Phase history array used for the measurement. Complex-valued, [num_slow_times, num_fast_times].
        """
        return self._phase_history_py[:, :, 0] + 1.j * self._phase_history_py[:, :, 1]

    @property
    def scattering_amplitudes_py(self) -> NDArray:
        """
        Gives access to the scattering amplitude array used for the scene.
        Returns
        -------
        scattering_amplitudes : NDArray
            Scattering amplitude array used for the scene. Complex-valued, [num_scatterers].
        """
        return self._scattering_amplitudes_py[:, 0] + 1.j * self._scattering_amplitudes_py[:, 1]

    @property
    def waveform_fft_py(self) -> NDArray:
        """
        Gives access to the waveform fft array used in calculations.
        Returns
        -------
        waveform_fft : NDArray
            Waveform fft used in the calculations. Complex-valued, [num_fast_times].
        """
        return self._waveform_fft_py[:, 0] + 1.0j * self._waveform_fft_py[:, 1]

    @property
    def slow_time_weighting_py(self) -> NDArray:
        """
        Gives access to the slow time weighting used in the calculations.
        Returns
        -------
        slow_time_weighting : NDArray
            Slow time weighting used in the calculations. Complex-valued, [num_slow_times].
        """
        return self._slow_time_weighting_py[:, 0] + 1.0j * self._slow_time_weighting_py[:, 1]


forward_evaluate = my_lib.forward_evaluate
forward_evaluate.restype = None
forward_evaluate.argtypes = [POINTER(SarCalculationInfo)]

adjoint_evaluate = my_lib.adjoint_evaluate
adjoint_evaluate.restype = None
adjoint_evaluate.argtypes = [POINTER(SarCalculationInfo)]
