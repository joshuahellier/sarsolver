from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from ..base import BaseSarScene, BaseSimpleSarAperture
from ..cxx_binding import multi_forward_evaluate, multi_adjoint_evaluate


class NumpySimpleSarOperator(LinearOperator):
    def __init__(self, scene: BaseSarScene, aperture: BaseSimpleSarAperture, num_threads: Optional[int] = None,
                 upsample_ratio: float = 128.0, verbose: bool = False, num_calculation_blocks: int = 1):
        """
        Numpy-based implementation of the scalar Born SAR operator.
        Parameters
        ----------
        scene : BaseSarScene
            The scene we're projecting from.
        aperture : BaseSimpleSarAperture
            The SAR aperture we're measuring with.
        num_threads : int
            Number of threads to use when parallelising.
        upsample_ratio : float
            Upsample ratio used in calculation. Calculations for individual slow times are carried out at a sample rate
            of upsample_ratio*sample_freq, allowing us to use zeroth-order interpolation in the fast time domain (and
            therefore enabling us to do a cheap calculation). Defaults to 128, which is a good trade in terms of
            modelling error from interpolation (around 1%) and RAM usage/access time.
        verbose : bool
            Whether to have tqdm progress bar for calculation. Defaults to false.
        num_calculation_blocks : int
            Number of sequential blocks to cut calculation into, to reduce RAM at the cost of a performance hit.
            Useful for large problems. Defaults to 1.
        """
        super().__init__(dtype=np.complex128, shape=[aperture.num_freqs * aperture.num_slow_times,
                                                     scene.num_scatterers])
        self._scene = scene
        self._aperture = aperture
        self._num_calculation_blocks = num_calculation_blocks
        self._upsample_ratio = upsample_ratio
        self._verbose = verbose
        self._num_threads = num_threads

    def _matvec(self, x: NDArray) -> NDArray:
        u = x.reshape([self._scene.num_scatterers])
        v = multi_forward_evaluate(scene=self._scene, measurements=self._aperture, scattering_amplitudes=u,
                                   num_threads=self._num_threads, num_calculation_blocks=self._num_calculation_blocks,
                                   upsample_ratio=self._upsample_ratio, verbose=self._verbose)
        return v.reshape([self._aperture.num_freqs * self._aperture.num_slow_times])

    def _rmatvec(self, y: NDArray) -> NDArray:
        v = y.reshape([self._aperture.num_slow_times, self._aperture.num_freqs])
        u = multi_adjoint_evaluate(scene=self._scene, measurements=self._aperture, adjoint_amplitudes=v,
                                   num_threads=self._num_threads, num_calculation_blocks=self._num_calculation_blocks,
                                   upsample_ratio=self._upsample_ratio, verbose=self._verbose)
        return u.reshape([self._scene.num_scatterers])
