from typing import Optional

from cil.framework import DataContainer
from cil.optimisation.operators import LinearOperator

from .hypothesis import CilSarGridGeometry
from .measurement import SimpleCilSarGeometry
from ..cxx_binding import multi_adjoint_evaluate, multi_forward_evaluate


class CpuSarOperator(LinearOperator):
    def __init__(self, image_geometry: CilSarGridGeometry, measurement_geometry: SimpleCilSarGeometry,
                 upsample_ratio: float = 128.0, num_threads: Optional[int] = None, verbose: bool = False,
                 num_calculation_blocks: Optional[int] = 1):
        """
        CIL LinearOperator extension for multithreaded CPU SAR, using our bound C++ code.
        Parameters
        ----------
        image_geometry : CilSarGridGeometry
            CIL geometry for the scene.
        measurement_geometry : SimpleCilSarGeometry
            CIL geometry description of the collection manifold
        upsample_ratio : float
            Upsampling factor used internally in the calculation. More is better for accuracy.
            128x (the default) gives about 1 part in 100 resampling error.
        num_threads : int
            Number of threads to parallelise over. Default (None) will trigger evaluation with one less
            than the number of threads available on the system.
        verbose : bool
            Whether to have a progress bar or not.
        num_calculation_blocks : int
            Number of slow time blocks to chop the calculation into. Blocking saves RAM and doesn't hit
            performance too hard.
        """
        self._domain_geometry = image_geometry
        self._range_geometry = measurement_geometry
        self._norm = None
        self._upsample_ratio = upsample_ratio
        self._num_threads = num_threads
        self._num_calculation_blocks = num_calculation_blocks
        self._verbose = verbose

    @property
    def domain(self) -> CilSarGridGeometry:
        """
        CIL operator domain geometry.
        Returns
        -------
        domain : CilSarGridGeometry
           Domain geometry, describing scene.
        """
        return self._domain_geometry

    @property
    def range(self) -> SimpleCilSarGeometry:
        """
        CIL operator range geometry/
        Returns
        -------
        range : SimpleCilSarGeometry
            Range geometry, describing measurement.
        """
        return self._range_geometry

    def direct(self, x: DataContainer, out: Optional[DataContainer] = None) -> DataContainer:
        """
        Direct operation for CIL operator.
        Parameters
        ----------
        x : DataContainer
            Image hypothesis we wish to predict measurement from.
        out : Optional[DataContainer]
            Measurement DataContainer to be mutated to contain the predicted measurement.

        Returns
        -------
        ret_val : DataContainer
            Measurement DataContainer containing the predicted measurement.
        """
        if out is None:
            out = self.range.allocate()
        h = x.array.reshape([-1])
        m = multi_forward_evaluate(self.domain.scene, self.range.aperture, h, num_threads=self._num_threads,
                                   upsample_ratio=self._upsample_ratio, verbose=self._verbose,
                                   num_calculation_blocks=self._num_calculation_blocks)
        out.array = m
        return out

    def adjoint(self, x: DataContainer, out: Optional[DataContainer] = None) -> DataContainer:
        """
        Adjoint operation for CIL operator.
        Parameters
        ----------
        x : DataContainer
            Measurement to be backprojected onto image domain.
        out : Optional[DataContainer]
            DataContainer to be mutated to contain backprojection.

        Returns
        -------
        ret_val : DataContainer
            Image DataContainer containing backprojected image.
        """
        if out is None:
            out = self.domain.allocate()
        m = x.array
        h = multi_adjoint_evaluate(self.domain.scene, self.range.aperture, m, num_threads=self._num_threads,
                                   upsample_ratio=self._upsample_ratio, verbose=self._verbose,
                                   num_calculation_blocks=self._num_calculation_blocks)
        out.array = h.reshape(self.domain.shape)
        return out
