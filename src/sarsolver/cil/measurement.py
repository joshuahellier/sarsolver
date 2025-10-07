from typing import List

import numpy as np

from .base import GenericCilGeometry
from ..base import BaseSimpleSarAperture


class SimpleCilSarGeometry(GenericCilGeometry):
    """
    CIL Geometry representing a simple bistatic SAR measurement (set frequency range and sampling, with transmitters
    and receivers moving with slow time). Actual content is contained in a BaseSimpleSarAperture.
    """

    def __init__(self, aperture: BaseSimpleSarAperture):
        """
        Constructor for SimpleCilSarGeometry.
        Parameters
        ----------
        aperture : BaseSimpleSarAperture
            The collection aperture we want to represent in CIL.
        """
        self._aperture = aperture
        self._dtype = np.complex128

    @property
    def aperture(self) -> BaseSimpleSarAperture:
        """
        Our SAR measurement aperture.
        Returns
        -------
        aperture : BaseSimpleSarAperture
            Our SAR measurement aperture.
        """
        return self._aperture

    @property
    def shape(self) -> List[int]:
        """
        Measurement array shape, for CIL. We always use
        Returns
        -------
        shape : List[int]
            Shape of the measurement array this geometry would allocate.
        """
        return [self.aperture.num_slow_times, self.aperture.num_freqs]

    @property
    def dtype(self) -> type:
        """
        Datatype of array this geometry would allocate.
        Returns
        -------
        dtype : type
            Relevant datatype, probably np.complex128.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, val: type):
        """
        Setter for datatype. Required for CIL.
        Parameters
        ----------
        val : type
            Type we want to set for array allocation.
        Returns
        -------
        None
        """
        self._dtype = val
