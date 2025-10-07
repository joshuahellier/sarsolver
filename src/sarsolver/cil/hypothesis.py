from typing import Sequence

import numpy as np

from .base import GenericCilGeometry
from ..numpy import GridSarScene


class CilSarGridGeometry(GenericCilGeometry):
    """
    GenericCilGeometry which represents a grid-based volumetric SAR scene. It contains a GridSarScene from the numpy
    submodule.
    """

    def __init__(self, scene: GridSarScene):
        """
        Constructor for CilSarGridGeometry. It contains a GridSarScene from the numpy submodule.
        Parameters
        ----------
        scene : GridSarScene
            Previously-constructed GridSarScene we want to use.
        """
        self._scene = scene
        self._dtype = np.complex128

    @property
    def scene(self) -> GridSarScene:
        """
        The GridSarScene we're using.
        Returns
        -------
        scene : GridSarScene
            The GridSarScene we're using for inversion.
        """
        return self._scene

    @property
    def shape(self) -> Sequence[int]:
        """
        The shape of the array this Geometry should allocate.
        Returns
        -------
        shape : Sequence[int]
           The relevant array shape. CIL needs to know this for allocating memory and using regularisation functions.
        """
        return tuple(self._scene.grid_shape)

    @property
    def dtype(self) -> type:
        """
        Datatype of the array this Geometry shoud allocate.
        Returns
        -------
        dtype : type
           Relevant datatype. CIL needs this for allocating memory.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, val: type):
        """
        Setter for dtype. We need to be able to set it for CIL to work properly.
        Parameters
        ----------
        val : type
            Type we want to use.
        Returns
        -------
        None
        """
        self._dtype = val
