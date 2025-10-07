from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Number
from typing import Optional, Sequence

import numpy as np
from cil.framework import DataContainer
from cil.framework.labels import FillType


class GenericCilGeometry(ABC):
    """
    Class designed to abstract CIL's AcquisitionGeometry and ImageGeometry. The purpose of an instance
    of these is to be able to allocate memory and create a DataContainer. To do that, we need to know the shape
    of the array to be allocated, and the datatype. We've also got forced deep copying here because we
    encountered issues with CIL solvers mutating geometries.
    """

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        """
        The shape of the DataContainer this geometry allocates.
        Returns
        -------
        shape : Sequence[int]
            The shape.
        """
        raise NotImplementedError

    @property
    def dtype(self) -> type:
        """
        The datatype (e.g. np.complex128) of the DataContainer this geometry allocates.
        Returns
        -------
        dtype : type
            The type.
        """
        raise NotImplementedError

    def allocate(self, value: Number = 0.0, dtype: Optional[type] = None,
                 seed: Optional[int] = None, max_value: int = 100) -> "ExtendedDataContainer":
        """
        Create an ActualDataContainer using this Geometry. Largely copied from CIL's DataContainer, but able to handle
        complex numbers as is required for SAR.
        Parameters
        ----------
        value : Number
            Value to initialise the output's array to. Defaults to zero.
        dtype : Optional[dtype]
            Allows override of the output's dtype. Defaults to the one used by this geometry.
        seed : Optional[int]
            Seed used if randomness required. Defaults to None, in which case no seed will be set,
            and default seed will be used.
        max_value : int
            Maximum value used during random number generation. Defaults to 100.
        Returns
        -------
            An ActualDataContainer corresponding to this geometry, with memory pre-allocated for its array.
        """

        if dtype is None:
            dtype = self.dtype

        out = ExtendedDataContainer(array=np.full(shape=self.shape, fill_value=0, dtype=dtype), geometry=self.copy())

        if value is None:
            return out

        if value == FillType.RANDOM:
            if seed is not None:
                np.random.seed(seed)
            if np.iscomplexobj(out.array):
                r = np.random.random_sample(self.shape) + 1j * np.random.random_sample(self.shape)
                out.fill(r)
            else:
                out.fill(np.random.random_sample(self.shape))
            return out

        if value == FillType.RANDOM_INT:
            if seed is not None:
                np.random.seed(seed)
            r = np.random.randint(max_value, size=self.shape, dtype=np.int32)
            out.fill(np.asarray(r, dtype=self.dtype))
            return out

        out.fill(value)
        return out

    def copy(self) -> "GenericCilGeometry":
        """
        Make a deep copy of this geometry. Will be triggered by calls such as allocate().
        Returns
        -------
        copy : GenericCilGeometry
            A deep copy of this geometry.
        """
        return deepcopy(self)


class ExtendedDataContainer(DataContainer):
    """
    Our extension of DataContainer in which we override the geometry property so that it can have a nontrivial
    geometry (presumably this is enforced in CIL to prevent DataContainer from being used).
    """

    def __init__(self, *args, **kwargs):
        self._dimension_labels = None
        super().__init__(*args, **kwargs)

    @property
    def geometry(self) -> GenericCilGeometry:
        """
        The geometry used by this DataContainer. Using its allocate() method should give us another DataContainer
        just like this.
        Returns
        -------
        geometry : GenericCilGeometry
            The geometry which describes the layout of the data in this DataContainer.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, val: GenericCilGeometry):
        """
        Geometry setter method.
        Parameters
        ----------
        val

        Returns
        -------

        """
        self._geometry = val
