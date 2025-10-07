from abc import ABC, abstractmethod
from typing import List

from numpy.typing import NDArray


def block_hypothesis_vector(h: NDArray, num_blocks: int = 1) -> List[NDArray]:
    """
    Cuts an NDArray into a list of num_blocks NDArrays, of roughly equal size. Cut done on first axis.
    Parameters
    ----------
    h : array_like
        Array to be chopped up.
    num_blocks : int
        Number of blocks to chop h into.
    Returns
    -------
    output : List[NDArray]
        Arrays h has been chopped into
    """
    if num_blocks == 1:
        return [h]
    block_length = h.shape[0] // num_blocks
    output = []
    for block_index in range(num_blocks):
        start_index = block_length * block_index
        end_index = start_index + block_length
        if block_index == num_blocks - 1:
            end_index = h.shape[0]
        output.append(h[start_index:end_index])
    return output


class BaseSarScene(ABC):
    """
    ABC for a generic SAR scene, i.e. discretised hypothesis space. Consists of a bunch of point scatterers and
    their positions.
    """

    @property
    @abstractmethod
    def positions(self) -> NDArray:
        """
        Locations of the scene's point scatterers.
        Returns
        -------
        positions : array_like
            [n, 3] float-like dtype NDArray with Cartesian point scatterer locations.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_scatterers(self) -> int:
        """
        The number of point scatterers in the scene.
        Returns
        -------
        num_scatterers : int
            The number of point scatterers in the scene.
        """
        raise NotImplementedError

    @abstractmethod
    def to_blocks(self, num_blocks: int = 1) -> List["BaseSarScene"]:
        """
        Creates num_blocks subscenes from this scene, and returns them in a list. This is for solver use;
        for large scenes it is often beneficial to break the calculation into parts.
        Parameters
        ----------
        num_blocks : int
            Number of subscenes to create.

        Returns
        -------
        subscenes : List[BaseSarScene]
            List of subscenes.
        """
        raise NotImplementedError
