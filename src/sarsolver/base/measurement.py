from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi


def normalise(x: NDArray) -> NDArray:
    """
    Treats an array like an array of vectors in the last index, and returns the same array but with those vectors
    2-normalised.
    Parameters
    ----------
    x : array_like
        Array. Should be at least 2-dimensional, and the vectors x[i, ..., j, :] shouldn't be 0 for best results.

    Returns
    -------
    y : NDArray
        Array of normalised vectors.
    """
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def bad_indices(x: NDArray) -> NDArray:
    """
    For 2D arrays, returns 0-indices where there's an issue scanning in the last axis (NaN, Inf, etc)
    Parameters
    ----------
    x : array_like
        Array to be checked for bad indices.

    Returns
    -------
    output : NDArray
        Array of indices where bad behaviour was found scanning in the last index.
    """
    is_bad = np.logical_not(np.isfinite(x))
    is_any_bad = np.any(is_bad, axis=-1)
    return np.argwhere(is_any_bad)[:, 0]


def block_phase_history_array(m: NDArray, num_blocks: int = 1) -> List[NDArray]:
    """
    Function which chops up a 2D array into num_blocks approximately equal components.
    Parameters
    ----------
    m : array_like
        Array to be chopped up
    num_blocks : int
        Number of blocks to chop into

    Returns
    -------
    output : NDArray
        List of arrays. If concatenated along 0-axis, would yield m.
    """
    if num_blocks == 1:
        return [m]
    block_length = m.shape[0] // num_blocks
    output = []
    for block_index in range(num_blocks):
        start_index = block_length * block_index
        end_index = start_index + block_length
        if block_index == num_blocks - 1:
            end_index = m.shape[0]
        output.append(m[start_index:end_index, :])
    return output


class BaseSimpleSarAperture(ABC):
    """
    ABC for a generic bistatic SAR collection. Must have standard things implemented (transmitter/receiver positions,
    frequencies, etc). Has methods for blocking, slicing, and finding broken behaviour within itself (e.g. NaN
    readings in transmitter positions), as well as methods to calculate k-space support for this aperture at a given
    point.
    """

    @property
    @abstractmethod
    def num_slow_times(self) -> int:
        """
        The number of slow times in the aperture.
        Returns
        -------
        num_slow_times : int
            Number of slow times.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_freqs(self) -> int:
        """
        The number of fast times/fast frequencies in the aperture. This must be the same for all pulses, having
         it change pulse-to-pulse is annoying and unnecessary (just resample/window!).
        Returns
        -------
        num_freqs : int
            Number of fast times.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def centre_frequency(self) -> float:
        """
        The centre frequency, which we're modulating/demodulating wrt.
        Returns
        -------
        centre_frequency : float
            The centre frequency, in Hz.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_bandwidth(self) -> float:
        """
        The fast-time sample rate for the signal recording.
        Returns
        -------
        sample_bandwidth : float
            Fast-time sampling frequency, in Hz.
        """
        raise NotImplementedError

    @property
    def freqs(self) -> np.array:
        """
        Method for getting the fast frequency abcissae for phase history associated with this aperture.
        It accounts for whether the number of fast frequencies is even or odd.
        Returns
        -------
        freqs : NDArray
            The fast frequencies for this collect.
        """
        return self.centre_frequency + np.fft.fftshift(np.fft.fftfreq(self.num_freqs, 1.0 / self.sample_bandwidth))

    @property
    @abstractmethod
    def transmitter_positions(self) -> NDArray:
        """
        Array of transmitter phase centre positions (Cartesian coordinates, in metres). Should be a
        [num_slow_times, 3] float (probably 128-bit)-valued array.
        Returns
        -------
        transmitter_positions : NDArray
            Transmitter positions in this aperture.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def receiver_positions(self) -> NDArray:
        """
        Array of receiver phase centre positions (Cartesian coordinates, in metres). Should be a
        [num_slow_times, 3] float (probably 128-bit)-valued array.
        Returns
        -------
        receiver_positions : NDArray
            Receiver positions in this aperture.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def stab_ref_posns(self) -> NDArray:
        """
        Array of stabilisation reference points (SRPs) (Cartesian coordinates, in metres). Should be a
        [num_slow_times, 3] float (probably 128-bit)-valued array.
        Returns
        -------
        receiver_positions : NDArray
            Stabilisation reference points in this aperture.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def waveform_fft(self) -> NDArray:
        """
        Fast-frequency domain copy of the waveform used, or at least its envelope.
        Can be useful for range sidelobe control.
        Returns
        -------
        waveform_fft : NDArray
            Length num_freqs complex-valued array.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def slow_time_weighting(self) -> NDArray:
        """
        Weighting applied to different measurements as a function of slow time.
        Can be useful for cross-range sidelobe control, although
        gets complicated in volumetrics.
        Returns
        -------
        slow_time_weighting : NDArray
            Length num_slow_times complex-valued array.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sign_multiplier(self) -> float:
        """
        Modulation sign convention (+1/-1) being used in this dataset. Depends how the receiver circuit in the
        radar is set up.
        Returns
        -------
        sign_multiplier : float
            -1.0 or 1.0
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def c_eff(self) -> float:
        """
        Radio wave propagation speed to be used for this dataset. Often in practice slightly lower than vacuum value.
        In m/s.
        Returns
        -------
        c_eff : float
            Radio wave speed
        """
        raise NotImplementedError

    @abstractmethod
    def to_blocks(self, num_blocks: int = 1) -> List["BaseSimpleSarAperture"]:
        """
        Creates num_blocks subapertures from this aperture, and returns them in a list. This is for solver use;
        for large apertures it is often beneficial to break the calculation into parts.
        Parameters
        ----------
        num_blocks : int
            Number of subapertures we want to end up with.

        Returns
        -------
        blocks : List[BaseSimpleSarAperture]
            List of subapertures.
        """
        raise NotImplementedError

    def _unit_k_vecs(self, real_space_origin: Optional[np.array] = None) -> NDArray:
        """
        Computes the k_space destinations of 1Hz waves, in terms of equivalent monostatic aperture.
        We can then multiply up to get other frequencies.
        These are the difference between an incoming unit vector from the transmitter and an outgoing
        unit vector bound for the receiver. This is effectively the 'momentum transfer' in scattering language,
        and corresponds to the part of the scene's local spatial spectrum we expect to be probed in that pulse.
        Parameters
        ----------
        real_space_origin : array_like
            Cartesian coordinates, in m, of point for which we want to investigate the k-space support
            this aperture provides. Length 3 array.
        Returns
        -------
        unit_k_vecs : NDArray
            Array of 'unit_k_vecs', float-valued, shape [num_slow_times, 3].
        """
        if real_space_origin is not None:
            origins = np.expand_dims(real_space_origin, axis=0)
        else:
            origins = self.stab_ref_posns
        tran_vecs = normalise(origins - self.transmitter_positions)
        rec_vecs = normalise(self.receiver_positions - origins)
        return rec_vecs - tran_vecs

    def _inner_k_vecs(self, real_space_origin: Optional[np.array] = None) -> NDArray:
        """
        Computes the lowest-frequency k-vectors we expect to be probing at real_space_origin.
        Parameters
        ----------
        real_space_origin : array_like
            Cartesian coordinates, in m, of point for which we want to investigate the k-space support
            this aperture provides. Length 3 array.
        Returns
        -------
        inner_k_vecs : NDArray
            Array of 'inner_k_vecs', float-valued, shape [num_slow_times, 3].
        """
        return self._unit_k_vecs(real_space_origin) * (2.0 * pi / self.c_eff) * (self.centre_frequency
                                                                                 - 0.5 * self.sample_bandwidth)

    def _outer_k_vecs(self, real_space_origin: Optional[np.array] = None) -> NDArray:
        """
        Computes the highest-frequency k-vectors we expect to be probing at real_space_origin.
        Parameters
        ----------
        real_space_origin : array_like
            Cartesian coordinates, in m, of point for which we want to investigate the k-space support
            this aperture provides. Length 3 array.
        Returns
        -------
        outer_k_vecs : NDArray
            Array of 'outer_k_vecs', float-valued, shape [num_slow_times, 3].
        """
        return self._unit_k_vecs(real_space_origin) * (2.0 * pi / self.c_eff) * (self.centre_frequency
                                                                                 + 0.5 * self.sample_bandwidth)

    def edge_k_vecs(self, real_space_origin: Optional[np.array] = None) -> NDArray:
        """
        Computes the highest and lowest frequency k-vectors we expect to be probing at real_space_origin.
        Useful for k-space analysis. In particular, bounding the output of this function gives good methods
        for calculating Nyquist-compliant sample criteria for SAR scenes.
        Parameters
        ----------
        real_space_origin : array_like
            Cartesian coordinates, in m, of point for which we want to investigate the k-space support
            this aperture provides. Length 3 array.
        Returns
        -------
        edge_k_vecs : NDArray
            Array of 'edge_k_vecs', float-valued, shape [num_slow_times, 3].
        """
        return np.concatenate([self._inner_k_vecs(real_space_origin),
                               self._outer_k_vecs(real_space_origin)], axis=0)

    @property
    def bad_slow_times(self) -> List[int]:
        """
        Finds the slow times in which there is something broken about transmitter, receiver or scene reference
        positions. Broken means NaN or Infs etc.
        Returns
        -------
        bad_slow_times : List[int]
            Sorted list of troublesome slow times.
        """
        indices = list(set(bad_indices(self.transmitter_positions).tolist() +
                           bad_indices(self.receiver_positions).tolist() +
                           bad_indices(self.stab_ref_posns).tolist()))
        indices.sort()
        return indices

    @abstractmethod
    def get_subset(self, item: slice = None) -> "BaseSimpleSarAperture":
        """
        Produce a smaller version of this aperture by slicing in slow time using Python slicing.
        Parameters
        ----------
        item : slice
            A slice detailing the slow time indices we want used in the output.

        Returns
        -------
        subaperture : BaseSimpleSarAperture
            Aperture containing the desired slow times, as indicated by item.
        """
        raise NotImplementedError


class BaseSimpleSarDataset(ABC):
    """ABC for a SAR dataset, which is basically an aperture and some measured phase history.
    This is useful for I/O, as it corresponds roughly to the contents of a CPHD file."""

    @property
    @abstractmethod
    def aperture(self) -> BaseSimpleSarAperture:
        """
        The aperture for the dataset, with the details about the measurement.
        Returns
        -------
        aperture : BaseSimpleSarAperture
            The dataset's aperture.
        """
        raise NotImplementedError

    @abstractmethod
    def phase_history(self, item: slice = slice(None)) -> Optional[NDArray]:
        """
        The phase history for this dataset, i.e. the complex-valued radar data collected.
        Not a property as one often doesn't want the whole phase history. Instead,
        use slow-time slicing to get what you want.
        Parameters
        ----------
        item : slice

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def get_subset(self, item: slice = None) -> "BaseSimpleSarDataset":
        """
        Produce a smaller version of this dataset by slicing in slow time using Python slicing.
        Parameters
        ----------
        item : slice
            A slice detailing the slow time indices we want used in the output.

        Returns
        -------
        subset : BaseSimpleSarDataset
            Dataset containing the desired slow times, as indicated by item.
        """
        raise NotImplementedError
