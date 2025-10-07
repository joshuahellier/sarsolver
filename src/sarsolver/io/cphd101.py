from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from sarpy.io.phase_history.base import CPHDTypeReader
from sarpy.io.phase_history.converter import open_phase_history
from scipy.constants import c

from ..base import BaseSimpleSarDataset
from ..numpy import NumpySimpleSarDataset, NumpySimpleSarAperture


class Cphd101Dataset(BaseSimpleSarDataset):
    """Class for representing one channel of a dataset stored as a CPHD 0.3 file."""

    def __init__(self, file_address: str, channel_index: int = 0, c_eff: float = c,
                 waveform_fft: Optional[NDArray] = None, slow_time_weighting: Optional[NDArray] = None):
        """
        Class for representing a dataset stored as a CPHD 0.3 file. Allows lazy loading from file.
        Parameters
        ----------
        file_address : str
            Address of file in filesystem.
        channel_index : int
            CPHD channel data will be drawn from. Defaults to the 0th channel.
        c_eff : float
            Effective speed of light to be used for this dataset, in metres/second. Defaults to vacuum value.
        waveform_fft : NDArray
            Waveform FFT used for measurement. Defaults to uniform across band.
        slow_time_weighting : NDArray
            Slow time weighting applied to measurement. Defaults to unit uniform.
        """
        self._file_address = file_address
        self._reader = open_phase_history(self._file_address)
        self._channel_index = channel_index
        self._waveform_fft = waveform_fft
        self._slow_time_weighting = slow_time_weighting
        self._c_eff = c_eff
        # Not sure what's going on here...
        self._num_slow_times = self.reader.cphd_meta.Data.Channels[self._channel_index].NumVectors
        if self.transmitter_positions.shape[0] < self._num_slow_times:
            self._num_slow_times = self.transmitter_positions.shape[0]
        self._num_freqs = self.reader.cphd_meta.Data.Channels[self._channel_index].NumSamples
        sign = self.reader.cphd_meta.Global.SGN
        if sign > 0:
            self._sign_multiplier = 1.0
        else:
            self._sign_multiplier = -1.0

    @property
    def reader(self) -> CPHDTypeReader:
        """
        A sarpy CPHD reader reading from the datafile.
        Returns
        -------
        reader : CPHDTypeReader
            Reader for this dataset. Note that a new one is summoned on demand if one isn't currently instantiated.
        """
        if self._reader is None:
            self._reader = open_phase_history(self._file_address)
        return self._reader

    @property
    def aperture(self) -> NumpySimpleSarAperture:
        """
        Aperture associated with this dataset channel.
        Returns
        -------
        ap : NumpySimpleSarAperture
            Aperture associated with this dataset channel.
        """
        ap = NumpySimpleSarAperture(self.transmitter_positions, self.receiver_positions, self.stab_ref_posns,
                                    self.centre_frequency, self.sample_bandwidth, self.waveform_fft,
                                    self.slow_time_weighting, num_freqs=None, sign_multiplier=self.sign_multiplier,
                                    c_eff=self.c_eff)
        return ap

    @property
    def num_slow_times(self) -> int:
        """
        Number of pulses in dataset.
        Returns
        -------
        num_slow_times : int
            Number of slow times in dataset.
        """
        return self._num_slow_times

    @property
    def num_freqs(self) -> int:
        """
        Number of frequencies sampled in dataset.
        Returns
        -------
        num_freqs : int
            Number of frequencies sampled in dataset.
        """
        return self._num_freqs

    @property
    def centre_frequency(self) -> float:
        """
        Centre frequency of dataset.
        Returns
        -------
        centre_frequency : float
            Centre frequency of dataset, in Hz.
        """
        f0 = self.reader.read_pvp_variable("SC0", self._channel_index, 0)[0]
        delta_f = self.reader.read_pvp_variable("SCSS", self._channel_index, 0)[0]
        samp_freq = self.num_freqs * delta_f
        return f0 + 0.5 * samp_freq

    @property
    def sample_bandwidth(self) -> float:
        """
        Sample rate used for dataset.
        Returns
        -------
        samp_freq : float
            Sample rate used for dataset, in Hz.
        """
        delta_f = self.reader.read_pvp_variable("SCSS", self._channel_index, 0)[0]
        samp_freq = self.num_freqs * delta_f
        return samp_freq

    @property
    def transmitter_positions(self) -> NDArray:
        """
        Transmitter positions.
        Returns
        -------
        trans_posns : NDArray
            [num_slow_times, 3] real float-valued array of the transmitter positions, in metres.
        """
        return self.reader.read_pvp_variable("TxPos", self._channel_index, slice(None))

    @property
    def receiver_positions(self) -> NDArray:
        """
        Receiver positions.
        Returns
        -------
        rec_posns : NDArray
            [num_slow_times, 3] real float-valued array of the receiver positions, in metres.
        """
        return self.reader.read_pvp_variable("RcvPos", self._channel_index, slice(None))

    @property
    def stab_ref_posns(self) -> NDArray:
        """
        Stabilisation reference points.
        Returns
        -------
        srps : NDArray
            [num_slow_times, 3] real float-valued array of the SRPs, in metres.
        """
        return self.reader.read_pvp_variable("SRPPos", self._channel_index, slice(None))

    @property
    def waveform_fft(self) -> NDArray:
        """
        FFT of waveform used.
        Returns
        -------
        waveform_fft : NDArray
            FFT of waveform used. [num_freqs] complex NDArray.
        """
        if self._waveform_fft is None:
            self._waveform_fft = np.ones(self.num_freqs, dtype=np.complex128)
        return self._waveform_fft

    @property
    def slow_time_weighting(self) -> NDArray:
        """
        Slow time weighting used.
        Returns
        -------
        slow_time_weighting : NDArray
            Slow time weighting used. [num_slow_times] complex NDArray.
        """
        if self._slow_time_weighting is None:
            self._slow_time_weighting = np.ones(self.num_slow_times, dtype=np.complex128)
        return self._slow_time_weighting

    @property
    def sign_multiplier(self) -> float:
        """
        Sign convention used for modulation/demodulation circuits.
        Returns
        -------
        sign_multiplier : float
            Sign convention for modulation/demodulation in this measurement. Should be -1.0 or +1.0.
        """
        return self._sign_multiplier

    @property
    def c_eff(self) -> float:
        """
        Effective speed of light for this measurement.
        Returns
        -------
        c_eff : float
            Effective speed of light for this measurement, in metres/second.
        """
        return self._c_eff

    def to_blocks(self, num_blocks: int = 1) -> List["NumpySimpleSarDataset"]:
        """
        Chops dataset into num_blocks blocks of roughly equal size. Useful for reducing max RAM usage.
        Parameters
        ----------
        num_blocks : int
            Number of blocks we want to end up with.
        Returns
        -------
        blocks : List[NumpySimpleSarDataset]
            List of num_blocks NumpySimpleSarDatasets containing blocks of the original dataset.
        """
        if num_blocks == 1:
            return [NumpySimpleSarDataset.from_scratch(trans_posns=self.transmitter_positions,
                                                       rec_posns=self.receiver_positions, srps=self.stab_ref_posns,
                                                       centre_freq=self.centre_frequency,
                                                       sample_freq=self.sample_bandwidth, c_eff=self.c_eff,
                                                       waveform_fft=self.waveform_fft, num_freqs=self.num_freqs,
                                                       sign_multiplier=self.sign_multiplier,
                                                       slow_time_weighting=self.slow_time_weighting,
                                                       phase_history=None)]
        block_length = self.num_slow_times // num_blocks
        output = []
        for block_index in range(num_blocks):
            start_index = block_length * block_index
            end_index = start_index + block_length
            if block_index == num_blocks - 1:
                end_index = self.num_slow_times
            measurement = NumpySimpleSarDataset.from_scratch(
                trans_posns=self.transmitter_positions[start_index:end_index, :],
                rec_posns=self.receiver_positions[start_index:end_index, :],
                srps=self.stab_ref_posns[start_index:end_index, :],
                centre_freq=self.centre_frequency,
                sample_freq=self.sample_bandwidth,
                waveform_fft=self.waveform_fft, num_freqs=self.num_freqs,
                sign_multiplier=self.sign_multiplier,
                slow_time_weighting=self.slow_time_weighting[start_index:end_index],
                phase_history=None)
            output.append(measurement)
        return output

    def phase_history(self, item: slice = slice(None)) -> NDArray:
        """
        Phase history for this dataset. Lazily retrieves the slices requested.
        Parameters
        ----------
        item : slice
            The python slice we want data for. Defaults to slice(None), which gives us everything.
        Returns
        -------
        phase_history : Optional[NDArray]
            [num_requested, num_freqs] complex-valued NDArray containing requested phase history. Could return
            None if there's an issue with the reader, but this shouldn't happen when loading a valid file.
        """
        return self.reader.read(item, index=self._channel_index)

    def __getstate__(self):
        state = self.__dict__
        state["_reader"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reader = open_phase_history(self._file_address)

    def get_subset(self, item: slice = None) -> "NumpySimpleSarDataset":
        """
        Get a slow-time subset of this dataset.
        Parameters
        ----------
        item : slice
            Python slice indicating which slow times we'd like to grab.
        Returns
        -------
        data_subset : NumpySimpleSarDataset
            NumpySimpleSarDataset containing the slow times we requested.
        """
        return NumpySimpleSarDataset(self.aperture.get_subset(item), self.phase_history(item))
