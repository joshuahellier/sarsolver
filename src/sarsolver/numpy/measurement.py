from itertools import batched
from typing import List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.constants import c, pi

from ..base import BaseSimpleSarAperture, BaseSimpleSarDataset
from ..utils import move_srps, get_downsample_kernel, fast_freq_downsample, OrientedBoundingBox, default_num_threads


class NumpySimpleSarAperture(BaseSimpleSarAperture):
    def __init__(self, trans_posns: NDArray, rec_posns: NDArray, srps: NDArray, centre_freq: float,
                 sample_freq: float, waveform_fft: Optional[NDArray] = None,
                 slow_time_weighting: Optional[NDArray] = None, num_freqs: Optional[int] = None,
                 sign_multiplier: float = 1.0, c_eff: float = c):
        """
        Class representing a bistatic scalar SAR data collection manifold.
        Parameters
        ----------
        trans_posns : NDArray
            [num_slow_times, 3] real-valued array of transmitter positions, in metres.
        rec_posns : NDArray
            [num_slow_times, 3] real-valued array of receiver positions, in metres.
        srps : NDArray
            [num_slow_times, 3] real-valued array of stabilisation reference positions, in metres.
        centre_freq : float
            Centre frequency of the radiation used, in Hz.
        sample_freq : float
            Sampling rate of the measurement, in Hz.
        waveform_fft : NDArray
            [num_fast_times] complex-valued array, intended to be the Fourier transform of the waveform. Defaults to
            an array of 1s.
        slow_time_weighting : NDArray
            [num_slow_times] complex-valued array, intended to represent some weighting in slow time (due to,
            apodisation, simple beam weighting, etc).
        num_freqs : int
            Number of samples in fast time/frequency. Can be inferred from waveform_fft if given.
        sign_multiplier : float
            Sign convention used for modulation/demodulation. Should be -1.0 or +1.0/
        c_eff : float
            The radiation speed used. Defaults to standard vacuum value.
        """
        self._stab_ref_posns = srps
        self._slow_time_weighting = slow_time_weighting
        self._sample_bandwidth = sample_freq
        self._centre_freq = centre_freq
        self._num_freqs = num_freqs
        self._sign_multiplier = sign_multiplier
        self._transmitter_positions = trans_posns
        self._c_eff = c_eff
        if waveform_fft is None:
            if num_freqs is None:
                raise ValueError("Number of frequencies needs to be set somehow")
            waveform_fft = np.ones(num_freqs, dtype=np.complex128)
        self._num_freqs = waveform_fft.shape[0]
        self._waveform_fft = waveform_fft
        self._receiver_positions = rec_posns
        self._num_slow_times = trans_posns.shape[0]
        if slow_time_weighting is None:
            self._slow_time_weighting = np.ones(self._num_slow_times, dtype=np.complex128)

    def to_blocks(self, num_blocks: int = 1) -> List["NumpySimpleSarAperture"]:
        """
        Chops this aperture into num_blocks continuous subapertures. Used in calculation.
        Parameters
        ----------
        num_blocks : int
            Number of roughly-equal subapertures we want to end up with.

        Returns
        -------
        subapertures : List[NumpySimpleSarAperture]
            List of subapertures produced by cutting this aperture into roughly equal pieces.
        """
        if num_blocks == 1:
            return [NumpySimpleSarAperture(trans_posns=self.transmitter_positions,
                                           rec_posns=self.receiver_positions,
                                           srps=self.stab_ref_posns,
                                           centre_freq=self.centre_frequency,
                                           sample_freq=self.sample_bandwidth,
                                           waveform_fft=self.waveform_fft, num_freqs=self.num_freqs,
                                           sign_multiplier=self.sign_multiplier,
                                           slow_time_weighting=self.slow_time_weighting)]
        block_length = self.num_slow_times // num_blocks
        output = []
        for block_index in range(num_blocks):
            start_index = block_length * block_index
            end_index = start_index + block_length
            if block_index == num_blocks - 1:
                end_index = self.num_slow_times
            # Note that this blocking process is for the measurement manifold, NOT the phase history!
            measurement = NumpySimpleSarAperture(trans_posns=self.transmitter_positions[start_index:end_index, :],
                                                 rec_posns=self.receiver_positions[start_index:end_index, :],
                                                 srps=self.stab_ref_posns[start_index:end_index, :],
                                                 centre_freq=self.centre_frequency,
                                                 sample_freq=self.sample_bandwidth,
                                                 waveform_fft=self.waveform_fft, num_freqs=self.num_freqs,
                                                 sign_multiplier=self.sign_multiplier,
                                                 slow_time_weighting=self.slow_time_weighting[start_index:end_index])
            output.append(measurement)
        return output

    @property
    def sample_bandwidth(self) -> float:
        """
        Sample rate used for dataset.
        Returns
        -------
        samp_freq : float
            Sample rate used for dataset.
        """
        return self._sample_bandwidth

    @property
    def waveform_fft(self) -> NDArray:
        """
        FFT of waveform used.
        Returns
        -------
        waveform_fft : NDArray
            FFT of waveform used. [num_freqs] complex NDArray.
        """
        return self._waveform_fft

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
    def stab_ref_posns(self) -> NDArray:
        """
        Stabilisation reference points.
        Returns
        -------
        srps : NDArray
            [num_slow_times, 3] real float-valued array of the SRPs, in metres.
        """
        return self._stab_ref_posns

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
    def c_eff(self) -> float:
        """
        Effective speed of light for this measurement.
        Returns
        -------
        c_eff : float
            Effective speed of light for this measurement, , in metres/second.
        """
        return self._c_eff

    @property
    def centre_frequency(self) -> float:
        """
        Centre frequency of dataset.
        Returns
        -------
        centre_frequency : float
            Centre frequency of dataset, in Hz.
        """
        return self._centre_freq

    @property
    def transmitter_positions(self) -> NDArray:
        """
        Transmitter positions.
        Returns
        -------
        trans_posns : NDArray
            [num_slow_times, 3] real float-valued array of the transmitter positions, in metres.
        """
        return self._transmitter_positions

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
    def receiver_positions(self) -> NDArray:
        """
        Receiver positions.
        Returns
        -------
        rec_posns : NDArray
            [num_slow_times, 3] real float-valued array of the receiver positions, in metres.
        """
        return self._receiver_positions

    @property
    def slow_time_weighting(self) -> NDArray:
        """
        Slow time weighting used.
        Returns
        -------
        slow_time_weighting : NDArray
            Slow time weighting used. [num_slow_times] complex NDArray.
        """
        return self._slow_time_weighting

    def get_subset(self, item: slice = None) -> "NumpySimpleSarAperture":
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
        return NumpySimpleSarAperture(self.transmitter_positions[item, :], self.receiver_positions[item, :],
                                      self.stab_ref_posns[item, :], self.centre_frequency,
                                      self.sample_bandwidth, self.waveform_fft, self.slow_time_weighting[item],
                                      sign_multiplier=self.sign_multiplier, c_eff=self.c_eff)


class NumpySimpleSarDataset(BaseSimpleSarDataset):
    def __init__(self, aperture: BaseSimpleSarAperture, phase_history: NDArray):
        """
        Combines an existing aperture with a phase history to make a measurement dataset.
        Parameters
        ----------
        aperture : BaseSimpleSarAperture
            The aperture for the measurement.
        phase_history : NDArray
            [num_slow_times, num_freqs] complex-valued array of the relevant phase history.
        """
        self._phase_history = phase_history
        self._aperture = aperture

    @staticmethod
    def from_scratch(trans_posns: NDArray, rec_posns: NDArray, srps: NDArray, centre_freq: float,
                     sample_freq: float, phase_history: Optional[NDArray] = None,
                     waveform_fft: Optional[NDArray] = None,
                     slow_time_weighting: Optional[NDArray] = None, num_freqs: Optional[int] = None,
                     sign_multiplier: float = 1.0, c_eff: float = c) -> "NumpySimpleSarDataset":
        """
        Alternative constructor to make a dataset and aperture simultaneously using the arguments the aperture would've
        had.
        Parameters
        ----------
        trans_posns : NDArray
            [num_slow_times, 3] real-valued array of transmitter positions, in metres.
        rec_posns : NDArray
            [num_slow_times, 3] real-valued array of receiver positions, in metres.
        srps : NDArray
            [num_slow_times, 3] real-valued array of stabilisation reference positions, in metres.
        centre_freq : float
            Centre frequency of the radiation used, in Hz.
        sample_freq : float
            Sampling rate of the measurement, in Hz.
        phase_history : Optional[NDArray]
            Phase history for this dataset. Defaults to None. If used, should be a [num_slow_times, num_freqs]
            complex-valued array.
        waveform_fft : NDArray
            [num_fast_times] complex-valued array, intended to be the Fourier transform of the waveform. Defaults to
            an array of 1s.
        slow_time_weighting : NDArray
            [num_slow_times] complex-valued array, intended to represent some weighting in slow time (due to,
            apodisation, simple beam weighting, etc).
        num_freqs : int
            Number of samples in fast time/frequency. Can be inferred from waveform_fft if given.
        sign_multiplier : float
            Sign convention used for modulation/demodulation. Should be -1.0 or +1.0.
        c_eff : float
            The radiation speed used, in metres/second. Defaults to standard vacuum value.

        Returns
        -------
        dataset : NumpySimpleSarDataset
            Dataset made from the ingredients specified in the function arguments.
        """
        aperture = NumpySimpleSarAperture(trans_posns=trans_posns, rec_posns=rec_posns, srps=srps,
                                          centre_freq=centre_freq, sample_freq=sample_freq, waveform_fft=waveform_fft,
                                          slow_time_weighting=slow_time_weighting, num_freqs=num_freqs,
                                          sign_multiplier=sign_multiplier, c_eff=c_eff)
        return NumpySimpleSarDataset(aperture, phase_history)

    def phase_history(self, item: slice = slice(None)) -> Optional[NDArray]:
        """
        Phase history for this dataset. Lazily retrieves the slices requested.
        Parameters
        ----------
        item : slice
            The python slice we want data for. Defaults to slice(None), which gives us everything.
        Returns
        -------
        phase_history : Optional[NDArray]
            [num_requested, num_freqs] complex-valued NDArray containing requested phase history, if present.
            Returns None is not present.
        """
        if self._phase_history is None:
            return None
        return self._phase_history[item]

    @property
    def aperture(self) -> BaseSimpleSarAperture:
        """
        Aperture associated with this dataset.
        Returns
        -------
        ap : NumpySimpleSarAperture
            Aperture associated with this dataset.
        """
        return self._aperture

    def remove_phase_history(self):
        """
        Method for disposing of unwanted (RAM-hungry) phase history. Doesn't explicitly delete, but should unreference,
        which is more Pythonic.
        """
        self._phase_history = None

    @staticmethod
    def from_resampling(original_dataset: BaseSimpleSarDataset, range_amb: Optional[float] = None,
                        cross_range_amb: Optional[float] = None, fast_downsample_factor: float = 1.0,
                        slow_downsample_factor: int = 1, new_srp: Optional[NDArray] = None,
                        safety_factor: float = 1.2, num_threads: Optional[int] = None) -> "NumpySimpleSarDataset":
        """
        Alternative constructor which makes a new SAR dataset by (lazily) resampling an existing one, usually to produce
        something smaller. This works very well for standard satellite SAR collection, where pulses are taken
        sequentially at evenly-spaced times along an almost-straight flightpath; it won't work so well for arbitrary
        datasets! Use when sensible.

        Parameters
        ----------
        original_dataset : BaseSimpleSarDataset
            An existing SAR dataset.
        range_amb : float
            Ambiguous range in metres we want after resampling. Radar returns originating from outside this should be heavily
             suppressed. Leave as None to set downsample factor manually.
        cross_range_amb : float
            Ambiguous cross range in metres we want after resampling. Radar returns originating from outside this should be
            heavily suppressed. Leave as None to set downsample factor manually.
        fast_downsample_factor : float
            Factor by which we will resample in fast time. Defaults to 1.0, but will be altered if range_amb is set.
        slow_downsample_factor : float
            Factor by which we will resample in fast time. Defaults to 1.0, but will be altered if range_amb is set.
        new_srp : NDArray
            [3]-shape real-valued float array indicating new SRP, in metres. The downsampling will be done relative to
            this point, so place it in the middle of the scene you wish to cut down to.
        safety_factor : float
            Factor by which we want to oversample when downsampling. Defaults to 1.2, giving 20% more range and
            cross-range than indicated, in order to ensure proper preservation of data in the zone of interest.
        num_threads : int
            Number of threads we wish to use for downsampling process.

        Returns
        -------
        dataset : NumpySimpleSarDataset
            Dataset of downsampled data.
        """
        if num_threads is None:
            num_threads = default_num_threads()
        ap = original_dataset.aperture
        if new_srp is None:
            new_srp = ap.stab_ref_posns[0]
        k_space_obb = OrientedBoundingBox(ap.edge_k_vecs(new_srp))
        current_range_amb = 2.0 * pi * ap.num_freqs / k_space_obb.range_span
        current_cross_range_amb = 2.0 * pi * ap.num_slow_times / k_space_obb.cross_range_span
        if cross_range_amb is not None:
            slow_downsample_factor = int(current_cross_range_amb / (2.0 * safety_factor * cross_range_amb))
        if slow_downsample_factor < 1:
            slow_downsample_factor = 1.0
        if range_amb is not None:
            fast_downsample_factor = current_range_amb / (2.0 * safety_factor * range_amb)
        old_num_freqs = ap.num_freqs
        new_num_freqs = int(1.0 + old_num_freqs / fast_downsample_factor)
        if new_num_freqs > old_num_freqs:
            new_num_freqs = old_num_freqs
        downsample_kernel = np.expand_dims(get_downsample_kernel(slow_downsample_factor), axis=-1)
        kernel_length = downsample_kernel.shape[0]
        new_num_slow_times = ((ap.num_slow_times - kernel_length) // slow_downsample_factor) - 1
        new_waveform_fft = fast_freq_downsample(np.expand_dims(ap.waveform_fft, axis=0),
                                                new_num_freqs=new_num_freqs, safety_factor=safety_factor)[0, :]
        bad_indices = ap.bad_slow_times

        new_slow_indices = range(new_num_slow_times)

        def partial_downsample(slow_time_index_subset: Sequence[NDArray]):
            partial_trans_list = []
            partial_rec_list = []
            partial_srp_list = []
            partial_slow_weighting_list = []
            partial_phase_history_list = []
            in_ap = original_dataset.aperture

            for new_slow_index in slow_time_index_subset:
                old_slow_index = slow_downsample_factor * new_slow_index
                old_slow_indices = list(range(old_slow_index, old_slow_index + kernel_length))
                bad_index_detected = len(set(old_slow_indices) & set(bad_indices)) > 0
                if not bad_index_detected:
                    partial_trans_list.append(in_ap.transmitter_positions[old_slow_index + kernel_length // 2, :])
                    partial_rec_list.append(in_ap.receiver_positions[old_slow_index + kernel_length // 2, :])
                    partial_srp_list.append(new_srp)
                    old_phase_history = original_dataset.phase_history(
                        slice(old_slow_index, old_slow_index + kernel_length, 1))
                    shifted_ph = move_srps(old_phase_history,
                                           in_ap.stab_ref_posns[old_slow_index:old_slow_index + kernel_length, :],
                                           new_srp,
                                           in_ap.transmitter_positions[old_slow_index:old_slow_index + kernel_length,
                                           :],
                                           in_ap.receiver_positions[old_slow_index:old_slow_index + kernel_length, :],
                                           in_ap.centre_frequency, in_ap.sample_bandwidth, in_ap.sign_multiplier,
                                           in_ap.c_eff)
                    ds_ph = fast_freq_downsample(shifted_ph, new_num_freqs, safety_factor)
                    old_slow_weighting = in_ap.slow_time_weighting[old_slow_index:old_slow_index + kernel_length]
                    new_freq_profile = np.sum(ds_ph * downsample_kernel, axis=0)
                    new_partial_slow_weighting = np.sum(downsample_kernel[:, 0] * old_slow_weighting)
                    partial_slow_weighting_list.append(new_partial_slow_weighting)
                    partial_phase_history_list.append(new_freq_profile)

            return (np.stack(partial_trans_list, axis=0), np.stack(partial_rec_list, axis=0),
                    np.stack(partial_srp_list, axis=0), np.array(partial_slow_weighting_list),
                    np.stack(partial_phase_history_list, axis=0))

        new_slow_index_subsets = list(batched(new_slow_indices, int(1.0 + new_num_slow_times / num_threads)))
        new_vars = Parallel(n_jobs=num_threads)([delayed(partial_downsample)(slow_index_subset)
                                                 for slow_index_subset in new_slow_index_subsets])

        new_trans = np.concatenate([x[0] for x in new_vars], axis=0)
        new_rec = np.concatenate([x[1] for x in new_vars], axis=0)
        new_srps = np.concatenate([x[2] for x in new_vars], axis=0)
        new_slow_weighting = np.concatenate([x[3] for x in new_vars], axis=0)
        new_phase_history = np.concatenate([x[4] for x in new_vars], axis=0)

        return NumpySimpleSarDataset.from_scratch(trans_posns=new_trans, rec_posns=new_rec, srps=new_srps,
                                                  centre_freq=ap.centre_frequency, sample_freq=ap.sample_bandwidth,
                                                  waveform_fft=new_waveform_fft, slow_time_weighting=new_slow_weighting,
                                                  phase_history=new_phase_history, sign_multiplier=ap.sign_multiplier,
                                                  c_eff=ap.c_eff)

    def get_subset(self, item: slice = None) -> "NumpySimpleSarDataset":
        return NumpySimpleSarDataset(self.aperture.get_subset(item), self.phase_history(item))

    @staticmethod
    def from_datasets(datasets: Sequence[BaseSimpleSarDataset]) -> "NumpySimpleSarDataset":
        """
        Alternative constructor to make a dataset from a sequence of datasets.
        Parameters
        ----------
        datasets : Sequence[BaseSimpleSarDataset]
            Datasets we wish to concatenate (not in-place).

        Returns
        -------
        dataset : NumpySimpleSarDataset
            Dataset consisting of concatenated datasets.
        """
        ap = datasets[0].aperture
        sample_freq = ap.sample_bandwidth
        centre_freq = ap.centre_frequency
        sign_multiplier = ap.sign_multiplier
        c_eff = ap.c_eff
        waveform_fft = ap.waveform_fft
        partial_trans_list = []
        partial_rec_list = []
        partial_srp_list = []
        partial_slow_weighting_list = []
        partial_phase_history_list = []
        for dataset in datasets:
            ap = dataset.aperture
            partial_trans_list.append(ap.transmitter_positions)
            partial_rec_list.append(ap.receiver_positions)
            partial_srp_list.append(ap.stab_ref_posns)
            partial_slow_weighting_list.append(ap.slow_time_weighting)
            partial_phase_history_list.append(dataset.phase_history(slice(None)))
        trans_posns = np.concatenate(partial_trans_list, axis=0)
        rec_posns = np.concatenate(partial_rec_list, axis=0)
        srp_posns = np.concatenate(partial_srp_list, axis=0)
        slow_time_weighting = np.concatenate(partial_slow_weighting_list, axis=0)
        phase_history = np.concatenate(partial_phase_history_list, axis=0)
        return NumpySimpleSarDataset.from_scratch(trans_posns=trans_posns, rec_posns=rec_posns, srps=srp_posns,
                                                  centre_freq=centre_freq, sample_freq=sample_freq,
                                                  phase_history=phase_history, waveform_fft=waveform_fft,
                                                  slow_time_weighting=slow_time_weighting,
                                                  sign_multiplier=sign_multiplier, c_eff=c_eff)
