import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pymap3d import ecef2geodetic, geodetic2ecef, enu2ecef
from scipy.constants import c, pi

from .k_space import OrientedBoundingBox
from ..base import BaseSimpleSarAperture


def normalise(x: NDArray, tol=1.0E-12) -> NDArray:
    """
    Returns a copy of an array which is Euclidean-normalised in the last axis.
    Parameters
    ----------
    x : NDArray
        Array to be normalised.
    tol : float
        Size of smallest vector we wish to normalise. Error will be thrown if this function is used on a vector smaller
        than this. Defaults to 1.0E-12. Normalisation doesn't make sense for numerically small vectors, don't do it!

    Returns
    -------
    out : NDArray
        Array of normalised vectors.
    """
    x_norm = np.linalg.norm(x, axis=-1)
    if x_norm < tol:
        raise ValueError("Shouldn't normalise vectors of zero/negligible length!")
    return x / x_norm


def move_srps(phase_history: NDArray, old_srps: NDArray, new_srp: NDArray, trans_posns: NDArray, rec_posns: NDArray,
              centre_freq: float, sample_bandwidth: float, sign_factor: float = 1.0, c_eff: float = c) -> NDArray:
    """
    Copies a phase history array, applying phase ramps to account for changes in stabilisation reference points (SRPs).
    Useful for manipulating datasets, especially when they're going to be resampled.
    Parameters
    ----------
    phase_history : NDArray
        Complex-valued phase history array, of shape [num_slow_times, num_freqs].
    old_srps : NDArray
        [num_slow_times, 3] real-valued array of the old SRPs we were using, in metres.
    new_srp : NDArray
        [num_slow_times, 3] real-valued array of the new SRPs we want to use, in metres.
    trans_posns : NDArray
        [num_slow_times, 3] real-valued array of the transmitter positions, in metres.
    rec_posns : NDArray
        [num_slow_times, 3] real-valued array of the receiver positions, in metres.
    centre_freq : float
        Centre frequency of the measurements, in Hz.
    sample_bandwidth : float
        Sample rate of the measurements, in Hz.
    sign_factor : float
        Sign convention used for modulation/demodulation. Should be -1.0 or +1.0.
    c_eff : float
        The radiation speed used, in metres/second. Defaults to standard vacuum value.

    Returns
    -------
    new_phase_history : NDArray
        Copy of the original phase history, rephased to use the new SRPs.
    """
    num_freqs = phase_history.shape[1]
    new_srp = np.expand_dims(new_srp, axis=0)
    new_range = np.linalg.norm(trans_posns - new_srp, axis=-1) + np.linalg.norm(rec_posns - new_srp, axis=-1)
    old_range = np.linalg.norm(trans_posns - old_srps, axis=-1) + np.linalg.norm(rec_posns - old_srps, axis=-1)
    delta_range = np.expand_dims(new_range - old_range, axis=-1)
    baseband_freqs = np.fft.fftshift(np.fft.fftfreq(num_freqs, 1.0 / sample_bandwidth))
    baseband_ks = np.expand_dims((2.0 * pi / c_eff) * baseband_freqs, axis=0)
    range_phase_ramps = np.exp(1.j * (baseband_ks * delta_range))
    centre_modulation = np.exp((-2.0j * pi * sign_factor * centre_freq / c_eff) * delta_range)
    return phase_history * range_phase_ramps * centre_modulation


def ecefs_to_geodetics(x_ecef: NDArray, num_threads: int = -2) -> NDArray:
    """
    Function to accelerate the use of pymap3d.ecef2geodetic using joblib.
    Parameters
    ----------
    x_ecef : NDArray
        [num_coords, 3] real-valued array of ECEF coordinates.
    num_threads : int
        Number of threads to parallelise over.

    Returns
    -------
    x_geodetic : NDArray
        [num_coords, 3] real-valued array of corresponding geodetic coordinates.
    """

    def wrap_coord_change(x: NDArray) -> NDArray:
        return np.array(ecef2geodetic(*x))

    results = (Parallel(n_jobs=num_threads)([delayed(wrap_coord_change)(x) for x in x_ecef]))
    return np.stack(results, axis=0)


def scene_params_enu(scene_centre_geodetic: NDArray) -> tuple[NDArray, NDArray]:
    """
    Makes the scene centre and orientation matrix for a scene oriented with a local ENU coordinate system.
    Parameters
    ----------
    scene_centre_geodetic : NDArray
        Desired scene centre, as a [3]-shape real-valued array, in geodetic coordinates. Avoid coordinate singularities
        for that system!

    Returns
    -------
    centre_ecef : NDArray
        Scene centre translated into ECEF coordinates, as a [3]-shape real-valued array.
    rotation_matrix : NDArray
        NDArray of matrix components for the rotation which takes us from base ECEF to this local ENU-aligned
        coordinate system.
    """
    centre_ecef = np.array(geodetic2ecef(*scene_centre_geodetic))
    up = np.array(enu2ecef(0.0, 0.0, 1.0, *scene_centre_geodetic)) - centre_ecef
    north = np.array(enu2ecef(0.0, 1.0, 0.0, *scene_centre_geodetic)) - centre_ecef
    east = np.array(enu2ecef(1.0, 0.0, 0.0, *scene_centre_geodetic)) - centre_ecef
    rotation_matrix = np.stack([east, north, up], axis=-1)
    return centre_ecef, rotation_matrix


def scene_params_classic_sar(scene_centre_ecef: NDArray, aperture: BaseSimpleSarAperture,
                             tol: float = 1.0E-6) -> tuple[NDArray, NDArray, NDArray]:
    """
    Function to generate scene parameters for a "classic SAR" scenario. This means, a scene which is oriented in
    alignment with the local range, cross-range and out-of-plane axes at the scene centre. Appropriate Nyquist-compliant
    spatial sample rates are also calculated, as well as cosine factors to aid interpretation in terms of the ground
    plane.
    Parameters
    ----------
    scene_centre_ecef : NDArray
        Desired scene centre, in ECEF coordinates.
    aperture : BaseSimpleSarAperture
        Aperture we're going to induce the scene from (range, cross-range and out-of-plane don't exist as concepts
        without a measurement aperture in mind).
    tol : float
        Tolerance used to detect if we're too close to the ECEF origin, in metres.

    Returns
    -------
    rot_mat : NDArray
        Rotation matrix components to take us from the standard ECEF axes to the range, cross-range, out-of-plane
        axes.
    sample_rates : NDArray
        Spatial sample rates required for Nyquist-compliance in each axis, in metres^(-1)
    ground_cosines : NDArray
        Cosine weighting factor to be used when rescaling slant-plane imagery to the ground plane.
    """
    if np.linalg.norm(scene_centre_ecef) > tol:
        scene_centre_geodetic = np.array(ecef2geodetic(*scene_centre_ecef))
        up = np.array(enu2ecef(0.0, 0.0, 1.0, *scene_centre_geodetic)) - scene_centre_ecef
    else:
        up = np.array([0.0, 0.0, 1.0])
    up = normalise(up)
    obb = OrientedBoundingBox(aperture.edge_k_vecs())
    range_displacement = obb.obb_basis.T[obb.range_index]
    range_displacement = normalise(range_displacement)
    cross_range_displacement = obb.obb_basis.T[obb.cross_range_index]
    cross_range_displacement = normalise(cross_range_displacement)
    out_of_plane_displacement = obb.obb_basis.T[obb.out_of_plane_index]
    out_of_plane_displacement = normalise(out_of_plane_displacement)
    if np.dot(out_of_plane_displacement, up) < 0.0:
        out_of_plane_displacement = -out_of_plane_displacement
        cross_range_displacement = -cross_range_displacement
    rot_mat = np.stack([range_displacement, cross_range_displacement, out_of_plane_displacement], axis=0)
    ground_cosines = np.array([np.sqrt(1.0 - np.dot(range_displacement, up) ** 2),
                               np.sqrt(1.0 - np.dot(cross_range_displacement, up) ** 2),
                               np.dot(out_of_plane_displacement, up)]) / (2.0 * pi)
    sample_rates = obb.spans / (2.0 * pi)
    return rot_mat.T, sample_rates, ground_cosines
