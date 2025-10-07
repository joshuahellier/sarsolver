from os.path import join
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from numpy.random import Generator, PCG64
from pymap3d import geodetic2ecef
from pytest import fixture

from sarsolver.cil import CilSarGridGeometry, SimpleCilSarGeometry, ExtendedDataContainer, CpuSarOperator
from sarsolver.io import Cphd101Dataset
from sarsolver.numpy import GridSarScene, NumpySimpleSarOperator, NumpySimpleSarDataset, NumpySimpleSarAperture

rng = Generator(PCG64())

cphd_url = ("http://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/Tanna%20Island%2C%20Vanuatu/"
            "79b0967e-d2b2-4335-bb37-c3aa3dfd9211/2023-09-11-10-37-05_UMBRA-05/2023-09-11-10-37-05_UMBRA-05_CPHD.cphd")

vanuatu_centre_geodetic = np.array([-19.528445, 169.447415, 300.0])
vanuatu_centre_ecef = np.array(geodetic2ecef(*vanuatu_centre_geodetic))
this_file = Path(__file__)


@fixture()
def vanuatu_dataset() -> NumpySimpleSarDataset:
    print("Downloading small CC by 4.0-compliant CPHD file from Umbra for testing.")
    output_path, http_message = urlretrieve(cphd_url)
    print(http_message)
    print("CPHD file retrieved.")
    full_dataset = Cphd101Dataset(str(output_path))
    reduced_dataset = NumpySimpleSarDataset.from_resampling(full_dataset, 150.0, 100.0,
                                                            new_srp=vanuatu_centre_ecef)
    return reduced_dataset


@fixture()
def vanuatu_scene(vanuatu_dataset) -> GridSarScene:
    return GridSarScene.from_aperture(vanuatu_dataset.aperture, vanuatu_centre_ecef, [0.0, 100.0, 100.0])


@fixture()
def vanuatu_operator(vanuatu_scene, vanuatu_dataset) -> NumpySimpleSarOperator:
    return NumpySimpleSarOperator(vanuatu_scene, vanuatu_dataset.aperture)


@fixture()
def little_aperture() -> NumpySimpleSarAperture:
    num_slow_times = 249
    num_fast_times = 251
    centre_freq = 10.0E9  # Hz
    sample_freq = 300.0E6  # Hz
    standoff_range = 2000.0  # m
    track_length = 60.0  # m

    transmit_posns = np.linspace(np.array([standoff_range, -0.5 * track_length, 0.0]),
                                 np.array([standoff_range, 0.5 * track_length, 0.0]), num_slow_times)  # m
    receive_posns = np.linspace(np.array([standoff_range, -0.5 * track_length, 0.0]),
                                np.array([standoff_range, 0.5 * track_length, 0.0]), num_slow_times)  # m
    srps = np.zeros([num_slow_times, 3])  # m

    aperture = NumpySimpleSarAperture(trans_posns=transmit_posns, rec_posns=receive_posns, srps=srps,
                                      centre_freq=centre_freq, sample_freq=sample_freq, num_freqs=num_fast_times)
    return aperture


@fixture()
def little_scene(little_aperture) -> GridSarScene:
    side_x = 40.0
    side_y = 40.0
    scene = GridSarScene.from_aperture(little_aperture, np.array([0.0, 0.0, 0.0]), np.array([0.0, side_y, side_x]),
                                       safety_factor=3.0)
    return scene


@fixture()
def little_scene_geometry(little_scene) -> CilSarGridGeometry:
    return CilSarGridGeometry(little_scene)


@fixture()
def little_measurement_geometry(little_aperture) -> SimpleCilSarGeometry:
    return SimpleCilSarGeometry(little_aperture)


@fixture()
def little_ground_truth(little_scene_geometry) -> ExtendedDataContainer:
    gt_container = little_scene_geometry.allocate()
    gt_container.array = np.load(join(this_file.parent, "artifacts", "little_gt.npy"))
    return gt_container


@fixture
def little_sar_operator(little_scene_geometry, little_measurement_geometry) -> CpuSarOperator:
    op = CpuSarOperator(little_scene_geometry, little_measurement_geometry)
    op._norm = 756.6231381576064
    return op


@fixture()
def little_backproj(little_scene_geometry) -> ExtendedDataContainer:
    backproj_container = little_scene_geometry.allocate()
    backproj_container.array = np.load(join(this_file.parent, "artifacts", "little_backproj.npy"))
    return backproj_container


@fixture()
def little_soln(little_scene_geometry) -> ExtendedDataContainer:
    soln_container = little_scene_geometry.allocate()
    soln_container.array = np.load(join(this_file.parent, "artifacts", "little_soln.npy"))
    return soln_container


@fixture()
def little_synthetic_measurement(little_measurement_geometry) -> ExtendedDataContainer:
    ph_container = little_measurement_geometry.allocate()
    ph_container.array = np.load(join(this_file.parent, "artifacts", "little_noisy_meas.npy"))
    return ph_container
