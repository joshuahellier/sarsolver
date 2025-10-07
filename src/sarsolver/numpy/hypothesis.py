from typing import List, Sequence, Optional

import numpy as np
from numpy.typing import NDArray

from ..base import BaseSarScene, BaseSimpleSarAperture
from ..utils import scene_params_enu, scene_params_classic_sar


class SimpleSarScene(BaseSarScene):
    """Class representing a scene comprised of point scatterers. No structure assumed."""

    def __init__(self, scatterer_positions: NDArray):
        """
        Class representing a scene comprised of point scatterers. No structure assumed.
        Parameters
        ----------
        scatterer_positions : NDArray
            [num_scatterers, 3] real-valued array of scatterer locations.
        """
        self._scatterer_positions = scatterer_positions
        self._num_scatterers = scatterer_positions.shape[0]

    @property
    def positions(self) -> NDArray:
        """
        Locations of the scatterers.
        Returns
        -------
        scatterer_positions : NDArray
            [num_scatterers, 3] real-valued array of scatterer locations.
        """
        return self._scatterer_positions

    @property
    def num_scatterers(self) -> int:
        """
        The number of scatterers.
        Returns
        -------
        num_scatterers: int
            Number of scatterers in scene.
        """
        return self._num_scatterers

    def to_blocks(self, num_blocks: int = 1) -> List["BaseSarScene"]:
        """
        Cuts scene into num_blocks blocks with similar numbers of scatterers. Useful for reducing RAM usage
        and enabling parallelisation.
        Parameters
        ----------
        num_blocks : int
            Number of blocks to cut into.
        Returns
        -------
        blocks : List[BaseSarScene]
            List of num_blocks scenes, which combined form the original scene.
        """
        if num_blocks == 1:
            return [self]
        block_length = self.num_scatterers // num_blocks
        output = []
        for block_index in range(num_blocks):
            start_index = block_length * block_index
            end_index = start_index + block_length
            if block_index == num_blocks - 1:
                end_index = self.num_scatterers
            scene_params = SimpleSarScene(scatterer_positions=self.positions[start_index:end_index, :])
            output.append(scene_params)
        return output


class GridSarScene(BaseSarScene):
    """
    Class representing a scene comprised of a regular grid of scatterers in 3D space.
    """

    def __init__(self, grid_shape: Sequence[int], grid_dimensions: Sequence[float],
                 grid_centre: Optional[NDArray] = None, rotation_matrix: Optional[NDArray] = None):
        """
        Class representing a scene comprised of a regular grid of scatterers in 3D space.
        Parameters
        ----------
        grid_shape : Sequence[int]
            Sequence of 3 integers indicating the shape of the grid (i.e. how many regularly-spaced planes of
            points are in it). Sequence should be in z, y, x order, i.e. C-array ordering, for consistency.
        grid_dimensions : Sequence[float]
            Sequence of 3 floats indicating the dimensions of the cuboid it lives in (i.e. the edge lengths).
            Points fil all the way up to the faces/edges/vertices. Ordering must be l_z, l_y, l_x, for consistency
            with image arrays that will eventually be produced using these classes.
        grid_centre : NDArray
            Length 3 real-valued array indicating the cartesian centre point of the scene's cuboid. Defaults to the origin.
        rotation_matrix : NDArray
            Orthogonal matrix indicating the orientation of the cuboid. It is the matrix through which an original
            cuboid aligned with the x, y, z axes has been rotated to become the desired cuboid. Defaults to the identity.
        """
        if grid_centre is None:
            grid_centre = np.array([0.0, 0.0, 0.0])
        self._grid_centre = grid_centre
        if rotation_matrix is None:
            rotation_matrix = np.identity(3)
        self._rotation_matrix = rotation_matrix
        self._grid_shape = grid_shape
        self._grid_dimensions = grid_dimensions

    def to_blocks(self, num_blocks: int = 1) -> List["BaseSarScene"]:
        """
        Cuts scene into num_blocks blocks with similar numbers of scatterers. Useful for reducing RAM usage
        and enabling parallelisation.
        Parameters
        ----------
        num_blocks : int
            Number of blocks to cut into.
        Returns
        -------
        blocks : List[BaseSarScene]
            List of num_blocks scenes, which combined form the original scene.
        """
        if num_blocks == 1:
            return [self]
        block_length = self.num_scatterers // num_blocks
        output = []
        for block_index in range(num_blocks):
            start_index = block_length * block_index
            end_index = start_index + block_length
            if block_index == num_blocks - 1:
                end_index = self.num_scatterers
            scene = SimpleSarScene(scatterer_positions=self.positions[start_index:end_index, :])
            output.append(scene)
        return output

    @property
    def num_scatterers(self) -> int:
        """
        The number of scatterers.
        Returns
        -------
        num_scatterers: int
            Number of scatterers in scene.
        """
        return np.prod(np.array(self.grid_shape))

    @property
    def rotation_matrix(self) -> NDArray:
        """
        Matrix indicating orientation of the scene cuboid.
        Returns
        -------
        rotation_matrix : NDArray
            Orthogonal matrix indicating the orientation of the cuboid. It is the matrix through which an original
            cuboid aligned with the x, y, z axes has been rotated to become the desired cuboid.
        """
        return self._rotation_matrix

    @property
    def grid_centre(self) -> NDArray:
        """
        Centre of the grid's cuboid.
        Returns
        -------
        grid_centre : NDArray
            Length 3 real-valued array indicating the cartesian centre point of the scene's cuboid.
        """
        return self._grid_centre

    @property
    def grid_dimensions(self) -> Sequence[float]:
        """
        Edge lengths of grid cuboid.
        Returns
        -------
        grid_dimensions : Sequence[float]
            Sequence of 3 floats indicating the dimensions of the cuboid it lives in (i.e. the edge lengths).
            Points fil all the way up to the faces/edges/vertices. Ordering must be l_z, l_y, l_x, for consistency
            with image arrays that will eventually be produced using these classes.
        """
        return self._grid_dimensions

    @property
    def grid_shape(self) -> Sequence[int]:
        """
        Shape of grid.
        Returns
        -------
        grid_shape: Sequence[int]
            Sequence of 3 integers indicating the shape of the grid (i.e. how many regularly-spaced planes of
            points are in it). Sequence should be in z, y, x order, i.e. C-array ordering, for consistency.
        """
        return self._grid_shape

    @property
    def positions(self) -> NDArray:
        """
        Locations of the scatterers.
        Returns
        -------
        scatterer_positions : NDArray
            [num_scatterers, 3] real-valued array of scatterer locations. Note the use of the rotation matrix,
            if there's any confusion about convention.
        """
        xs = np.linspace(-0.5 * self.grid_dimensions[2], 0.5 * self.grid_dimensions[2], self.grid_shape[2])
        ys = np.linspace(-0.5 * self.grid_dimensions[1], 0.5 * self.grid_dimensions[1], self.grid_shape[1])
        zs = np.linspace(-0.5 * self.grid_dimensions[0], 0.5 * self.grid_dimensions[0], self.grid_shape[0])
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        base_posns = np.stack([xx.reshape([-1]), yy.reshape([-1]), zz.reshape([-1])], axis=-1)
        return (self.rotation_matrix @ base_posns.T).T + np.expand_dims(self.grid_centre, axis=0)

    @staticmethod
    def from_geodetic_enu(grid_shape: Sequence[int], grid_dimensions: Sequence[float],
                          scene_centre_geodetic: NDArray) -> "GridSarScene":
        """
        Creates a GridSarScene aligned with local ENU (East, North, Up) axes. Length, breadth and depth will be aligned
        with East, North and Up respectively. Avoid using at geodetic coordinate singularities
        (i.e. along the line joining the centre of the Earth to the North Pole).
        Parameters
        ----------
        grid_shape : Sequence[int]
            Sequence of 3 integers indicating the shape of the grid (i.e. how many regularly-spaced planes of
            points are in it). Sequence should be in z, y, x order (depth, breadth, length), i.e. C-array ordering,
            for consistency.
        grid_dimensions : Sequence of 3 floats indicating the dimensions of the cuboid it lives in (i.e. the edge lengths).
            Points fil all the way up to the faces/edges/vertices. Ordering must be l_z, l_y, l_x,
            (depth, breadth, length) for consistency with image arrays that will eventually be produced using these classes.
        scene_centre_geodetic : NDArray
            Length 3 real-valued array specifying scene centre position in geodetic coordinates
            (e.g. [0.0, 0.0, 0.0], Null Island at sea level).
        Returns
        -------
        grid_sar_scene : A GridSarScene aligned with local ENU (East, North, Up) axes.
        Length, breadth and depth will be aligned with East, North and Up respectively.
        """
        centre_ecef, rotation_matrix = scene_params_enu(scene_centre_geodetic)
        return GridSarScene(grid_shape, grid_dimensions, centre_ecef, rotation_matrix)

    @staticmethod
    def from_aperture(aperture: BaseSimpleSarAperture, scene_centre_ecef: NDArray, grid_dimensions: Sequence[float],
                      grid_shape: Optional[Sequence[int]] = None, safety_factor: float = 1.2) -> "GridSarScene":
        """
        Creates a GridSarScene aligned to an existing aperture. Length, breadth and depth of scene cuboid should
        correspond to conventional SAR notions of range, cross range and out-of-plane respectively.
        Parameters
        ----------
        aperture : BaseSimpleSarAperture
            Aperture we're aligning to.
        scene_centre_ecef : NDArray
            Length 3 real-valued array indicating cartesian point we're forming our scene around.
        grid_dimensions : Sequence[float]
            Sequence of 3 floats indicating the dimensions of the cuboid it lives in (i.e. the edge lengths).
            Points fil all the way up to the faces/edges/vertices. Ordering must be l_z, l_y, l_x,
            (depth, breadth, length) for consistency with image arrays that will eventually be produced using these classes.
        grid_shape : Optional[Sequence[int]]
            If provided, sequence of 3 integers indicating the shape of the grid (i.e. how many regularly-spaced
            planes of points are in it). Sequence should be in z, y, x order (depth, breadth, length), i.e.
            C-array ordering, for consistency. If not provided (default, recommended), these numbers will be
            chosen for the user in accordance with k-space SAR sampling theory.
        safety_factor : float
            Safety factor which determines how much we will oversample in real space to guarantee we cover
            local k-space properly (losslessly). Default value of 1.2 is usually sensible, override at user's risk.
        Returns
        -------

        """
        rot_mat, sample_rates, _ = scene_params_classic_sar(scene_centre_ecef, aperture)
        if grid_shape is None:
            grid_shape = np.rint(np.clip(np.array(grid_dimensions) * sample_rates * safety_factor, a_min=1.0,
                                         a_max=None)).astype(int).tolist()
        return GridSarScene(grid_shape, grid_dimensions, scene_centre_ecef, rot_mat)
