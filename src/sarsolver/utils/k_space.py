from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..base import BaseSarScene, BaseSimpleSarAperture


def get_obb_basis(xs: NDArray) -> (NDArray, NDArray):
    """
    Function to compute the eigendecomposition needed for easy oriented bounding box
    construction.
    Parameters
    ----------
    xs : NDArray
        The positions of the points, Cartesian.

    Returns
    -------
    evals: NDArray, evecs: NDArray
        The eigenvalues and eigenvectors of the PCA of the xs.
    """
    zs = xs - np.mean(xs, axis=0, keepdims=True)
    evals, evecs = np.linalg.eigh(np.dot(zs.T, zs))
    return evals, evecs


class OrientedBoundingBox:
    def __init__(self, xs: NDArray):
        """
        A class to represent an oriented bounding box. Useful for k-space analysis.
        The class has a bunch of spans, an overall displacement for the centre of
        the box, and a basis for the box contained within a matrix.
        Parameters
        ----------
        xs : NDArray
            The points we're making a bounding box around.
        """
        xs = xs[np.isfinite(xs).all(axis=1)]
        obb_evals, obb_basis = get_obb_basis(xs)

        if np.linalg.det(obb_basis) < 0.0:
            obb_basis = -obb_basis

        # Want to work after rotation into the xs' PCA eigenbasis
        transformed_xs = np.dot(obb_basis.T, xs.T).T
        mins = np.min(transformed_xs, axis=0)
        maxs = np.max(transformed_xs, axis=0)
        transformed_centre = 0.5 * (mins + maxs)
        displacement = np.dot(obb_basis, transformed_centre.T).T
        # We generally end up defining the "range" span direction to be the one most closely aligned
        # with the overall displacement. Consistent with conventional SAR!
        range_index = np.argmax(np.abs(transformed_centre))
        spans = np.abs(maxs - mins)
        range_span = spans[range_index]
        # "Cross range" is the biggest span range which isn't also range
        if range_span < np.max(spans):
            cross_range_index = np.argmax(spans)
        else:
            sorted_indices = np.argsort(spans)
            cross_range_index = sorted_indices[1]
        indices = np.arange(3)
        # There's a leftover direction which we can call "out of plane"
        remaining_index = np.delete(indices, [range_index, cross_range_index])[0]
        self.range_index = range_index
        self.cross_range_index = cross_range_index
        self.out_of_plane_index = remaining_index

        self.range_span = spans[self.range_index]
        self.cross_range_span = spans[self.cross_range_index]
        self.out_of_plane_span = spans[self.out_of_plane_index]
        self.obb_basis = obb_basis
        self.displacement = displacement
        self.spans = spans

    def to_obb_coords(self, points: NDArray) -> NDArray:
        """
        Transforms from default coordinates into box-oriented coordinates.
        Parameters
        ----------
        points : NDArray
            Points to be transformed.

        Returns
        -------
        xs : NDArray
            Transformed points, in box coordinates.
        """
        xs = points - np.expand_dims(self.displacement, axis=0)
        return np.dot(self.obb_basis.T, xs.T).T

    def from_obb_coords(self, points: NDArray) -> NDArray:
        """
        Transforms back from box coordinates into default ones.
        Parameters
        ----------
        points : NDArray
            Points to be transformed.

        Returns
        -------
        xs : NDArray
            Transformed points, in default coordinates.
        """
        xs = np.dot(self.obb_basis, points.T).T
        return xs + self.displacement


def generate_simple_basebanding_values(scene: BaseSarScene, aperture: BaseSimpleSarAperture,
                                       basebanding_spot: Optional[NDArray] = None) -> NDArray:
    """
    Makes an array of phases in order to baseband an image. This is simple basebanding about a point, not adaptive
    basebanding.
    Parameters
    ----------
    scene : BaseSarScene
        Scene containing the points we wish to baseband for.
    aperture : BaseSimpleSarAperture
        Aperture we're basebanding with respect to, in terms of k-space displacement.
    basebanding_spot : NDArray
        Real-valued [3]-shape array designating the point we're basebanding with respect to. Defaults to mean of
        the positions in the scene.

    Returns
    -------
    phases : NDArray
        Phases to be applied to an image vector corresponding to the scene in order to baseband it.
    """
    if basebanding_spot is None:
        basebanding_spot = np.mean(scene.positions, axis=0)
    k_space_bb = OrientedBoundingBox(aperture.edge_k_vecs(basebanding_spot))
    relative_positions = scene.positions - np.expand_dims(basebanding_spot, axis=0)
    projected_phases = relative_positions @ k_space_bb.displacement
    return np.exp(-1.j * aperture.sign_multiplier * projected_phases)
