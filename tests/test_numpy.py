from os.path import join
from pathlib import Path

import numpy as np
from pytest import approx


def test_projections(vanuatu_dataset, vanuatu_operator):
    y = vanuatu_dataset.phase_history().reshape([-1])
    x = vanuatu_operator.rmatvec(y)
    z = vanuatu_operator.matvec(x)
    this_file = Path(__file__)
    canned_x = np.load(join(this_file.parent, "artifacts", "vanuatu_x.npy"))
    canned_z = np.load(join(this_file.parent, "artifacts", "vanuatu_z.npy"))
    assert x == approx(canned_x, rel=1.0E-3)
    assert z == approx(canned_z, rel=1.0E-3)
