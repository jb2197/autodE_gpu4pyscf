import numpy as np
import os
import pytest

from autode.atoms import Atom
from autode.wrappers.PySCF import PySCF
from autode.calculations import Calculation
from autode.species.molecule import Molecule
from autode.point_charges import PointCharge
from autode.exceptions import UnsupportedCalculationInput
from autode.utils import work_in_tmp_dir
from autode.wrappers.keywords import SinglePointKeywords

here = os.path.dirname(os.path.abspath(__file__))

method = PySCF()

pytestmark = pytest.mark.skipif(
    not method.is_available, reason="PySCF is not available"
)

@work_in_tmp_dir()
def test_pyscf_calculation():
    test_mol = Molecule(
        name="test_mol", smiles="C"
    )
    calc = Calculation(
        name="sp",
        molecule=test_mol,
        method=method,
        keywords=method.keywords.sp,
    )
    calc.run()

    assert test_mol.n_atoms == 5
    assert test_mol.energy is not None
    assert test_mol.energy < -39.0

@work_in_tmp_dir()
def test_pyscf_opt_calculation():
    test_mol = Molecule(
        name="test_mol", smiles="C"
    )
    calc = Calculation(
        name="opt",
        molecule=test_mol,
        method=method,
        keywords=method.keywords.opt,
    )
    calc.run()

    assert calc.optimiser.converged
    assert test_mol.energy is not None
    assert test_mol.energy < -39.0

@work_in_tmp_dir()
def test_point_charge():
    test_mol = Molecule(name="test_mol", smiles="C")

    # Methane with a point charge fairly far away - unsupported by PySCF wrapper
    calc = Calculation(
        name="sp_point_charge",
        molecule=test_mol,
        method=method,
        keywords=method.keywords.sp,
        point_charges=[PointCharge(charge=1.0, x=10, y=1, z=1)],
    )
    
    with pytest.raises(UnsupportedCalculationInput):
        calc.run()

@work_in_tmp_dir()
def test_gradients():
    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.0)])
    h2.single_point(method)

    delta_r = 1e-6
    h2_disp = Molecule(
        name="h2_disp", atoms=[Atom("H"), Atom("H", x=1.0 + delta_r)]
    )
    h2_disp.single_point(method)

    delta_energy = h2_disp.energy - h2.energy  # Ha
    grad = delta_energy / delta_r  # Ha A^-1

    calc = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=method.keywords.grad,
    )

    calc.run()

    diff = h2.gradient[1, 0] - grad  # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1e-2

@work_in_tmp_dir()
def test_hessian():
    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.0)])
    calc = Calculation(
        name="h2_hess",
        molecule=h2,
        method=method,
        keywords=method.keywords.hess,
    )
    calc.run()
    
    assert h2.hessian is not None
    assert h2.hessian.shape == (6, 6)

@work_in_tmp_dir()
def test_solvation():
    methane = Molecule(smiles="C", solvent_name="water")
    calc = Calculation(
        name="methane_solv",
        molecule=methane,
        method=method,
        keywords=method.keywords.sp
    )
    calc.run()
    
    assert methane.energy is not None
    # Solvation should work and give a slightly different energy
    # but the main thing is that it runs without error.

@work_in_tmp_dir()
def test_dispersion():
    methane = Molecule(smiles="C")
    # PBE with D3BJ dispersion
    calc = Calculation(
        name="methane_disp",
        molecule=methane,
        method=method,
        keywords=SinglePointKeywords(["PBE", "def2-SVP", "D3BJ"])
    )
    calc.run()
    
    assert methane.energy is not None
    # This just ensures it runs without error; the wrapper 
    # should extract 'D3BJ' from the keywords.
