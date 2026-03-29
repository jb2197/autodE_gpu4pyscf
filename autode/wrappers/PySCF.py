from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, List, Optional, Dict, Any

import numpy as np
import autode.wrappers.keywords as kws
import autode.wrappers.methods

from autode.config import Config
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.hessians import Hessian
from autode.log import logger
from autode.exceptions import (
    CouldNotGetProperty,
    NotImplementedInMethod,
    UnsupportedCalculationInput,
)

# pyscf imports
from pyscf import gto, scf, dft

has_gpu4pyscf = False
try:
    import gpu4pyscf
    import cupy as cp

    has_gpu4pyscf = True
except ImportError:
    cp = None

if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.calculations.types import CalculationType



class PySCF(autode.wrappers.methods.Method):
    """
    PySCF wrapper that runs calculations in-memory (no external IO).
    Implements energy, gradient, and Hessian.
    """

    def __init__(self):
        pyscf_conf = getattr(Config, "PySCF", None)
        keywords = pyscf_conf.keywords if pyscf_conf else None
        doi_list = ["10.1039/C6CP41100A"]
        implicit = pyscf_conf.implicit_solvation_type if pyscf_conf else None
        super().__init__(
            name="pyscf",
            keywords_set=keywords,
            doi_list=doi_list,
        )
        self.implicit_solvation_type = implicit or kws.cpcm

    def __repr__(self):
        return f"PySCF(available = {self.is_available})"

    @property
    def uses_external_io(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        spec = importlib.util.find_spec("pyscf")
        available = spec is not None
        logger.info(f"Setting the availability of pyscf -> {available}")
        return available

    def implements(self, calculation_type: "CalculationType") -> bool:
        from autode.calculations.types import CalculationType as ct
        return calculation_type in (ct.energy, ct.gradient, ct.hessian)

    def version_in(self, calc: "CalculationExecutor") -> str:
        try:
            from pyscf import __version__ as pyscf_version
            return str(pyscf_version)
        except Exception:
            return "???"

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        """Provide energy for parent ExternalMethod API if used."""
        cache = getattr(calc, "_pyscf_cache", None)
        if not cache or "energy" not in cache:
            raise CouldNotGetProperty("energy")
        return PotentialEnergy(float(cache["energy"]), units="Ha")

    def execute(self, calc: "CalculationExecutor") -> None:
        assert calc.input.keywords is not None, "Must have input keywords"
        molecule = calc.molecule

        if not self.is_available:
            raise RuntimeError("PySCF is not available in this Python environment")

        if calc.input.point_charges is not None:
            raise UnsupportedCalculationInput("Point charges not supported in this PySCF wrapper")

        jobtype = self._jobtype_for(calc.input.keywords)
        basis = self._basis_for(calc.input.keywords)
        xc = self._functional_for(calc.input.keywords)
        disp = self._dispersion_for(calc.input.keywords)

        logger.info(f"PySCF run: jobtype={jobtype}, basis={basis or 'sto-3g'}, xc={xc or 'HF'}")

        atom_lines = ""
        for atom in molecule.atoms:
            x, y, z = atom.coord
            atom_lines += f"{atom.label} {x:.15f} {y:.15f} {z:.15f}\n"
        mol = gto.Mole()
        mol.atom = atom_lines
        mol.unit = "Angstrom"
        mol.charge = molecule.charge
        mol.spin = molecule.mult - 1
        mol.basis = basis or "def2-svp"
        mol.build()

        if mol.spin == 0:
            if xc:
                mf = dft.RKS(mol)
                mf.xc = xc
            else:
                mf = scf.RHF(mol)
        else:
            if xc:
                mf = dft.UKS(mol)
                mf.xc = xc
            else:
                mf = scf.UHF(mol)
        mf.verbose = 0

        # Apply empirical dispersion correction if requested and supported
        if disp is not None:
            try:
                disp = disp.lower() if isinstance(disp, str) else disp
                mf.disp = disp
                logger.info(f"Applying empirical dispersion correction: {disp}")
            except Exception as e:
                logger.warning(
                    f"Failed to apply dispersion '{disp}': {e}. "
                    f"Proceeding without dispersion"
                )

        if has_gpu4pyscf:
            logger.info("moving to gpu")
            mf = mf.to_gpu()
            logger.info("applying density fitting for GPU job")
            mf = mf.density_fit()

        if molecule.is_implicitly_solvated:
            mf = mf.PCM()
            mf.with_solvent.eps = molecule.solvent.dielectric

        E = mf.kernel()

        cache: Dict[str, Any] = {"energy": float(E)}
        # Set energy on the molecule for internal execution
        calc.molecule.energy = PotentialEnergy(cache["energy"], units="Ha")

        if jobtype == "force":
            g_obj = mf.Gradients()
            g = g_obj.kernel()

            if cp is not None and isinstance(g, cp.ndarray):
                g = cp.asnumpy(g)

            cache["gradient"] = np.asarray(g, dtype=np.float64)
            calc.molecule.gradient = Gradient(
                cache["gradient"], units="Ha a0^-1"
            ).to("Ha Å^-1")

        elif jobtype == "freq":
            try:
                H_obj = mf.Hessian()
                H_arr = H_obj.kernel()
            except Exception as e:
                print(f"Error computing Hessian with PySCF: {e}")
                raise CouldNotGetProperty("Hessian") from e

            if cp is not None and isinstance(H_arr, cp.ndarray):
                H_arr = cp.asnumpy(H_arr)

            n_atoms = calc.molecule.n_atoms
            H_arr = np.asarray(H_arr, dtype=np.float64)
            # Rearrange from (n_atoms, n_atoms, 3, 3) to (3*n_atoms, 3*n_atoms)
            H_arr = H_arr.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)

            cache["hessian"] = H_arr
            logger.info(f"Converted Hessian shape={cache['hessian'].shape}")
            calc.molecule.hessian = Hessian(
                cache["hessian"],
                atoms=calc.molecule.atoms,
                functional=getattr(calc.input.keywords, "functional", None),
                units="Ha a0^-2",
            ).to("Ha Å^-2")

        setattr(calc, "_pyscf_cache", cache)
        return None

    def energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        cache = getattr(calc, "_pyscf_cache", None)
        if not cache or "energy" not in cache:
            raise CouldNotGetProperty("energy")
        energy = PotentialEnergy(cache["energy"], units="Ha")
        energy.set_method_str(method=self, keywords=calc.input.keywords)
        return energy

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        cache = getattr(calc, "_pyscf_cache", None)
        if not cache or "gradient" not in cache:
            raise CouldNotGetProperty("gradient")
        return Gradient(cache["gradient"], units="Ha a0^-1").to("Ha Å^-1")

    def hessian_from(self, calc: "CalculationExecutor") -> Hessian:
        cache = getattr(calc, "_pyscf_cache", None)
        if not cache or "hessian" not in cache:
            raise CouldNotGetProperty("Hessian")
        return Hessian(
            cache["hessian"],
            atoms=calc.molecule.atoms,
            functional=getattr(calc.input.keywords, "functional", None),
            units="Ha a0^-2",
        ).to("Ha Å^-2")

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        return calc.molecule.coordinates

    def atoms_from(self, calc: "CalculationExecutor") -> "autode.atoms.Atoms":
        atoms = calc.molecule.atoms.copy()
        atoms.coordinates = self.coordinates_from(calc)
        return atoms

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        raise NotImplementedInMethod

    @staticmethod
    def _jobtype_for(keywords) -> str:
        for word in keywords:
            if isinstance(word, str) and "jobtype" in word.lower():
                lw = word.lower()
                if "force" in lw:
                    return "force"
                if "freq" in lw:
                    return "freq"
                if "opt" in lw:
                    logger.warning("PySCF wrapper does not implement optimisation; using energy")
                    return "energy"
        if isinstance(keywords, kws.HessianKeywords):
            return "freq"
        if isinstance(keywords, kws.GradientKeywords):
            return "force"
        return "energy"

    @staticmethod
    def _basis_for(keywords) -> Optional[str]:
        for word in keywords:
            if isinstance(word, kws.BasisSet):
                return getattr(word, "pyscf", None) or getattr(word, "name", None) or str(word)
        return None

    @staticmethod
    def _functional_for(keywords) -> Optional[str]:
        for word in keywords:
            if isinstance(word, kws.Functional):
                return getattr(word, "pyscf", None) or getattr(word, "name", None)
        for word in keywords:
            if isinstance(word, str):
                lw = word.lower()
                for xc in ("b3lyp", "pbe", "blyp", "bp86", "wb97x", "m06", "pbesol"):
                    if xc in lw:
                        return xc
        return None

    @staticmethod
    def _dispersion_for(keywords) -> Optional[str]:
        """
        Determine requested empirical dispersion correction from keywords.
        Returns a string identifier (e.g., 'D3BJ') or None if not set.
        """
        for word in keywords:
            if isinstance(word, kws.DispersionCorrection):
                # Prefer method-specific mapping
                return getattr(word, "pyscf", None) or getattr(word, "name", None)
        for word in keywords:
            if isinstance(word, str):
                lw = word.lower()
                if "d3bj" in lw:
                    return "D3BJ"
                if "d3" in lw:
                    return "D3"
        return None

pyscf_instance = PySCF()