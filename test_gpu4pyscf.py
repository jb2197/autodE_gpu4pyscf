'''
Test script comparing PySCF (CPU) and GPU4PySCF (GPU) performance.

- Fetches a 3D molecular geometry from PubChem (SMILES "COC").
- Runs RHF SCF and CCSD (incore) on both CPU (PySCF) and GPU (gpu4pyscf) backends.
- Prints energies, timings, and speedups for SCF and CCSD.

Requirements:
- pyscf
- gpu4pyscf
- pubchempy
- CUDA-capable GPU with appropriate drivers (for gpu4pyscf)

Usage:
python /home/aaldo/egsmole/test/test_gpu4pyscf.py

TODO: make it a unit test
'''

import time
import pyscf
from pyscf import gto, scf
import gpu4pyscf
from pyscf import cc as cc_cpu
from gpu4pyscf.cc import ccsd_incore as cc_gpu
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# get molecule from PubChem
rdkit_mol = Chem.MolFromSmiles("COC")
rdkit_mol = Chem.AddHs(rdkit_mol)
AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG())
AllChem.UFFOptimizeMolecule(rdkit_mol)
conf = rdkit_mol.GetConformer()
xyz = ""
for atom in rdkit_mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    xyz += f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"

# Create a simple molecule (water)
mol = gto.M()
mol.atom = xyz
mol.basis = 'def2-svp'
mol.build()

# Run SCF calculation with PySCF (CPU)
print("Running PySCF (CPU) calculation...")
start_time = time.time()
mf_cpu = scf.RHF(mol)
e_cpu = mf_cpu.kernel()
cpu_scf_time = time.time() - start_time
# print(f"PySCF CPU energy: {e_cpu}")
print(f"PySCF CPU SCF time: {cpu_scf_time:.4f} seconds")

# Run SCF calculation with GPU4PySCF (GPU)
print("\nRunning GPU4PySCF (GPU) calculation...")
start_time = time.time()
mf_gpu = scf.RHF(mol).to_gpu()
e_gpu = mf_gpu.kernel()
gpu_scf_time = time.time() - start_time
# print(f"GPU4PySCF GPU energy: {e_gpu}")
print(f"GPU4PySCF GPU SCF time: {gpu_scf_time:.4f} seconds")

print(f"\nSCF Energy difference: {abs(e_cpu - e_gpu)}")
print(f"SCF Speedup (CPU/GPU): {cpu_scf_time/gpu_scf_time:.2f}x")

print("\n" + "="*50)
print("CCSD Calculations")
print("="*50)

# Run CCSD calculation with PySCF (CPU)
print("Running PySCF (CPU) CCSD calculation...")
start_time = time.time()
cc_obj_cpu = cc_cpu.CCSD(mf_cpu)
cc_obj_cpu.incore_complete = True
cc_obj_cpu.kernel()
cpu_ccsd_time = time.time() - start_time
e_cpu = cc_obj_cpu.e_tot
# print(f"PySCF CPU CCSD energy: {e_cpu}")
print(f"PySCF CPU CCSD time: {cpu_ccsd_time:.4f} seconds")

# Run CCSD calculation with GPU4PySCF (GPU)
print("\nRunning GPU4PySCF (GPU) CCSD calculation...")
start_time = time.time()
cc_obj_gpu = cc_gpu.CCSD(mf_gpu)
cc_obj_gpu.incore_complete = True
cc_obj_gpu.kernel()
gpu_ccsd_time = time.time() - start_time
e_gpu = cc_obj_gpu.e_tot
# print(f"GPU4PySCF GPU CCSD energy: {e_gpu}")
print(f"GPU4PySCF GPU CCSD time: {gpu_ccsd_time:.4f} seconds")

print(f"\nCCSD Energy difference: {abs(e_cpu - e_gpu)}")
print(f"CCSD Speedup (CPU/GPU): {cpu_ccsd_time/gpu_ccsd_time:.2f}x")

print("\n" + "="*50)
print("Summary")
print("="*50)
print(f"Total CPU time: {cpu_scf_time + cpu_ccsd_time:.4f} seconds")
print(f"Total GPU time: {gpu_scf_time + gpu_ccsd_time:.4f} seconds")
print(f"Overall Speedup: {(cpu_scf_time + cpu_ccsd_time)/(gpu_scf_time + gpu_ccsd_time):.2f}x")