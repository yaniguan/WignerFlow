"""All unit conversions live here.

Raw HDF5 files use OpenMM units: nm, ps, kJ/mol, amu.
The model and the analysis use Å and fs. Energies stay in kJ/mol.
"""

NM_TO_A = 10.0
PS_TO_FS = 1000.0

# nm/ps -> Å/fs: 10 Å / 1000 fs
NM_PER_PS_TO_A_PER_FS = NM_TO_A / PS_TO_FS

# Boltzmann constant in kJ/(mol K)
KB = 0.0083144626

# 1 amu * (Å/fs)^2 in kJ/mol.
# 1 amu = 1 g/mol = 1e-3 kg/mol; 1 Å/fs = 1e5 m/s -> 1e-3 * 1e10 J/mol = 1e7 J/mol = 1e4 kJ/mol
AMU_A2_PER_FS2_TO_KJ_PER_MOL = 1.0e4

# Wavenumber (cm^-1) of a frequency given in 1/fs: nu[1/s] / c[cm/s]
C_CM_PER_S = 2.99792458e10


def per_fs_to_wavenumber(f_per_fs):
    return f_per_fs * 1e15 / C_CM_PER_S
