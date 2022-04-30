import lammps_api as lmp

lmp_dump = lmp.lammps_dump()
lmp_dump.set_dump_file("data/1.dump", 135, 4000, 5)
lmp_dump.set_born_charge()
lmp_dump.set_eigen_vector()
lmp_dump.read_dump_file()
lmp_dump.build_map2supercell_cubic(lat_cons_uc=4.03, nx=3, ny=3, nz=3)
lmp_dump.project2phonon()
# lmp_dump.print_magnitude_phonon()
lmp_dump.print_total_dipole()
