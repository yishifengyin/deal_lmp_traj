import numpy as np


class lammps_dump:
    def __init__(self) -> None:
        self.nstep = 0
        self.natom = 0
        self.iso_flag = True

    def allocate(self):
        assert (self.nstep > 0) & (self.natom > 0)
        self.x = np.zeros((self.nstep, self.natom, 3))
        self.scale = np.zeros((self.nstep))
        self.timestep = np.zeros((self.nstep))
        self.lohi = np.zeros((self.nstep, 3, 2))
        self.lattice_matrix = np.zeros((self.nstep, 6))
        self.type = np.zeros((self.natom)).astype(np.int)

    def set_dump_file(self, dump_file_path, natom, nstep, ntype):
        self.dump_file = dump_file_path
        self.natom = natom
        self.nstep = nstep
        self.ntype = ntype
        self.allocate()

    def read_dump_file(self):
        ncount_step = 0
        with open(self.dump_file, "r") as infile:
            for istep in range(self.nstep):
                for iline in range(9 + self.natom):
                    words = infile.readline().split()
                    if iline == 1:
                        self.timestep[istep] = int(words[0])
                        ncount_step += 1
                        if ncount_step == self.nstep:
                            break
                    if iline == 3:
                        assert int(words[0]) == self.natom
                    if (iline >= 5) & (iline < 8):
                        self.lohi[istep, iline - 5, :] = np.array(words[0:2]).astype(np.float)
                        self.lattice_matrix[istep, iline - 5] = (
                            self.lohi[istep, iline - 5, 1] - self.lohi[istep, iline - 5, 0]
                        )
                        self.lattice_matrix[istep, iline - 2] = float(words[2])
                    if (iline >= 9) & (iline < (9 + self.natom)):
                        self.x[istep, iline - 9, :] = np.array(words[2:5]).astype(np.float)
                    if self.iso_flag:
                        self.scale[istep] = self.lohi[istep, 0, 1] - self.lohi[istep, 0, 0]
            infile.close()
        if self.iso_flag:
            for istep in range(1, self.nstep):
                self.scale[istep] /= self.scale[0]
            self.scale[0] = 1.0

    def build_map2supercell_cubic(self, lat_cons_uc, nx, ny, nz):
        self.nsc = nx * ny * nz
        for i in range(5):
            self.type[i * self.nsc : (i + 1) * self.nsc] = i
        xf_0 = np.array([[0.5, 0.5, 0.5], [1, 1, 1], [0.5, 1, 1], [1, 0.5, 1], [1, 1, 0.5]])
        pos_tag = ["corner", "center", "facex", "facey", "facez"]
        x_0 = self.x[0, :, :] - self.lohi[0, :, 0]
        self.weightpertype = [0.125, 1.0, 0.5, 0.5, 0.5]
        self.mapatom2xyz = np.zeros((self.natom, 3)).astype(np.int)
        self.mapatom2supercell = []
        self.mapsupercell2atom = [[]] * self.nsc
        for iatom in range(self.natom):
            itype = self.type[iatom]
            ix = x_0[iatom, 0] / lat_cons_uc - xf_0[itype, 0]
            iy = x_0[iatom, 1] / lat_cons_uc - xf_0[itype, 1]
            iz = x_0[iatom, 2] / lat_cons_uc - xf_0[itype, 2]
            assert check_round(ix) & check_round(iy) & check_round(iz)
            ix = period_pos_int(round(ix), nx)
            iy = period_pos_int(round(iy), ny)
            iz = period_pos_int(round(iz), nz)
            self.mapatom2xyz[iatom, :] = np.array([ix, iy, iz])
            sc_list = self.supercell_shared(pos_tag[itype], ix, iy, iz, nx, ny, nz)
            self.mapatom2supercell.append(sc_list)
            for isc in sc_list:
                self.mapsupercell2atom[isc].append(iatom)
        self.x_0 = x_0

    def supercell_shared(self, pos, ix, iy, iz, nx, ny, nz):
        sc_list = []
        if pos == "center":
            sc_list.append(ix * ny * nz + iy * nz + iz)
        elif pos == "corner":
            sc_list.append(ix * ny * nz + iy * nz + iz)
            sc_list.append(period_pos_int(ix - 1, nx) * ny * nz + iy * nz + iz)
            sc_list.append(ix * ny * nz + period_pos_int(iy - 1, ny) * nz + iz)
            sc_list.append(ix * ny * nz + iy * nz + period_pos_int(iz - 1, nz))
            sc_list.append(
                period_pos_int(ix - 1, nx) * ny * nz + period_pos_int(iy - 1, ny) * nz + iz
            )
            sc_list.append(
                ix * ny * nz + period_pos_int(iy - 1, ny) * nz + period_pos_int(iz - 1, nz)
            )
            sc_list.append(
                period_pos_int(ix - 1, nx) * ny * nz + iy * nz + period_pos_int(iz - 1, nz)
            )
            sc_list.append(
                period_pos_int(ix - 1, nx) * ny * nz
                + period_pos_int(iy - 1, ny) * nz
                + period_pos_int(iz - 1, nz)
            )
        elif pos == "facex":
            sc_list.append(ix * ny * nz + iy * nz + iz)
            sc_list.append(period_pos_int(ix - 1, nx) * ny * nz + iy * nz + iz)
        elif pos == "facey":
            sc_list.append(ix * ny * nz + iy * nz + iz)
            sc_list.append(ix * ny * nz + period_pos_int(iy - 1, ny) * nz + iz)
        elif pos == "facez":
            sc_list.append(ix * ny * nz + iy * nz + iz)
            sc_list.append(ix * ny * nz + iy * nz + period_pos_int(iz - 1, nz))
        else:
            print("Wrong position argument")
        return sc_list

    def project2phonon(self):
        mag_phonon = np.zeros((self.nstep, self.nsc))
        for istep in range(self.nstep):
            dis = self.x[istep, :, :] - self.lohi[istep, :, 0] - self.x_0[:, :] * self.scale[istep]
            for iatom in range(self.natom):
                itype = self.type[iatom]
                for isc in self.mapatom2supercell[iatom]:
                    mag_phonon[istep, isc] += (
                        self.weightpertype[itype]
                        * project2vec(dis[iatom, :], self.eigen_vec[itype, :])
                        * self.born_charge[itype]
                    )
        self.mag_phonon = mag_phonon

    def set_eigen_vector(self):
        # self.eigen_vec = np.zeros((self.ntype, 3))
        self.eigen_vec = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

    def set_born_charge(self):
        # self.born_charge = np.zeros((self.ntype))
        self.born_charge = [1.0, 1.0, 1.0, 1.0, 1.0]

    def print_magnitude_phonon(self):
        np.savetxt("magnitude_phonon.dat", self.mag_phonon)

    def print_total_dipole(self):
        tot_mag = np.zeros((self.nstep))
        for istep in range(self.nstep):
            tot_mag[istep] = np.sum(self.mag_phonon[istep, :])
        np.savetxt("total_dipole.dat", tot_mag)


def period_pos_int(x, p) -> int:
    if x < 0:
        return period_pos_int(x + p, p)
    elif x >= p:
        return period_pos_int(x - p, p)
    else:
        return int(x)


def check_round(x, tolerence=0.01) -> bool:
    rx = round(x)
    if abs(x - rx) < tolerence:
        return True
    else:
        return False


def project2vec(v1, vp) -> float:
    mag_vp = np.linalg.norm(vp)
    if mag_vp == 0.0:
        return 0.0
    else:
        return np.dot(v1, vp) / np.linalg.norm(vp)
