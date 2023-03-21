import qutip as qt

ZERO_FIELD_SPLITTING = 2.87e9 # Hz
E_GYROMAGNETIC_RATIO = 28.031679357966418e9  # Hz/Tesla (2.8 MHz/Gauss) = u_B * g / h
NITROGEN_GYROMAGNETIC_RATIO = 3.076417303e6  # Hz/Tesla

#reference - Principes and Techniques of the Quantum Diamond Microscope
E_NITROGEN_HYPERFINE_AXIAL = {14:-2.14e6, 15:3.03e6}
E_NITROGEN_HYPERFINE_TRANSVERSE = {14:-2.70e6, 15:3.65e6}

NITROGEN14_AXIAL_QUADRUPOLE_MOMENT = -5.01e6

## ground state hamiltonian

class NVNegativeGroundState:

    def __init__(self, isotope: int = 14, temperature: float = 290):
        self.isotope = isotope
        self.temperature = temperature

        if isotope not in [14, 15]:
            raise ValueError(f'{isotope} must be in [14, 15]')

        self.nitrogen_spin = 1
        if isotope == 15:
            self.nitrogen_spin = 1/2

        self.nitrogen_spin_op_dims = 2*self.nitrogen_spin + 1


    def zero_field_splitting(self):
        """
        Returns the temperature dependent zero field splitting

        Right now, this is modeled at ~room temperature and likely
        fails outside of that.
        """
        return ZERO_FIELD_SPLITTING  - 74.2e3 * (self.temperature - 290)


    def zero_field_hamiltonian_ms0(self):
        """
        Returns the zero field Hamiltonian term with zero energy ground state

          D * S_z**2

        Using this will set the energy of |m_s = 0> to 0 and the +- state
        energies to D
        """
        h = qt.tensor(self.zero_field_splitting() * qt.spin_Jz(1)**2,
                         qt.qeye(self.nitrogen_spin_op_dims))
        return h


    def zero_field_hamiltonian(self):
        """
        Returns the zero field Hamiltonian term as shown in
        'Principles and Techniques of the Quantum Diamond Microscope', Levine
        et al 2019

          D * (S_z**2 - (S**2)/3)

        Using this will set the ground state energy to -2/3 D and +- states'
        energies 1/3 D
        """
        S_squared = qt.spin_Jx(1)**2 + qt.spin_Jy(1)**2 + qt.spin_Jz(1)**2
        h = self.zero_field_splitting() * (qt.spin_Jz(1)**2 - S_squared/3)
        h = qt.tensor(h, qt.qeye(self.nitrogen_spin_op_dims))
        return h


    def rotate_frame(self, frequency, axis = 'z'):
        """
        typically, frequency = -driving_frequency, where driving_frequency
        is usually set to an |0> to |+-1> resonance frequency
        """

        h_e = frequency * qt.jmat(1, axis)
        h_e = qt.tensor(h_e, qt.qeye(self.nitrogen_spin_op_dims))

        h_n = frequency * qt.jmat(self.nitrogen_spin, axis)
        h_n = qt.tensor(qt.qeye(3), h_n)

        return h_e + h_n


    def static_mag_field_hamiltonian(self, B: list[float]=[0,0,0],
                                            include_nucleus: bool = False):
        """
        Returns the Hamiltonian term for an NV center in a static B field.

          gamma_e B dot S

        where gamma_e is the electron gyromagnetic ratio, B is in Tesla,
        and S is the spin-1 spin vector (and the sign is + because e charge is -1)

        If include_nucleus is True, will also apply B field to
        nitrogen. However, the amplitude of this effect is 10,000x smaller
        compared to B field interaction with electron.

          -gamma_n B dot I

        where gamma_n is the nitrogen gryomagnetic ratio, B is in Tesla,
        and I is the spin vector operator for 14N (s = 1) or 15N (s = 1/2).

        """
        # gamma_e * B dot S
        h = B[0] * qt.spin_Jx(1)
        h += B[1] * qt.spin_Jy(1)
        h += B[2] * qt.spin_Jz(1)
        h *= E_GYROMAGNETIC_RATIO
        h = qt.tensor(h, qt.qeye(self.nitrogen_spin_op_dims))

        hnit = 0
        if include_nucleus:
            hnit = B[0] * qt.spin_Jx(self.nitrogen_spin)
            hnit += B[1] * qt.spin_Jy(self.nitrogen_spin)
            hnit += B[2] * qt.spin_Jz(self.nitrogen_spin)
            hnit *= -NITROGEN_GYROMAGNETIC_RATIO
            hnit = qt.tensor(qt.qeye(3), hnit)

        return h + hnit

    def nitrogen_hyperfine_hamiltonian(self):
        """
        Returns the Hamiltonian interaction term for Nitrogen-electron
        spin-spin interaction.

          A_axial * S_z I_z  +  A_transverse * [ S_x I_x + S_y I_y ]

        """
        #A_axial * S_z (x) I_z
        h_axial = qt.tensor(qt.spin_Jz(1), qt.spin_Jz(self.nitrogen_spin))
        h_axial *= E_NITROGEN_HYPERFINE_AXIAL[self.isotope]

        #A_transverse * [ S_x (x) I_x + S_y (x) I_y ]
        h_transverse = qt.tensor(qt.spin_Jx(1), qt.spin_Jx(self.nitrogen_spin))
        h_transverse += qt.tensor(qt.spin_Jy(1), qt.spin_Jy(self.nitrogen_spin))
        h_transverse *= E_NITROGEN_HYPERFINE_TRANSVERSE[self.isotope]

        return h_axial + h_transverse

    def nitrogen_electric_quadrupole_hamiltonian(self):
        """
        Returns the Hamiltonian interaction term for Nitrogen quadrupole moment.

        returns P * [I_z**2 - (I**2)/3]

        """
        if self.isotope == 15:
            h = 0 # is this right?
        else:
            #there must be a function in qutip,numpy or scipy for this?
            I_squared = qt.spin_Jx(self.nitrogen_spin)**2
            I_squared += qt.spin_Jy(self.nitrogen_spin)**2
            I_squared += qt.spin_Jz(self.nitrogen_spin)**2
            # yes, this is the same as s(s+1)*identity(2*s + 1) = 2 Identity_3
            # self.nitrogen_spin(self.nitrogen_spin + 1)*qt.qeye(self.nitrogen_spin_op_dims)

            h = qt.spin_Jz(self.nitrogen_spin)**2 - I_squared/3
            h *= NITROGEN14_AXIAL_QUADRUPOLE_MOMENT
            h = qt.tensor(qt.qeye(3), h)

        return h
