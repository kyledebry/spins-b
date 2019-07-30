Examples
========

This folder contains various examples of SPINS-B and its modifications. Start with the ``wdm2`` example, as it is the same as typical SPINS-B.

Then, look at the ``phase`` and ``photonic_crystal`` example folders.

Breakdown of ``phc.py``
-----------------------
``phc.py`` is found in the ``photonic_crystal`` folder. It is the most successful example of using SPINS-B to do some optimization of the GVD of a waveguide.

.. code:: python

    # Yee cell grid spacing in nanometers.
    GRID_SPACING = 40
    # If `True`, perform the simulation in 2D. Else in 3D.
    SIM_2D = True
    # Silicon refractive index to use for 2D simulations. This should be the
    # effective index value.
    SI_2D_INDEX = 2.20
    # Silicon refractive index to use for 3D simulations.
    SI_3D_INDEX = 3.45

    WIDTH = 2500  # Width from source to output
    NM_TO_M = 1e-9  # Conversion from nm to m
    THZ_TO_HZ = 1e12  # Conversion from THz to Hz
    S2M_TO_FS2MM = 1e27  # Conversion from s^2/m to fs^2/mm for GVD
    C = 299792458  # Speed of light (m/s)

This section just sets up constants and conversions.

.. code:: python

    def create_sim_space(gds_fg: str, gds_bg: str) -> optplan.SimulationSpace:
        """Creates the simulation space.

        The simulation space contains information about the boundary conditions,
        gridding, and design region of the simulation. The material stack is
        220 nm of Ta2O5 surrounded by oxide. The refractive indices used are NOT
        effective 2D indices, so the simulations will likely not translate to
        real 3D devices until that is fixed.

        Args:
            gds_fg: Location of the foreground GDS file.
            gds_bg: Location of the background GDS file.

        Returns:
            A `SimulationSpace` description.
        """

        mat_oxide = optplan.Material(mat_name="SiO2")

        mat_stack = optplan.GdsMaterialStack(
            background=mat_oxide,
            stack=[
                optplan.GdsMaterialStackLayer(
                    foreground=mat_oxide,
                    background=mat_oxide,
                    gds_layer=[100, 0],
                    extents=[-10000, -110],
                ),
                optplan.GdsMaterialStackLayer(
                    foreground=optplan.Material(mat_name="Ta2O5"),
                    background=mat_oxide,
                    gds_layer=[100, 0],
                    extents=[-110, 110],
                ),
            ],
        )

Here, we're setting up the material of the foreground and background. The oxide material `mat_oxide` is being defined as SiO2. The definition of this material is found in `spins>material>material.py`, and it uses the Sellmier coefficients for SiO2 to calculate `n` and `k` at different wavelengths. The foreground material is the material that is being shaped by SPINS, and is being set as Ta2O5, also defined in `material.py`.

.. code:: python

        if SIM_2D:
            # If the simulation is 2D, then we just take a slice through the
            # device layer at z = 0. We apply periodic boundary conditions along
            # the z-axis by setting PML thickness to zero.
            sim_region = optplan.Box3d(
                center=[0, 0, 0], extents=[7500, 4000, GRID_SPACING])
            pml_thickness = [10, 10, 10, 10, 0, 0]
        else:
            sim_region = optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 2000])
            pml_thickness = [10, 10, 10, 10, 10, 10]

This is setting up the boundaries of the simulation. Here is where the simulation volume can be altered, and the '2D' simulation is enforced bychanging the PML (perfectly matched layer) along the z axis to 0.

.. code::python

        return optplan.SimulationSpace(
            name="simspace_cont",
            mesh=optplan.UniformMesh(dx=GRID_SPACING),
            eps_fg=optplan.GdsEps(gds=gds_fg, mat_stack=mat_stack),
            eps_bg=optplan.GdsEps(gds=gds_bg, mat_stack=mat_stack),
            sim_region=sim_region,
            selection_matrix_type="direct_lattice",
            boundary_conditions=[optplan.BlochBoundary()] * 6,
            pml_thickness=pml_thickness,
        )

This is creating the ``SimulationSpace`` that will be used to generate all the instances of simulations that will be run. The GDS files ``eps_fg`` and ``eps_bg`` describe where the epsilon is forced to be the foreground material (``eps_fg``) and where it is allowed to be optimized (the area that is in ``eps_bg`` but not ``eps_fg``).

.. code:: python

    def finite_difference_second_derivative(f: List[optplan.Function], delta: float, i: int) -> optplan.Function:
        """Approximates the second derivative of f at the i-th point using finite difference coefficients.

        See https://en.wikipedia.org/wiki/Finite_difference_coefficient
        Code generated with http://web.media.mit.edu/~crtaylor/calculator.html"""
        assert len(f) >= 5
        values_after = len(f) - 1 - i

        if i == 0:
            difference = (35 * f[i + 0] - 104 * f[i + 1] + 114 * f[i + 2] - 56 * f[i + 3] + 11 * f[i + 4]) / 12
        elif i == 1:
            difference = (11 * f[i - 1] - 20 * f[i + 0] + 6 * f[i + 1] + 4 * f[i + 2] - 1 * f[i + 3]) / 12
        elif values_after == 0:
            difference = (11 * f[i - 4] - 56 * f[i - 3] + 114 * f[i - 2] - 104 * f[i - 1] + 35 * f[i + 0]) / 12
        elif values_after == 1:
            difference = (-1 * f[i - 3] + 4 * f[i - 2] + 6 * f[i - 1] - 20 * f[i + 0] + 11 * f[i + 1]) / 12
        else:
            difference = (-1 * f[i - 2] + 16 * f[i - 1] - 30 * f[i + 0] + 16 * f[i + 1] - 1 * f[i + 2]) / 12

        derivative = difference / (delta ** 2)

        return derivative

This function uses the finite difference approximation method (see linked Wikipedia page in comment) to more effectively approximate the second derivative of a set of points. I have here only defined it for 5 or more points, so there must be at least 5 ``k`` monitors to use this to calculate GVD.

.. code:: python

    def create_gvd_multiple_difference(k: List[optplan.Function], frequency_step: float = None,
                                       ignore_endpoints: bool = False) -> List[optplan.Function]:
        gvd = []
        omega_step = 2 * np.pi * frequency_step

        for i in range(len(k)):
            derivative = finite_difference_second_derivative(k, omega_step, i) * S2M_TO_FS2MM
            gvd.append(derivative)

        if ignore_endpoints:
            gvd = gvd[1:-1]

        return gvd

This function uses the previous function to calculate the GVD. It assumes that all inputs are in SI units, and the returned valules of GVD are in the traditional fs^2/mm units (**NOT** SI units).

.. code:: python

    def create_objective(sim_space: optplan.SimulationSpace
                         ) -> Tuple[optplan.Function, List[optplan.Monitor]]:
        """Creates the objective function to be minimized.

        The objective is `(1 - I_Kerr)^2 + (1 - out)^2` where `I_Kerr` is the
         intensity in the objective region that overlaps with the foreground
         layer (places where there is waveguide material, e.g. Si), and `out`
         is the power at the output port. Note that in an actual device, one
         should also add terms corresponding to the rejection modes as well.

        Args:
            sim_space: Simulation space to use.

        Returns:
            A tuple `(obj, monitors)` where `obj` is a description of objective
            function and `monitors` is a list of values to monitor (save) during
            the optimization process.
        """

        path_length = 6250

        # Create the waveguide source at the input.
        wg_source = optplan.WaveguideModeSource(
            center=[-path_length // 2, -250, 0],
            extents=[GRID_SPACING, 2500, 600],
            normal=[1, 0, 0],
            mode_num=0,
            power=1.0,
        )

        # Create a path from the source to the output to track the phase over.
        phase_path = optplan.Region(
            center=[0, -250, 0],
            extents=[max(path_length, GRID_SPACING), GRID_SPACING, GRID_SPACING],
            power=1
        )

        # Create the modal overlap at the input waveguide
        overlap_in = optplan.WaveguideModeOverlap(
            center=[-path_length // 2, -250, 0],
            extents=[GRID_SPACING, 2500, 600],
            mode_num=0,
            normal=[1, 0, 0],
            power=1,
        )

        # Measure power in waveguide mode that has been reflected to go the other direction
        overlap_reverse = optplan.WaveguideModeOverlap(
            center=[-(path_length // 2) - 3 * GRID_SPACING, -250, 0],
            extents=[GRID_SPACING, 2500, 600],
            mode_num=0,
            normal=[-1, 0, 0],
            power=1
        )

        # Create the modal overlap at the output waveguide.
        overlap_out = optplan.WaveguideModeOverlap(
            center=[path_length // 2, -250, 0],
            extents=[GRID_SPACING, 2500, 600],
            mode_num=0,
            normal=[1, 0, 0],
            power=1,
        )

This section creates the source of the field (``wg_source``) at the given position ``center`` and size ``extents``. It also creates several ``WaveguideModeOverlap`` s that are needed to measure several things. First, there is one at the output for measuring the output power, and one pointing backwards (``normal=[-1, 0, 0]``) near the input to measure any power reflected into the reverse waveguide mode. Then, there is ``phase_path`` which is a ``Region``, or simple rectangular prism. This and ``overlap_in`` are needed to measure the phase using the ``WaveguidePhase`` class. The ``phase_path`` should extend from the center of ``overlap_in`` to the center of ``overlap_out`` so the phase can be 'unrolled' along it.
