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

.. code:: python

        # Keep track of metrics and fields that we want to monitor.
        k_list = []
        power_forward_objs = []
        power_reverse_objs = []
        monitor_list = []
        yaml_phase_monitors = []
        yaml_field_monitors = []
        yaml_power_monitors = []
        yaml_scalar_monitors = []
        yaml_epsilon_monitors = []
        yaml_spec = {'monitor_list': []}

        # Set the wavelengths, wavelength differences, and goal GVDs to simulate and optimize
        optimization_frequencies, frequency_step = np.linspace(start=160, stop=220, num=31, retstep=True)

        # Calculate the GVD at each wavelength
        for frequency in optimization_frequencies:
            sim_wavelength = thz_to_nm(frequency)

            epsilon = optplan.Epsilon(
                simulation_space=sim_space,
                wavelength=sim_wavelength,
            )
            yaml_epsilon_monitors.append('{} THz Epsilon'.format(frequency))
            monitor_list.append(optplan.FieldMonitor(name='{} THz Epsilon'.format(frequency),
                                                     function=epsilon,
                                                     normal=[0, 0, 1],
                                                     center=[0, 0, 0]))

            sim = optplan.FdfdSimulation(
                source=wg_source,
                # Use a direct matrix solver (e.g. LU-factorization) on CPU for
                # 2D simulations and the GPU Maxwell solver for 3D.
                solver="local_direct" if SIM_2D else "maxwell_cg",
                wavelength=sim_wavelength,
                simulation_space=sim_space,
                epsilon=epsilon,
            )
                       
This part of the loop creates a bew ``FdfdSimulation`` and several monitors for each frequency in ``optimization_frequencies``.

.. code:: python

            # Create wave vector objectives and monitors
            phase = optplan.WaveguidePhase(simulation=sim, overlap_in=overlap_in, overlap_out=overlap_out, path=phase_path)
            k = optplan.abs(phase / (path_length * NM_TO_M))  # Path length is in nanometers

            monitor_list.append(optplan.SimpleMonitor(name="{} THz Wave Vector".format(frequency), function=k))
            # yaml_scalar_monitors.append('{} THz Wave Vector'.format(frequency))
            k_list.append(k)

Here, we make a ``WaveguidePhase`` monitor, and then divide that by the length of the path between the input and output regions where the phase is being measured to approximate the wavenumber of the field ``k``.

.. code:: python

            # Add the field to the monitor list
            monitor_list.append(
                optplan.FieldMonitor(
                    name="{} THz Field".format(frequency),
                    function=sim,
                    normal=[0, 0, 1],
                    center=[0, 0, 0],
                ))
            yaml_field_monitors.append('{} THz Field'.format(frequency))
            yaml_phase_monitors.append('{} THz Field'.format(frequency))

            # Only store epsilon information once because it is the same at each wavelength
            if frequency == optimization_frequencies[len(optimization_frequencies) // 2]:
                monitor_list.append(
                    optplan.FieldMonitor(
                        name="Epsilon",
                        function=epsilon,
                        normal=[0, 0, 1],
                        center=[0, 0, 0]))

This section of the loop is storing the fields and the epsilon distribution so we can look at them later.

.. code:: python

            # Create output power objectives and monitors
            overlap_out_obj = optplan.Overlap(simulation=sim, overlap=overlap_out)
            overlap_reverse_obj = optplan.Overlap(simulation=sim, overlap=overlap_reverse)
            power_out = optplan.abs(overlap_out_obj) ** 2
            power_reverse = optplan.abs(overlap_reverse_obj) ** 2
            power_forward_objs.append(power_out)
            power_reverse_objs.append(power_reverse)
            monitor_list.append(optplan.SimpleMonitor(name="{} THz Power Out".format(frequency), function=power_out))
            monitor_list.append(optplan.SimpleMonitor(name="{} THz Power Reverse".format(frequency),
                                                      function=power_reverse))
            yaml_power_monitors.append('{} THz Power Out'.format(frequency))
            yaml_power_monitors.append('{} THz Power Reverse'.format(frequency))

Here, we are making parts of the objective we want to minimize. We define objectives for the power in the forward and backward waveguide modes at each frequency.

.. code:: python

        # Calculate and store GVD functions and add to monitor list
        gvd_list = create_gvd_multiple_difference(k_list, frequency_step * THZ_TO_HZ, ignore_endpoints=True)
        for gvd, frequency in zip(gvd_list, optimization_frequencies[1:-1]):
            monitor_list.append(optplan.SimpleMonitor(name="{} THz GVD".format(frequency), function=gvd))
            yaml_scalar_monitors.append('{} THz GVD'.format(frequency))

Using the finite difference GVD function, we can use the list of wavenumbers ``k`` for each frequency to approximate the GVD of the waveguide. Setting ``ignore_endpoints`` to ``True`` excludes the first and last GVD values from the results, as they are the hardest to approximate and least reliable.

.. code:: python

        # Spins minimizes the objective function, so to make `power` maximized,
        # we minimize `1 - power`.
        loss_obj = 0
        edge_obj = 0
        resonance_obj = 0
        for power_fwd, power_rev in zip(power_forward_objs, power_reverse_objs):
            loss_obj += (1 - power_fwd - power_rev) ** 2
        for power_fwd in power_forward_objs[:12] + power_forward_objs[18:]:
            edge_obj += (1 - power_fwd) ** 2
        for power_rev in power_reverse_objs[:12] + power_reverse_objs[18:]:
            edge_obj += power_rev ** 2
        for power_fwd in power_forward_objs[14:16]:
            resonance_obj += (0.8 - power_fwd) ** 2
        for power_rev in power_reverse_objs[14:16]:
            resonance_obj += (0.2 - power_rev) ** 2

        monitor_list.append(optplan.SimpleMonitor(name="Loss Objective", function=loss_obj))
        monitor_list.append(optplan.SimpleMonitor(name="Transmission Objective", function=edge_obj))
        monitor_list.append(optplan.SimpleMonitor(name="Resonance Transmission Objective", function=resonance_obj))

        obj = 1E3 * loss_obj + 1E2 * edge_obj + 50 * resonance_obj

        monitor_list.append(optplan.SimpleMonitor(name="Objective", function=obj))
        yaml_scalar_monitors.append('Objective')

This is where the overall objective is defined. By adding the sub-objectives in quadrature, we can encourage SPINS to optimize all of the simultaneously. Picking the coefficients in front of each one is important to balance the weight of each part of the objective appropriately so none are ignored.

.. code:: python

        for monitor in yaml_power_monitors:
            yaml_spec['monitor_list'].append({'monitor_names':    [monitor],
                                              'monitor_type':     'scalar',
                                              'scalar_operation': 'magnitude_squared'})
        for monitor in yaml_field_monitors:
            yaml_spec['monitor_list'].append({'monitor_names':    [monitor],
                                              'monitor_type':     'planar',
                                              'vector_operation': 'magnitude'})
        for monitor in yaml_phase_monitors:
            yaml_spec['monitor_list'].append({'monitor_names':    [monitor],
                                              'monitor_type':     'planar',
                                              'vector_operation': 'z',
                                              'scalar_operation': 'phase'})
        for monitor in yaml_scalar_monitors:
            yaml_spec['monitor_list'].append({'monitor_names': [monitor],
                                              'monitor_type':  'scalar'})

        yaml_spec['monitor_list'].append({'monitor_names':    ['Epsilon'],
                                          'monitor_type':     'planar',
                                          'vector_operation': 'z'})

        with open('monitor_spec_dynamic.yml', 'w') as monitor_spec_dynamic:
            yaml.dump(yaml_spec, monitor_spec_dynamic, default_flow_style=False)
        
        return obj, monitor_list

The last part of the function creates the file ``monitor_spec_dynamic.yml`` automatically. This file can be read in by the ``monitor_plot.py`` file to print out graphs of all of the interesting monitors and objectives after completing the optimization routine.

.. code:: python

    def create_transformations(
            obj: optplan.Function,
            monitors: List[optplan.Monitor],
            sim_space: optplan.SimulationSpaceBase,
            cont_iters: int,
            num_stages: int = 3,
            min_feature: float = 100,
    ) -> List[optplan.Transformation]:
        """Creates a list of transformations for the device optimization.

        The transformations dictate the sequence of steps used to optimize the
        device. The optimization uses `num_stages` of continuous optimization. For
        each stage, the "discreteness" of the structure is increased (through
        controlling a parameter of a sigmoid function).

        Args:
            obj: The objective function to minimize.
            monitors: List of monitors to keep track of.
            sim_space: Simulation space ot use.
            cont_iters: Number of iterations to run in continuous optimization
                total across all stages.
            num_stages: Number of continuous stages to run. The more stages that
                are run, the more discrete the structure will become.
            min_feature: Minimum feature size in nanometers.
            fab_obj: Objective including fabrication penalties to be used in the
                second half of the optimization

        Returns:
            A list of transformations.
        """
        # Setup empty transformation list.
        trans_list = []

        # First do continuous relaxation optimization.
        # This is done through cubic interpolation and then applying a sigmoid
        # function.
        param = optplan.CubicParametrization(
            # Specify the coarseness of the cubic interpolation points in terms
            # of number of Yee cells. Feature size is approximated by having
            # control points on the order of `min_feature / GRID_SPACING`.
            undersample=3.5 * min_feature / GRID_SPACING,
            simulation_space=sim_space,
            # init_method=optplan.WaveguideInitializer3(lower_min=0, lower_max=.4, upper_min=.7, upper_max=1,
            #                                           extent_frac_x=1, extent_frac_y=1/2,
            #                                           center_frac_x=1/2, center_frac_y=1/8),
            init_method=optplan.GradientInitializer(min=0, max=1, random=0.3, extent_frac_x=1, extent_frac_y=0.4,
                                                    center_frac_x=0.5, center_frac_y=0.55)
            # init_method=optplan.UniformInitializer(min_val=0, max_val=0)
            # init_method=optplan.PeriodicInitializer(random=0.2, min=0, max=1, period=400, sim_width=6000,
            #                                         center_frac_y=0.5, extent_frac_y=0.4)
        )

This function creates the sequence of optimizations that the program will run. It also initializes the epsilon distribution. Here are four ways to initialize the distribution, three of which are commented out at the moment.

.. code:: python

    trans_list.append(
            optplan.Transformation(
                name="sigmoid_change_power_init",
                parametrization=param,
                # The larger the sigmoid strength value, the more "discrete"
                # structure will be.
                transformation=optplan.CubicParamSigmoidStrength(
                    value=2,
                )))

        iters = max(cont_iters // num_stages, 1)
        for stage in range(num_stages):
            trans_list.append(
                optplan.Transformation(
                    name="opt_cont{}".format(stage),
                    parametrization=param,
                    transformation=optplan.ScipyOptimizerTransformation(
                        optimizer="L-BFGS-B",
                        objective=obj,
                        monitor_lists=optplan.ScipyOptimizerMonitorList(
                            callback_monitors=monitors,
                            start_monitors=monitors,
                            end_monitors=monitors),
                        optimization_options=optplan.ScipyOptimizerOptions(
                            maxiter=iters),
                    ),
                ))

            if stage < num_stages - 1:
                # Make the structure more discrete.
                trans_list.append(
                    optplan.Transformation(
                        name="sigmoid_change{}".format(stage),
                        parametrization=param,
                        # The larger the sigmoid strength value, the more "discrete"
                        # structure will be.
                        transformation=optplan.CubicParamSigmoidStrength(
                            value=2 * (stage + 2)),
                    ))
        return trans_list

The last bit of code creates the different stages of optimization. The main thing here is setting the ``CubicParamSigmoidStrength`` for each stage, the number of iterations for each stage, and the number of stages. The ``CubicParamSigmoidStrength`` sets how 'smooth' the epsilon distribution is. SPINS-B only optimizes a continuous epsilon distribution, but increasing this value will make the distribution appear more binary. A value of 10 or so results in a distribution that is a very good approximation of binary, but it is best to start with a much lower value and increase over time.
