"""Optimizes a Kerr effect process occurring in the center of the device.

This example shows how to optimize a device to create nonlinear optical effects
by maximizing the field intensity in a region of the device.
This is shown diagrmatically below:

        __________
       |          |
    ___| +------+ |___
 in ___  | Kerr |  ___ out
       | +------+ |
       |__________|

By changing the `SIM_2D` global variable, the simulation can be done in either
2D or 3D. 2D simulations are performed on the CPU whereas 3D simulations require
using the GPU-based Maxwell electromagnetic solver.

Note that to run the 3D optimization, the 3D solver must be setup and running
already.

To process the optimization data, see the IPython notebook contained in this
folder, or the monitor_plot_kerr.py file.
"""
from typing import List, Tuple

import numpy as np

from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

# Yee cell grid spacing in nanometers.
GRID_SPACING = 40
# If `True`, perform the simulation in 2D. Else in 3D.
SIM_2D = True
# Silicon refractive index to use for 2D simulations. This should be the
# effective index value.
SI_2D_INDEX = 2.20
# Silicon refractive index to use for 3D simulations.
SI_3D_INDEX = 3.45

WIDTH = 2500            # Width from source to output
NM_TO_M = 1e-9          # Conversion from nm to m
S2M_TO_FS2MM = 1e27     # Conversion from s^2/m to fs^2/mm for GVD
C = 299792458           # Speed of light (m/s)


def main() -> None:
    """Runs the optimization."""
    # Create the simulation space using the GDS files.
    sim_space = create_sim_space("sim_fg.gds", "sim_bg.gds")

    # Setup the objectives and all values that should be recorded (monitors).
    obj, monitors = create_objective(sim_space)

    # Create the list of operations that should be performed during
    # optimization. In this case, we use a series of continuous parametrizations
    # that approximate a discrete structure.
    trans_list = create_transformations(
        obj, monitors, sim_space, cont_iters=100, min_feature=100)

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files). By default, all log files produced
    # during the optimization are also saved here. This can be changed by
    # passing in a third optional argument.
    plan = optplan.OptimizationPlan(transformations=trans_list)
    problem_graph.run_plan(plan, ".")


def create_sim_space(gds_fg: str, gds_bg: str) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation. The material stack is
    220 nm of silicon surrounded by oxide. The refractive index of the silicon
    changes based on whether the global variable `SIM_2D` is set.

    Args:
        gds_fg: Location of the foreground GDS file.
        gds_bg: Location of the background GDS file.

    Returns:
        A `SimulationSpace` description.
    """
    mat_oxide = optplan.Material(mat_name="SiO2")
    if SIM_2D:
        device_index = SI_2D_INDEX
    else:
        device_index = SI_3D_INDEX

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
                foreground=optplan.Material(mat_name="Si"),
                background=mat_oxide,
                gds_layer=[100, 0],
                extents=[-110, 110],
            ),
        ],
    )

    if SIM_2D:
        # If the simulation is 2D, then we just take a slice through the
        # device layer at z = 0. We apply periodic boundary conditions along
        # the z-axis by setting PML thickness to zero.
        sim_region = optplan.Box3d(
            center=[0, 0, 0], extents=[6000, 4000, GRID_SPACING])
        pml_thickness = [10, 10, 10, 10, 0, 0]
    else:
        sim_region = optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 2000])
        pml_thickness = [10, 10, 10, 10, 10, 10]

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


def create_gvd_objective(phase: List[optplan.PhaseAbsolute], d_wavelength) -> optplan.Function:
    """Creates a group velocity dispersion objective function evaluated at a given frequency.

    Args:
        phase: three phase functions that are closely spaced in frequency for calculating differences
        d_wavelength: the change in wavelength between the three phase measurements
    """

    # Convert wavelength to meters
    d_wavelength_m = d_wavelength * NM_TO_M

    # First derivative (up to constant)
    d_phase_1 = phase[1] - phase[0]
    d_phase_2 = phase[2] - phase[1]

    # Second derivative (including all constants)
    gvd_si_units = ((d_wavelength_m / C) ** 2) * (d_phase_2 - d_phase_1) / (WIDTH * NM_TO_M)

    # Convert to traditional units of fs^2/mm
    gvd = gvd_si_units * S2M_TO_FS2MM

    return gvd


def thz_to_nm(thz):
    hz = thz * 1e12
    wavelength = C / hz
    wavelength_nm = wavelength / NM_TO_M

    return round(wavelength_nm)


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

    # Create the waveguide source at the input.
    wg_source = optplan.WaveguideModeSource(
        center=[-2000, 0, 0],
        extents=[GRID_SPACING, 1500, 600],
        normal=[1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    # Create the region in which to optimize the phase in.
    phase_region = optplan.Region(
        center=[2000, 0, 0],
        extents=[GRID_SPACING, 1500, 6*GRID_SPACING],
        power=1,
    )

    # Create a path from the source to the output to track the phase over.
    phase_path = optplan.Region(
        center=[0, 0, 0],
        extents=[4000, GRID_SPACING, GRID_SPACING],
        power=1
    )

    # Create the modal overlap at the output waveguide.
    overlap_out = optplan.WaveguideModeOverlap(
        center=[2000, 0, 0],
        extents=[GRID_SPACING, 1500, 600],
        mode_num=0,
        normal=[1, 0, 0],
        power=1,
    )

    # Keep track of metrics and fields that we want to monitor.
    gvd_list = []
    power_objs = []
    monitor_list = []

    # Set the wavelengths, wavelength differences, and goal GVDs to simulate and optimize
    d_wavelength = 20
    optimization_wavelength = [thz_to_nm(185), thz_to_nm(195), thz_to_nm(205)]
    print(optimization_wavelength)
    optimization_gvd = [50, -5, 50]

    # Calculate the GVD at each wavelength
    for center_wavelength in optimization_wavelength:

        # Use three different wavelengths, spaced by d_wavelength, to approximate the GVD
        gvd_wavelengths = (center_wavelength - d_wavelength, center_wavelength, center_wavelength + d_wavelength)

        center_sim = None
        center_epsilon = None

        # Keep track of resulting phases for GVD calculation
        gvd_phases = []

        # Simulate at each of the three wavelengths, and save field and epsilon info for the middle
        # wavelength for things that do not depend strongly on wavelength (like output power)
        for center, sim_wavelength in zip([False, True, False], gvd_wavelengths):

            epsilon = optplan.Epsilon(
                simulation_space=sim_space,
                wavelength=sim_wavelength,
            )

            sim = optplan.FdfdSimulation(
                source=wg_source,
                # Use a direct matrix solver (e.g. LU-factorization) on CPU for
                # 2D simulations and the GPU Maxwell solver for 3D.
                solver="local_direct" if SIM_2D else "maxwell_cg",
                wavelength=sim_wavelength,
                simulation_space=sim_space,
                epsilon=epsilon,
            )

            # Save center wavelength simulation objects
            if center:
                center_sim = sim
                center_epsilon = epsilon

            # Create phase objectives and monitors
            phase = optplan.PhaseAbsolute(simulation=sim, region=phase_region, path=phase_path)
            monitor_list.append(optplan.SimpleMonitor(name="phase{}".format(sim_wavelength), function=phase))
            gvd_phases.append(phase)

        # Calculate GVD from the three phases at three wavelengths
        gvd = create_gvd_objective(gvd_phases, d_wavelength)

        # Store GVD function and add to monitor list
        gvd_list.append(gvd)
        monitor_list.append(optplan.SimpleMonitor(name="GVD{}".format(center_wavelength), function=gvd))

        # Add the field at the center wavelength to the monitor list
        monitor_list.append(
            optplan.FieldMonitor(
                name="field{}".format(center_wavelength),
                function=center_sim,
                normal=[0, 0, 1],
                center=[0, 0, 0],
            ))

        # Only store epsilon information once because it is the same at each wavelength
        if center_wavelength == optimization_wavelength[len(optimization_wavelength) // 2]:
            monitor_list.append(
                optplan.FieldMonitor(
                    name="epsilon",
                    function=center_epsilon,
                    normal=[0, 0, 1],
                    center=[0, 0, 0]))

        # Create output power objectives and monitors
        overlap_out_obj = optplan.Overlap(simulation=center_sim, overlap=overlap_out)
        power_out = optplan.abs(overlap_out_obj)**2
        power_objs.append(power_out)
        monitor_list.append(optplan.SimpleMonitor(name="powerOut{}".format(center_wavelength), function=power_out))

    # Spins minimizes the objective function, so to make `power` maximized,
    # we minimize `1 - power`.
    obj = 0
    for power in power_objs:
        obj += (1 - power) ** 2

    # Minimize distance between simulated GVD and goal GVD at each wavelength
    for goal, gvd in zip(optimization_gvd, gvd_list):
        obj += 0.01 * optplan.abs(goal - gvd) ** 2

    monitor_list.append(optplan.SimpleMonitor(name="objective", function=obj))

    return obj, monitor_list


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
        init_method=optplan.UniformInitializer(min_val=0.6, max_val=0.9),
    )

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
                        value=4 * (stage + 1)),
                ))
    return trans_list


if __name__ == "__main__":
    main()
