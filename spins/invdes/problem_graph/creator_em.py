import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse

from spins import fdfd_solvers
from spins import fdfd_tools
from spins import gridlock
from spins.invdes import problem
from spins.fdfd_solvers import local_matrix_solvers
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
# Make a style guide exception here because `simspace` is already used as a
# variable.
from spins.invdes.problem_graph.simspace import SimulationSpace

# Have a single shared direct solver object because we need to use
# multiprocessing to actually parallelize the solve.
DIRECT_SOLVER = local_matrix_solvers.MultiprocessingSolver(
    local_matrix_solvers.DirectSolver())


@optplan.register_node(optplan.WaveguideModeSource)
class WaveguideModeSource:

    def __init__(self,
                 params: optplan.WaveguideModeSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a new waveguide mode source.

        Args:
            params: Waveguide source parameters.
            work: Workspace for this source. Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        """Creates the source vector.

        Args:
            simspace: Simulation space object to use.
            wlen: Wavelength to operate source.

        Returns:
            The vector field corresponding to the source.
        """
        space_inst = simspace(wlen)
        return fdfd_solvers.waveguide_mode.build_waveguide_source(
            omega=2 * np.pi / wlen,
            dxes=simspace.dxes,
            eps=space_inst.eps_bg.grids,
            mu=None,
            mode_num=self._params.mode_num,
            waveguide_slice=grid_utils.create_region_slices(
                simspace.edge_coords, self._params.center,
                self._params.extents),
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            power=self._params.power)


@optplan.register_node(optplan.PlaneWaveSource)
class PlaneWaveSource:

    def __init__(self,
                 params: optplan.PlaneWaveSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a plane wave source.

        Args:
            params: Parameters for the plane wave source.
            work: Unused.
        """
        self._params = params

    def __call__(
            self, simspace: SimulationSpace, wlen: float, **kwargs
    ) -> Union[fdfd_tools.VecField, Tuple[fdfd_tools.VecField, fdfd_tools.
                                          Vec3d]]:
        """Creates the plane wave source.

        Args:
            simspace: Simulation space to use for the source.
            wlen: Wavelength of source.

        Returns:
            If `overwrite_bloch_vector` is `True`, a tuple containing the source
            field and the Bloch vector corresponding to the plane wave source.
            Otherwise, only the source field is returned.
        """
        space_inst = simspace(wlen)

        # Calculate the border in gridpoints and igore the border if it's larger then the simulation.
        dx = simspace.dx
        border = [int(b // dx) for b in self._params.border]
        # The plane wave is assumed to be in the z direction so the border is 0 for z.
        border.append(0)

        source, kvector = fdfd_tools.free_space_sources.build_plane_wave_source(
            omega=2 * np.pi / wlen,
            eps_grid=space_inst.eps_bg,
            mu=None,
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            slices=grid_utils.create_region_slices(simspace.edge_coords,
                                                   self._params.center,
                                                   self._params.extents),
            theta=self._params.theta,
            psi=self._params.psi,
            polarization_angle=self._params.polarization_angle,
            border=border,
            power=self._params.power)

        if self._params.overwrite_bloch_vector:
            return source, kvector
        return source


@optplan.register_node(optplan.GaussianSource)
class GaussianSource:

    def __init__(self,
                 params: optplan.GaussianSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a Gaussian beam source.

        Args:
            params: Gaussian beam source parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float, solver: Callable,
                 **kwargs) -> fdfd_tools.VecField:
        """Creates the source vector.

        Args:
            simspace: Simulation space.
            wlen: Wavelength of source.
            solver: If `normalize_by_source` is `True`, `solver` will be used
                to run an EM simulation to normalize the source power.

        Returns:
            The source.
        """
        space_inst = simspace(wlen)
        source, _ = fdfd_tools.free_space_sources.build_gaussian_source(
            omega=2 * np.pi / wlen,
            eps_grid=space_inst.eps_bg,
            mu=None,
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            slices=grid_utils.create_region_slices(simspace.edge_coords,
                                                   self._params.center,
                                                   self._params.extents),
            theta=self._params.theta,
            psi=self._params.psi,
            polarization_angle=self._params.polarization_angle,
            w0=self._params.w0,
            center=self._params.center,
            power=self._params.power)

        if self._params.normalize_by_sim:
            source = fdfd_tools.free_space_sources.normalize_source_by_sim(
                omega=2 * np.pi / wlen,
                source=source,
                eps=space_inst.eps_bg.grids,
                dxes=simspace.dxes,
                pml_layers=simspace.pml_layers,
                solver=solver,
                power=self._params.power)

        return source


class FdfdSimulation(problem.OptimizationFunction):
    """Represents a FDFD simulation.

    Simulations are cached so that repeated calls with the same permittivity
    distribution does not incur multiple simulations. However, this cache is not
    thread-safe.
    """

    def __init__(
            self,
            eps: problem.OptimizationFunction,
            solver: Callable,
            wlen: float,
            source: np.ndarray,
            simspace: SimulationSpace,
            bloch_vector: Optional[fdfd_tools.Vec3d] = None,
            cache_size: int = 1,
    ) -> None:
        """Creates a FDFD simulation.

        Args:
            eps: Permittivity distribution to simulate.
            solver: Electromagnetic solver to use.
            wlen: Wavelength of simulation.
            source: Vector corresponding to the source of the simulation.
            simspace: Simulation space.
            bloch_vector: Bloch vector to use.
            cache_size: Size of cache used to store adjoint and forward fields.
                This should normally be `1`.
        """
        super().__init__(eps, heavy_compute=True)

        self._solver = solver
        self._wlen = wlen
        self._source = source
        self._simspace = simspace
        self._bloch_vector = bloch_vector

        # For caching uses.
        self._cache = [None] * cache_size
        self._cache_adjoint = [None] * cache_size

    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """Runs the simulation.

        Args:
            input_vals: List with single element corresponding to the
                permittivity distribution.

        Returns:
            Simulated fields.
        """
        return self._simulate(input_val[0])

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Computes gradient via a adjoint calculation.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        omega = 2 * np.pi / self._wlen
        efields = self._simulate(input_vals[0])
        B = omega**2 * scipy.sparse.diags(efields, 0)
        d = self._simulate_adjoint(input_vals[0],
                                   np.conj(grad_val) / (-1j * omega))
        total_df_dz = np.conj(np.transpose(d)) @ B
        # If this is a function that maps from real to complex, we have to
        # to take the real part to make gradient real.
        if np.isrealobj(input_vals[0]):
            total_df_dz = np.real(total_df_dz)

        return [total_df_dz]

    def _simulate(self, eps: np.ndarray) -> np.ndarray:
        """Computes the electric field distribution.

        Because simulations are very expensive, we cache the simulations.

        Args:
            eps: The structure.

        Returns:
            Vectorized form of the electric fields.
        """
        # Only solve for the fields if the structure has changed.
        electric_fields = None
        # The cache is implemented as a list where the most recent
        # access is at the back of the list.

        # Search through cache for fields.
        for cache_index in range(len(self._cache)):
            cache_item = self._cache[cache_index]
            if cache_item is None:
                continue
            cache_struc, cache_fields = cache_item
            if np.array_equal(eps, cache_struc):
                electric_fields = cache_fields
                # Remove the hit entry (it will be reinserted later).
                del self._cache[cache_index]
                break

        if electric_fields is None:
            # Perfrom the solve.
            electric_fields = self._solver.solve(
                omega=2 * np.pi / self._wlen,
                dxes=self._simspace.dxes,
                epsilon=eps,
                mu=None,
                J=fdfd_tools.vec(self._source),
                pml_layers=self._simspace.pml_layers,
                bloch_vec=self._bloch_vector,
            )
            # Remove the last used element.
            del self._cache[0]

        # Insert data into cache.
        self._cache.append((eps, electric_fields))
        return electric_fields

    def _simulate_adjoint(self, eps: np.ndarray,
                          source: np.ndarray) -> np.ndarray:
        """Computes an adjoint simulation.

        Args:
            eps: The structure.
            source: The excitation current.

        Returns:
            Vectorized form of the electric fields.
        """
        # Only solve for the fields if the structure has changed.
        electric_fields = None
        # The cache is implemented as a list where the most recent
        # access is at the back of the list.

        # Search through cache for fields.
        for cache_index in range(len(self._cache_adjoint)):
            cache_item = self._cache_adjoint[cache_index]
            if cache_item is None:
                continue
            cache_struc, cache_source, cache_fields = cache_item
            if (np.array_equal(eps, cache_struc) and
                    np.array_equal(source, cache_source)):
                electric_fields = cache_fields
                # Remove the hit entry (it will be reinserted later).
                del self._cache_adjoint[cache_index]
                break

        if electric_fields is None:
            electric_fields = self._solver.solve(
                omega=2 * np.pi / self._wlen,
                dxes=self._simspace.dxes,
                epsilon=eps,
                mu=None,
                J=source,
                pml_layers=self._simspace.pml_layers,
                bloch_vec=self._bloch_vector,
                adjoint=True,
            )

            # Remove the last used element.
            del self._cache_adjoint[0]

        # Insert data into cache.
        self._cache_adjoint.append((eps, source, electric_fields))

        return electric_fields

    def __str__(self):
        return "Simulation({})".format(self._wlen)


def _create_solver(solver_name: str, simspace: SimulationSpace) -> Callable:
    """Instantiates a Maxwell solver.

    Args:
        solver_name: Name of the solver.
        simspace: Simulation space.

    Returns:
         A callable solver object.
    """
    if solver_name == "maxwell_cg":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(simspace.dims, solver="CG")
    elif solver_name == "maxwell_bicgstab":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(simspace.dims, solver="biCGSTAB")
    elif solver_name == "local_direct":
        solver = DIRECT_SOLVER
    else:
        raise ValueError("Unknown solver, got {}".format(solver_name))

    return solver


@optplan.register_node(optplan.FdfdSimulation)
def create_fdfd_simulation(params: optplan.FdfdSimulation,
                           work: workspace.Workspace) -> FdfdSimulation:
    """Creates a `FdfdSimulation` object."""
    simspace = work.get_object(params.simulation_space)
    solver = _create_solver(params.solver, simspace)
    bloch_vector = params.get("bloch_vector", np.zeros(3))

    source = work.get_object(params.source)(
        simspace, params.wavelength, solver=solver)
    if isinstance(source, tuple):
        source, bloch_vector = source

    return FdfdSimulation(
        eps=work.get_object(params.epsilon),
        solver=solver,
        wlen=params.wavelength,
        source=source,
        simspace=simspace,
        bloch_vector=bloch_vector,
        cache_size=1)


class Epsilon(problem.OptimizationFunction):
    """Represents the permittivity distribution.

    This is a particular instantiation of the permittivity distribution
    described by `SimulationSapce`.
    """

    def __init__(
            self,
            input_function: problem.OptimizationFunction,
            wlen: float,
            simspace: SimulationSpace,
    ) -> None:
        """Creates a FDFD simulation Optimization function.

        Args:
            input_function: Input function corresponding to the structure.
                This should be a vector compatible with the selection matrix.
            wlen: Wavelength to evaluate permittivity.
            simspace: Simulation space from which to get permittivity.
        """
        super().__init__(input_function)

        self._wlen = wlen
        self._space = simspace(wlen)

    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """Returns simulated fields.

        Args:
            input_vals: List of the input values.

        Returns:
            Simulated fields.
        """
        return (fdfd_tools.vec(self._space.eps_bg.grids) +
                self._space.selection_matrix @ input_val[0])

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns gradient of the epsilon calculation.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        # In the backprop calculation, this should really be `2 * np.real(...)`.
        # However, because in our backprop calculations, we know that eventually
        # the complex value will hit `AbsoluteValue`, which is implemented
        # assuming that the input value is real (i.e.
        # `d abs(x)/dx = conj(x)/|x|` as opposed to `d abs(x)/dx = conj(x)/2|x|`
        # so the factor of 2 is cancelled out.
        return [
            np.real(
                np.squeeze(np.asarray(grad_val @ self._space.selection_matrix)))
        ]

    def __str__(self):
        return "Epsilon({})".format(self._wlen)


@optplan.register_node(optplan.Epsilon)
def create_epsilon(params: optplan.Epsilon,
                   work: workspace.Workspace) -> Epsilon:
    return Epsilon(
        input_function=work.get_object(workspace.VARIABLE_NODE),
        wlen=params.wavelength,
        simspace=work.get_object(params.simulation_space))


@optplan.register_node(optplan.WaveguideModeOverlap)
class WaveguideModeOverlap:

    def __init__(self,
                 params: optplan.WaveguideModeOverlap,
                 work: workspace.Workspace = None) -> None:
        """Creates a new waveguide mode overlap.

        Args:
            params: Waveguide mode parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        space_inst = simspace(wlen)
        return fdfd_solvers.waveguide_mode.build_overlap(
            omega=2 * np.pi / wlen,
            dxes=simspace.dxes,
            eps=space_inst.eps_bg.grids,
            mu=None,
            mode_num=self._params.mode_num,
            waveguide_slice=grid_utils.create_region_slices(
                simspace.edge_coords, self._params.center,
                self._params.extents),
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            power=self._params.power)


@optplan.register_node(optplan.KerrOverlap)
class KerrOverlap:

    def __init__(self,
                 params: optplan.KerrOverlap,
                 work: workspace.Workspace = None) -> None:
        """Creates a new Kerr overlap.

        Args:
            params: Kerr intensity optimization region parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        space_inst = simspace(wlen)
        region_slice = grid_utils.create_region_slices(
            simspace.edge_coords, self._params.center,
            self._params.extents)
        eps = space_inst.eps_bg.grids
        eps_min = eps.min()
        eps_max = eps.max()
        eps_norm = (eps - eps_min) / (eps_max - eps_min) + 1

        overlap = [None] * 3
        for i in range(3):
            overlap[i] = np.zeros_like(eps_norm[0], dtype=complex)
            overlap_i = overlap[i]
            eps_i = eps_norm[i]
            eps_i_slice = eps_i[tuple(region_slice)]
            overlap_i[tuple(region_slice)] = eps_i_slice

            overlap[i] = overlap_i

        return np.multiply(overlap, self._params.power)


@optplan.register_node(optplan.Region)
class Region:
    def __init__(self,
                 params: optplan.Region,
                 work: workspace.Workspace = None) -> None:
        """Creates a new region.

        Args:
            params: region parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        space_inst = simspace(wlen)
        region_slice = grid_utils.create_region_slices(
            simspace.edge_coords, self._params.center,
            self._params.extents)
        eps = space_inst.eps_bg.grids

        region = [None] * 3
        for i in range(3):
            region[i] = np.zeros_like(eps[0], dtype=complex)
            region_i = region[i]
            ones_i = np.ones_like(eps, dtype=complex)[i]
            ones_i_slice = ones_i[tuple(region_slice)]
            region_i[tuple(region_slice)] = ones_i_slice

            region[i] = region_i

        return np.multiply(region, self._params.power)


# TODO(logansu): This function appears just to be an inner product.
# Why is this a separate function right now?
class OverlapFunction(problem.OptimizationFunction):
    """Represents an optimization function for overlap."""

    def __init__(self, input_function: problem.OptimizationFunction,
                 overlap: np.ndarray):
        """Constructs the objective C*x.

        Args:
            input_function: Input objectives(typically a simulation).
            overlap: Vector to overlap with.
        """
        super().__init__(input_function)

        self._input = input_function
        self.overlap_vector = overlap

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Vector product of overlap and the input.
        """

        return self.overlap_vector @ input_vals[0]

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            gradient.
        """
        return [grad_val * self.overlap_vector]

    def __str__(self):
        return "Overlap({})".format(self._input)


@optplan.register_node(optplan.Overlap)
def create_overlap_function(params: optplan.ProblemGraphNode,
                            work: workspace.Workspace):
    simspace = work.get_object(params.simulation.simulation_space)
    wlen = params.simulation.wavelength
    overlap = fdfd_tools.vec(work.get_object(params.overlap)(simspace, wlen))
    return OverlapFunction(
        input_function=work.get_object(params.simulation), overlap=overlap)


class OverlapIntensityFunction(problem.OptimizationFunction):
    """Represents an optimization function for overlap of field intensity |E|^2."""

    def __init__(self, input_function: problem.OptimizationFunction,
                 overlap: np.ndarray):
        """Constructs the objective C*x.

        Args:
            input_function: Input objectives (typically a simulation).
            overlap: Vector to overlap with its intensity.
        """
        super().__init__(input_function)

        self._input = input_function
        self.overlap_vector = overlap

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Vector product of overlap and the input.
        """
        return np.dot(self.overlap_vector, input_vals[0]) / len(self.overlap_vector)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            gradient.
        """
        return [grad_val * self.overlap_vector]

    def __str__(self):
        return "OverlapIntensity({})".format(self._input)


@optplan.register_node(optplan.OverlapIntensity)
def create_overlap_intensity_function(params: optplan.ProblemGraphNode,
                                      work: workspace.Workspace):
    simspace = work.get_object(params.simulation.simulation_space)
    wlen = params.simulation.wavelength
    overlap = fdfd_tools.vec(work.get_object(params.overlap)(simspace, wlen))
    return OverlapIntensityFunction(
        input_function=work.get_object(params.simulation), overlap=overlap)


def _get_vector_components(field):
    third = len(field) // 3

    x = field[:third]
    y = field[third:2*third]
    z = field[2*third:]

    return [x, y, z]


class WaveguidePhaseFunction(problem.OptimizationFunction):
    """Represents an optimization function for the phase of the field in a waveguide mode."""

    def __init__(self, input_function: problem.OptimizationFunction,
                 overlap_in: np.array, overlap_out: np.array, path: np.array, wavelength):
        """Constructs the objective.

        Args:
            input_function: Input objectives (typically a simulation).
            overlap: WaveguideModeOverlap of the mode of interest
            path: path from the source along which to take the phase difference
        """
        super().__init__(input_function)

        self._input = input_function
        self.overlap_in = overlap_in
        self.overlap_out = overlap_out
        self.path = path
        self.wavelength = wavelength
        self.overlap_in_function = OverlapFunction(input_function, overlap_in)
        self.overlap_out_function = OverlapFunction(input_function, overlap_out)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input field values.

        Returns:
            Phase difference between the output waveguide mode and the beginning of the path.
        """

        # TODO(Kyle DeBry): these are just for debugging; remove
        overlap_out_vec = _get_vector_components(self.overlap_out[np.nonzero(self.overlap_out)])
        overlap_out_vec = _get_vector_components(self.overlap_out[np.nonzero(self.overlap_out)])

        # Get phase of overlap with waveguide modes
        waveguide_in_phase = np.angle(self.overlap_in_function.eval(input_vals))
        waveguide_out_raw_phase = np.angle(self.overlap_out_function.eval(input_vals))

        # Get cumulative phase difference along the path from the source to the waveguide mode location
        path_vals = input_vals[0][np.nonzero(self.path)]
        path_component_vals = [_get_vector_components(path_vals)[i][0:] for i in range(3)]
        path_raw_phase = np.angle(path_vals)
        path_raw_phase_components = _get_vector_components(path_raw_phase)
        path_phase_components = [np.unwrap(path_raw_phase_components[i][0:]) for i in range(3)]
        path_phase_weighted = np.sum([path_phase_components[i] * np.abs(path_component_vals)[i] for i in range(3)],
                                     axis=0)
        path_phase_norm = path_phase_weighted / np.sum(np.abs(path_component_vals), axis=0)

        # Patch together the phase along the path and the waveguide mode phases to get total phase difference
        # Note: tau = 2*pi is the one true circle constant
        waveguide_in_phase_jump = path_phase_norm[0] - waveguide_in_phase
        path_phase_norm -= waveguide_in_phase_jump
        waveguide_out_phase_jump = waveguide_out_raw_phase - path_phase_norm[-1]
        out_jump_num_tau = round(waveguide_out_phase_jump / math.tau)
        waveguide_out_phase = waveguide_out_raw_phase - out_jump_num_tau * math.tau

        return waveguide_out_phase - waveguide_in_phase

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input E field values.
            grad_val: Gradient of the input to the phase function (for chain rule).

        Returns:
            gradient.
        """

        # This is a composite function. Outer level is phase(overlap_result), overlap_result is a
        # function of the input values, and the input values have their own derivative
        # Final result is d(phase(overlap_result(input_vals(x)))) / dx
        #   = d(phase) / d(overlap_result) * d(overlap_result) / d(input_vals) * d(input_vals) / dx

        # Gradient of input value (last term)
        input_grad = grad_val

        # Get the overlap values (middle term)
        overlap_result = self.overlap_out_function.eval(input_vals)
        overlap_grad = self.overlap_out_function.grad(input_vals, np.array(1))[0]

        # Compute d(phase)/d(overlap_result) (first term)
        overlap_re = np.real(overlap_result)
        overlap_im = np.imag(overlap_result)
        overlap_abs = np.abs(overlap_result)

        # Should be divided by overlap_abs^2, but that causes singularity at 0. Removing one factor of
        # overlap_abs prevents the grad for blowing up for small field values. However, leaving it in
        # here for now because overlap is usually not super small, so it might be OK, as opposed to
        # the field value in one grid cell which is often very small.
        phase_grad_re = - overlap_im / overlap_abs ** 2
        phase_grad_im = overlap_re / overlap_abs ** 2
        phase_grad = (phase_grad_re + 1j*phase_grad_im)

        # Chain rule
        grad = phase_grad * overlap_grad * input_grad

        return [grad]


@optplan.register_node(optplan.WaveguidePhase)
def create_waveguide_phase_function(params: optplan.ProblemGraphNode,
                                    work: workspace.Workspace):
    sim_space = work.get_object(params.simulation.simulation_space)
    wavelength = params.simulation.wavelength
    overlap_in = fdfd_tools.vec(work.get_object(params.overlap_in)(sim_space, wavelength))
    overlap_out = fdfd_tools.vec(work.get_object(params.overlap_out)(sim_space, wavelength))
    path = fdfd_tools.vec(work.get_object(params.path)(sim_space, wavelength))
    return WaveguidePhaseFunction(
        input_function=work.get_object(params.simulation), overlap_in=overlap_in, overlap_out=overlap_out,
        path=path, wavelength=wavelength)


class PhaseAverageFunction(problem.OptimizationFunction):
    """Represents an optimization function for the average difference in phase of the field."""

    def __init__(self, input_function: problem.OptimizationFunction,
                 region: np.array, path: np.array, wavelength):
        """Constructs the objective C*x.

        Args:
            input_function: Input objectives (typically a simulation).
            region: area in which to optimize the phase.
        """
        super().__init__(input_function)

        self._input = input_function
        self.region = region
        self.path = path
        self.center_phases = [0, 0, 0]
        self.old_input = []
        self.current_phase_avg = None
        self.wavelength = wavelength

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Phase difference between the output region and the beginning of the path.
        """

        # Check cache
        if np.array_equal(self.old_input, input_vals):
            return self.current_phase_avg
        else:
            self.old_input = input_vals

        # Get the phase of each component of the field in the specified region
        phase_vector = np.angle(input_vals[0])
        phase_region = phase_vector * np.real(self.region)
        phase_components = _get_vector_components(phase_region)

        region_only_phase = [phase_components[i][np.nonzero(phase_components[i])] for i in range(3)]

        # Get the phase along the specified path (as in, from the source to the region) to get a
        # phase difference and eliminate the arbitrary +/- 2pi*n added to the phase
        phase_path_vector = np.angle(input_vals[0][np.nonzero(self.path)])
        # Unwrap phase along path and break into xyz components
        phase_path_components = _get_vector_components(phase_path_vector)
        phase_path = [np.unwrap(phase_path_components[i][2:]) for i in range(3)]

        # The last cell of the phase path is the center cell of region we are optimizing
        center_phase = [phase_path[i][-1] for i in range(3)]
        # We want the difference in phase from the source
        beginning_phase = [phase_path[i][0] for i in range(3)]

        phase_along_path = [center_phase[i] - beginning_phase[i] for i in range(3)]

        # Update the phases of the optimization region to reference the phase difference along the path
        mid = len(region_only_phase[0]) // 2
        path_phase_diff = [center_phase[i] - region_only_phase[i][mid] for i in range(3)]
        region_only_phase = [region_only_phase[i] + path_phase_diff[i] - beginning_phase[i] for i in range(3)]

        # 'Unwrap' the phase so that it is no longer periodic in [-pi, pi) but continuous
        # Unwrapping is done from the center cell outward in both directions
        for axis in region_only_phase:
            axis[mid:] = np.unwrap(axis[mid:])
            axis[:mid + 1] = np.flip(np.unwrap(np.flip(axis[:mid + 1])))

        # Update the center cell values to compare with on the next iteration
        self.center_phases = center_phase

        # In taking the average, weight the values by the field magnitude to avoid large changes in
        # phase due to small changes in the field for near-zero fields (similar to grad calculation)
        magnitude_region = np.abs(input_vals[0])[np.nonzero(self.region)]
        relative_magnitude = magnitude_region / np.average(magnitude_region)
        relative_magnitude_components = _get_vector_components(relative_magnitude)
        weight = [relative_magnitude_components[i] / len(magnitude_region) for i in range(3)]

        weighted_phase = [region_only_phase[i] * weight[i] for i in range(3)]

        # The weighted average is the sum of the weighted phases
        phase_avg = np.sum(weighted_phase)

        # Update cache
        self.current_phase_avg = phase_avg

        phase_avg_diff = phase_avg - phase_along_path[2]

        # print("{}: {}".format(self.wavelength, phase_avg))

        # Minus sign to correct for phase decreasing as wave propagates due to different sign convention
        return -phase_avg

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the 'gradient' of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the input to the phase function (for chain rule).

        Returns:
            gradient.
        """

        # Note that the arg function is not complex-differentiable anywhere, so we make a
        # pseudo-gradient for it that will still move the gradient descent algorithm in the
        # right direction.
        re = np.real(input_vals[0])
        im = np.imag(input_vals[0])

        grad_re = - im / (re**2 + im**2)
        grad_im = re / (re**2 + im**2)

        grad_global = (grad_re + 1j*grad_im) * np.abs(input_vals[0])
        region_sum = np.sum(self.region)

        region_only_grad = grad_global[np.nonzero(self.region)]

        grad = grad_global * self.region / region_sum

        return [grad_val * grad]

    def __str__(self):
        return "PhaseAbsolute({})".format(self._input)


@optplan.register_node(optplan.PhaseAbsolute)
def create_phase_absolute_function(params: optplan.ProblemGraphNode,
                                      work: workspace.Workspace):
    simspace = work.get_object(params.simulation.simulation_space)
    wlen = params.simulation.wavelength
    region = fdfd_tools.vec(work.get_object(params.region)(simspace, wlen))
    path = fdfd_tools.vec(work.get_object(params.path)(simspace, wlen))
    return PhaseAverageFunction(
        input_function=work.get_object(params.simulation), region=region, path=path, wavelength=wlen)


# TODO(logansu): Merge this into `AbsoluteValue`.
class DiffEpsilon(problem.OptimizationFunction):
    """Computes a L2 norm between two permittivity distributions.

    Specifically, this function computes `np.sum(np.abs(eps - eps_ref)**2)`.
    """

    def __init__(self, epsilon: problem.OptimizationFunction,
                 epsilon_ref: Callable[[], np.ndarray]) -> None:
        """Creates new `DiffEpsilon` function.

        Here we accept a callable because we may want to evaluate the target
        permittivity distribution dynamically (e.g. it may depend on the
        current value of a parametrization).

        Args:
            epsilon: Permittivity distribution that will be differentiated.
            epsilon_ref: Callable that returns a permittivity to which to
                compare `epsilon`.
        """
        super().__init__(epsilon)

        self._get_eps_ref = epsilon_ref

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Integrated sum of the difference between `epsilon` and
            `epsilon_ref`.
        """
        return np.sum(np.abs(input_vals[0] - self._get_eps_ref())**2)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        diff = np.conj(input_vals[0] - self._get_eps_ref())
        grad = 2 * diff
        return [grad_val * grad]


@optplan.register_node(optplan.DiffEpsilon)
def create_diff_epsilon(params: optplan.DiffEpsilon,
                        work: workspace.Workspace) -> DiffEpsilon:

    def epsilon_ref() -> np.ndarray:
        if params.epsilon_ref.type == "parametrization":
            space = work.get_object(params.epsilon_ref.simulation_space)(
                params.epsilon_ref.wavelength)
            structure = work.get_object(
                params.epsilon_ref.parametrization).get_structure()
            return (fdfd_tools.vec(space.eps_bg.grids) +
                    space.selection_matrix @ structure)
        else:
            raise NotImplementedError(
                "Epsilon spec with type {} not yet supported".format(
                    work.epsilon_ref.type))

    return DiffEpsilon(
        epsilon=work.get_object(params.epsilon), epsilon_ref=epsilon_ref)
