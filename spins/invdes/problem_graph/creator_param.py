from typing import List, Union

import numpy as np
from scipy.ndimage import filters

from spins import fdfd_tools
from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem_graph import optplan, grid_utils
from spins.invdes.problem_graph import workspace


@optplan.register_node(optplan.UniformInitializer)
class UniformDistribution:

    def __init__(self, params: optplan.UniformInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        return np.random.uniform(self._params.min_val, self._params.max_val,
                                 shape)


@optplan.register_node(optplan.NormalInitializer)
class NormalDistribution:

    def __init__(self, params: optplan.NormalInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        return np.random.normal(self._params.mean, self._params.std, shape)


@optplan.register_node(optplan.WaveguideInitializer)
class WaveguideDistribution:

    def __init__(self, params: optplan.WaveguideInitializer,
               work: workspace.Workspace) -> None:
        self._params = params
        self._work = work

    def __call__(self, shape: List[int]) -> np.ndarray:
        region_slice = grid_utils.create_region_slices(
            self._params.edge_coords, self._params.center,
            self._params.extents)
        region = np.zeros(shape)
        region[tuple(region_slice)] = 1

        outside_region = np.ones_like(self._region) - self._region
        lower_random = np.random.uniform(self._params.lower_min, self._params.lower_max, shape)
        upper_random = np.random.uniform(self._params.upper_min, self._params.upper_max, shape)
        return self._region * upper_random + outside_region * lower_random


class WaveguideDistribution2:
    def __init__(self, region: np.ndarray, lower_min: float, lower_max: float, upper_min: float, upper_max: float):
        self._region = region
        self._inverse_region = 1 - region
        self._lower_min = lower_min
        self._lower_max = lower_max
        self._upper_min = upper_min
        self._upper_max = upper_max

    def __call__(self, shape: List[int]) -> np.ndarray:
        lower_random = np.random.uniform(self._lower_min, self._lower_max, shape)
        upper_random = np.random.uniform(self._upper_min, self._upper_max, shape)
        return self._region * upper_random + self._inverse_region * lower_random


@optplan.register_node(optplan.WaveguideInitializer2)
def create_waveguide_distribution_2(params: optplan.WaveguideInitializer2,
                            work: workspace.Workspace):
    fake_wavelength = 1500
    sim_space = work.get_object(params.sim_space)
    region_3d = work.get_object(params.region)(sim_space, fake_wavelength)
    region = region_3d[0]
    return WaveguideDistribution2(region=region, lower_min=params.lower_min, lower_max=params.lower_max,
                                  upper_min=params.upper_min, upper_max=params.upper_max)


@optplan.register_node(optplan.WaveguideInitializer3)
class WaveguideDistribution3:
    def __init__(self, params: optplan.NormalInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        lower_random = np.random.uniform(self._params.lower_min, self._params.lower_max, shape)
        upper_random = np.random.uniform(self._params.upper_min, self._params.upper_max, shape)
        center_x = self._params.center_frac_x * shape[0]
        center_y = self._params.center_frac_y * shape[1]
        start_x = max(round(center_x - self._params.extent_frac_x * shape[0] / 2), 0)
        end_x = min(round(center_x + self._params.extent_frac_x * shape[0] / 2), shape[0])
        start_y = max(round(center_y - self._params.extent_frac_y * shape[1] / 2), 0)
        end_y = min(round(center_y + self._params.extent_frac_y * shape[1] / 2), shape[1])
        upper_slice = (slice(start_x, end_x), slice(start_y, end_y))
        distribution = lower_random
        distribution[upper_slice] = upper_random[upper_slice]
        return distribution


@optplan.register_node(optplan.GradientInitializer)
class GradientDistribution:
    def __init__(self, params: optplan.GradientInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        distribution = np.random.uniform(self._params.min, self._params.min + self._params.random, shape)
        center_x = self._params.center_frac_x * shape[0]
        center_y = self._params.center_frac_y * shape[1]
        start_x = max(round(center_x - self._params.extent_frac_x * shape[0] / 2), 0)
        end_x = min(round(center_x + self._params.extent_frac_x * shape[0] / 2), shape[0])
        start_y = max(round(center_y - self._params.extent_frac_y * shape[1] / 2), 0)
        end_y = min(round(center_y + self._params.extent_frac_y * shape[1] / 2), shape[1])
        num_steps = end_y - start_y
        step_size = (self._params.max - self._params.min - self._params.random) / num_steps

        for i in range(num_steps):
            step_slice = (slice(start_x, end_x), slice(end_y - i - 1, end_y - i))
            distribution[step_slice] += i * step_size

        upper_slice = (slice(start_x, end_x), slice(0, start_y))
        upper_dist = distribution[upper_slice] + step_size * num_steps
        distribution[upper_slice] = upper_dist

        return distribution


@optplan.register_node(optplan.PeriodicInitializer)
class PeriodicDistribution:
    def __init__(self, params: optplan.PeriodicInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        distribution = np.random.uniform(self._params.min, self._params.min + self._params.random, shape)
        add_max = self._params.max - self._params.min - self._params.random
        center_y = self._params.center_frac_y * shape[1]
        extent_y = self._params.extent_frac_y * shape[1]
        start_y = int(round(center_y - extent_y / 2))
        end_y = int(round(center_y + extent_y / 2))
        cell_size = self._params.sim_width / shape[0]
        period_num_cells = self._params.period / cell_size

        for x in range(shape[0]):
            period_frac_start = (x % period_num_cells) / period_num_cells
            period_frac_end = period_frac_start + 1 / period_num_cells
            if 0.5 <= period_frac_start < 1 and 0.5 <= period_frac_end < 1:
                distribution[x, start_y:end_y] += add_max
            elif 0 <= period_frac_start <= 0.5 <= period_frac_end <= 1:
                frac_high = (period_frac_end - 0.5) * period_num_cells
                distribution[x, start_y:end_y] += frac_high * add_max
            elif 0.5 <= period_frac_start <= 1 <= period_frac_end:
                frac_high = (1 - period_frac_start) * period_num_cells
                distribution[x, start_y:end_y] += frac_high * add_max

        distribution[:, 0:start_y] += add_max

        return distribution


@optplan.register_node(optplan.PixelParametrization)
def create_pixel_param(
        params: optplan.PixelParametrization,
        work: workspace.Workspace) -> parametrization.DirectParam:
    design_dims = work.get_object(params.simulation_space).design_dims
    init_val = work.get_object(params.init_method)(design_dims)
    return parametrization.DirectParam(init_val.flatten(order="F"))


@optplan.register_node(optplan.GratingParametrization)
def create_grating_param(
        params: optplan.GratingParametrization,
        work: workspace.Workspace) -> parametrization.GratingParam:
    # Only one of the design areas is nonzero. Figure out which one.
    design_dims = work.get_object(params.simulation_space).design_dims
    if design_dims[0] > 1 and design_dims[1] > 1:
        raise ValueError("Grating parametrization should have 1D design "
                         "area, got {}".format(design_dims))

    grating_len = np.max(design_dims)
    return parametrization.GratingParam([], num_pixels=grating_len)


@optplan.register_node(optplan.GratingFeatureConstraint)
def create_grating_feature_constraint(
        params: optplan.GratingFeatureConstraint,
        work: workspace.Workspace) -> problem.GratingFeatureConstraint:
    dx = work.get_object(params.simulation_space).dx
    return problem.GratingFeatureConstraint(params.min_feature_size / dx)


@optplan.register_node(optplan.CubicParametrization)
@optplan.register_node(optplan.HermiteLevelSetParametrization)
def create_cubic_or_hermite_levelset(
        params: Union[optplan.CubicParametrization, optplan.
                      HermiteLevelSetParametrization], work: workspace.Workspace
) -> parametrization.CubicParam:
    design_dims = work.get_object(params.simulation_space).design_dims

    # Calculate periodicity of the parametrization.
    periods = params.periods
    if periods is None:
        periods = np.array([0, 0])
    periodicity = [p != 0 for p in periods]

    # Calculate reflection symmetry.
    reflection_symmetry = params.reflection_symmetry
    if reflection_symmetry is None:
        reflection_symmetry = np.array([0, 0])

    # Make fine grid.
    undersample = params.undersample
    fine_x = np.arange(-design_dims[0] / 2, design_dims[0] / 2)
    fine_y = np.arange(-design_dims[1] / 2, design_dims[1] / 2)
    # Center the grid.
    fine_x -= (fine_x[-1] + fine_x[0]) / 2
    fine_y -= (fine_y[-1] + fine_y[0]) / 2

    # Create the coarse grid.
    if periodicity[0]:
        n_x = np.round((fine_x[-1] - fine_x[0]) / undersample) + 1
        coarse_x = np.linspace(fine_x[0], fine_x[-1] + 1, n_x)
    else:
        coarse_x = np.arange(-design_dims[0] / 2 - undersample,
                             design_dims[0] / 2 + undersample, undersample)
    coarse_x -= (coarse_x[-1] +
                 coarse_x[0]) / 2  # this is necessary to have correct symmetry

    if periodicity[1]:
        n_y = np.round((fine_y[-1] - fine_y[0]) / undersample) + 1
        coarse_y = np.linspace(fine_y[0], fine_y[-1] + 1, n_y)
    else:
        coarse_y = np.arange(-design_dims[1] / 2 - undersample,
                             design_dims[1] / 2 + undersample, undersample)
    coarse_y -= (coarse_y[-1] +
                 coarse_y[0]) / 2  # this is necessary to have correct symmetry

    init_val = work.get_object(
        params.init_method)([len(coarse_x), len(coarse_y)])
    init_val = filters.gaussian_filter(init_val, 1).flatten(order='F')

    # Make parametrization.
    if params.type == "parametrization.hermite_levelset":
        from spins.invdes.parametrization import levelset_parametrization
        param_class = levelset_parametrization.HermiteLevelSet
    elif params.type == "parametrization.cubic":
        param_class = parametrization.CubicParam
    else:
        raise ValueError("Unexpected parametrization type, got {}".format(
            params.type))

    return param_class(
        initial_value=init_val,
        coarse_x=coarse_x,
        coarse_y=coarse_y,
        fine_x=fine_x,
        fine_y=fine_y,
        symmetry=reflection_symmetry,
        periodicity=periodicity,
        periods=periods)


class FabricationPenalty(problem.OptimizationFunction):
    """
    Fabrication Penalty objective
    This optimization function evaluates the fabrication penalty of the parametrization
    for a certain fabrication size limit.
    """

    def __init__(self,
                 fcon_gap: float,
                 fcon_curv: float,
                 fabcon_method: int = 2,
                 apply_factors: bool = True):
        '''
        Arg:
            fcon_gap: the smallestallowed gap size.
            fcon_curv: the smallest allowed curvarure diameter.
            fabcon_method:
                0: only applies the gap constraint,
                1: applies the gap and curvature constraint by evaluating the curvature
                    constraint on the border (only available with BicubicLevelSet)
                2: applies the gap and curvature constraint (curvature is evaluated
                    everywhere) (only available with HermiteLevelSet)
            apply_factors: boolean that indiates whether or not you scale up the fcon
                values.
        '''

        self.d_gap = np.pi / fcon_gap
        self.d_curv = 2 / fcon_curv
        self.method = fabcon_method

        self.d_gap_factor = 1
        self.d_curv_factor = 1
        if apply_factors:
            self.d_gap_factor = 1.2**-1
            self.d_curv_factor = 1.1**-1

    def calculate_objective_function(self, param) -> np.ndarray:
        if self.method == 0:
            penalty = param.calculate_gap_penalty(
                self.d_gap_factor * self.d_gap)
        elif self.method == 1:
            penalty = param.calculate_curv_penalty(
                self.d_curv_factor * self.d_curv)
        elif self.method == 2:
            curv = param.calculate_curv_penalty(
                self.d_curv_factor * self.d_curv)
            gap = param.calculate_gap_penalty(self.d_gap_factor * self.d_gap)
            penalty = curv + gap
        else:
            raise ValueError("Fabcon method is invalid.")
        return penalty

    def calculate_gradient(self, param) -> List[np.ndarray]:
        if self.method == 0:
            gradient = param.calculate_gap_penalty_gradient(
                self.d_gap_factor * self.d_gap)
        elif self.method == 1:
            gradient = param.calculate_curv_penalty_gradient(
                self.d_curv_factor * self.d_curv)
        elif self.method == 2:
            curv = param.calculate_curv_penalty_gradient(
                self.d_curv_factor * self.d_curv)
            gap = param.calculate_gap_penalty_gradient(
                self.d_gap_factor * self.d_gap)
            gradient = curv + gap
        else:
            raise ValueError("Fabcon method is invalid.")
        return gradient

    def __str__(self):
        return 'FabCon(' + str(self.d_gap) + ')'


@optplan.register_node(optplan.FabricationConstraint)
def create_fabrication_constraint(
        params: optplan.FabricationConstraint,
        work: workspace.Workspace) -> FabricationPenalty:
    dx = work.get_object(params.simulation_space).dx
    minimum_curvature_diameter = params.minimum_curvature_diameter / (dx / 2)
    minimum_gap = params.minimum_gap / (dx / 2)
    methods = {"gap": 0, "curv": 1, "gap_and_curve": 2}

    return FabricationPenalty(
        fcon_gap=minimum_gap,
        fcon_curv=minimum_curvature_diameter,
        fabcon_method=methods[params.method],
        apply_factors=params.apply_factors)
