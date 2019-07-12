"""Defines schema for electromagnetic-related nodes."""
import enum

from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils

BOUNDARY_CONDITION_TYPES = []
MESH_TYPES = []


class Material(schema_utils.Model):
    """Defines a material.

    A material can be defined either by a name (e.g. "silicon") or by refractive
    refractive index.

    Attributes:
        mat_name: Name of a material. This needs to be a material defined in
            `spins.material`.
        index: Refractive index value.
    """
    mat_name = types.StringType()
    index = types.PolyModelType(optplan.ComplexNumber)


class GdsMaterialStackLayer(schema_utils.Model):
    """Defines a single layer in a material stack.

    Attributes:
        foreground: Material to fill any structure in the layer.
        background: Material to fill any non-structure areas in the layer.
        extents: Start and end coordiantes of the layer stack.
        gds_layer: Name of GDS layer that contains the polygons for this layer.
    """
    foreground = types.ModelType(Material)
    background = types.ModelType(Material)
    extents = optplan.vec2d()
    gds_layer = types.ListType(types.IntType())


class GdsMaterialStack(schema_utils.Model):
    """Defines a material stack.

    This is used by `GdsEps` to define the permittivity distribution.

    Attributes:
        background: Material to fill any regions that are not covered by
            a material stack layer.
        stack: A list of `MaterialStackLayer` that defines permittivity for
            each layer.
    """
    background = types.ModelType(Material)
    stack = types.ListType(types.ModelType(GdsMaterialStackLayer))


class EpsilonSpec(schema_utils.Model):
    """Describes a specification for permittivity distribution."""


@schema_utils.polymorphic_model()
class GdsEps(EpsilonSpec):
    """Defines a permittivity distribution using a GDS file.

    The GDS file will be flattened so that each layer only contains polygons.
    TODO(logansu): Expand description.

    Attributes:
        type: Must be "gds_epsilon".
        gds: URI of GDS file.
        mat_stack: Description of each GDS layer permittivity values and
            thicknesses.
        stack_normal: Direction considered the normal to the stack.
    """
    type = schema_utils.polymorphic_model_type("gds")
    gds = types.StringType()
    mat_stack = types.ModelType(GdsMaterialStack)
    stack_normal = optplan.vec3d()


@schema_utils.polymorphic_model()
class ParamEps(EpsilonSpec):
    """Defines a permittivity distribution based on a parametriation.

    Attributes:
        type: Must be "parametrization".
        parametrization: Name of the parametrization.
        simulation_space: Name of the simulation space.
        wavelength: Wavelength.
    """
    type = schema_utils.polymorphic_model_type("parametrization")
    parametrization = optplan.ReferenceType(optplan.Parametrization)
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    wavelength = types.FloatType()


@schema_utils.polymorphic_model(MESH_TYPES)
class UniformMesh(schema_utils.Model):
    """Defines a uniform mesh.

    Attributes:
        type: Must be "uniform".
        dx: Unit cell distance for EM grid (nm).
    """
    type = schema_utils.polymorphic_model_type("uniform")
    dx = types.FloatType()


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class BlochBoundary(schema_utils.Model):
    """Represents a Bloch boundary condition.

    Attributes:
        bloch_vector: 3D Bloch optplan.vector.
    """
    type = schema_utils.polymorphic_model_type("bloch")
    bloch_vector = optplan.vec3d(default=[0, 0, 0])


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class PecBoundary(schema_utils.Model):
    """Represents PEC boundary."""
    type = schema_utils.polymorphic_model_type("pec")


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class PmcBoundary(schema_utils.Model):
    """Represents PMC boundary."""
    type = schema_utils.polymorphic_model_type("pmc")


class SelectionMatrixType(enum.Enum):
    """Defines possible types for selection matrices."""
    # Direct lattice selection matrix where we select out all points in the
    # Yee grid.
    DIRECT = "direct_lattice"
    # Design dimensions is reduced by factor of 4 by parametrizing only the "z"
    # component.
    REDUCED = "uniform"


@optplan.register_node_type()
class SimulationSpace(optplan.SimulationSpaceBase):
    """Defines a simulation space.

    A simulation space contains information regarding the permittivity
    distributions but not the fields, i.e. no information regarding sources
    and wavelengths.

    Attributes:
        name: Name to identify the simulation space. Must be unique.
        eps_fg: Foreground permittivity.
        eps_bg: Background permittivity.
        mesh: Meshing information. This describes how the simulation region
            should be meshed.
        sim_region: Rectangular prism simulation domain.
        selection_matrix_type: The type of selection matrix to form. This
            is subject to change.
    """
    type = schema_utils.polymorphic_model_type("simulation_space")
    eps_fg = types.PolyModelType(EpsilonSpec)
    eps_bg = types.PolyModelType(EpsilonSpec)
    mesh = types.PolyModelType(MESH_TYPES)
    sim_region = types.ModelType(optplan.Box3d)
    boundary_conditions = types.ListType(
        types.PolyModelType(BOUNDARY_CONDITION_TYPES), min_size=6, max_size=6)
    pml_thickness = types.ListType(types.IntType(), min_size=6, max_size=6)
    selection_matrix_type = types.StringType(
        default=SelectionMatrixType.DIRECT.value,
        choices=tuple(select_type.value for select_type in SelectionMatrixType),
    )


@optplan.register_node_type()
class WaveguideMode(optplan.ProblemGraphNode):
    """Represents basic information for a waveguide mode.

    This class is not intended to be instantiable.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("em.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeSource(optplan.EmSource):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("source.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeOverlap(optplan.EmOverlap):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("overlap.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class KerrOverlap(optplan.EmOverlap):
    """Represents the area in which we want to have a strong Kerr nonlinearity

    Effectively just the epsilon value in the desired region.

    Attributes:
        center: Optimization area center
        extents: Width and height of the optimization region
        power: The transmission power of the mode.
    """

    type = schema_utils.polymorphic_model_type("overlap.kerr")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    power = types.FloatType()


@optplan.register_node_type()
class Region(optplan.EmRegion):
    """Represents the area in which we want to consider the objective function

    Effectively just the epsilon value in the desired region.

    Attributes:
        center: Optimization area center
        extents: Width and height of the optimization region
        power: Multiplier for the effects of the objective function (default 1).
    """

    type = schema_utils.polymorphic_model_type("overlap.region")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    power = types.FloatType()


@optplan.register_node_type()
class PlaneWaveSource(optplan.EmSource):
    """Represents a plane wave source.

    Attributes:
        type: Must be "source.plane_wave".
    """
    type = schema_utils.polymorphic_model_type("source.plane_wave")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    theta = types.FloatType()
    psi = types.FloatType()
    polarization_angle = types.FloatType()
    overwrite_bloch_vector = types.BooleanType()
    border = types.ListType(types.FloatType())
    power = types.FloatType()


@optplan.register_node_type()
class GaussianSource(optplan.EmSource):
    """Represents a gaussian source.

    Attributes:
        type: Must be "source.gaussian_beam".
        normalize_by_sim: If `True`, normalize the power by running a
            simulation.
    """
    type = schema_utils.polymorphic_model_type("source.gaussian_beam")
    w0 = types.FloatType()
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    theta = types.FloatType()
    psi = types.FloatType()
    polarization_angle = types.FloatType()
    overwrite_bloch_vector = types.BooleanType()
    power = types.FloatType()
    normalize_by_sim = types.BooleanType(default=False)


@optplan.register_node_type()
class WaveguideModeEigSource(optplan.EmSource):
    """Represents a photonic crystal waveguide mode.

    The waveguide does NOT have to be axis-aligned. The waveguide mode is
    computed as a 3D eigenmode solve.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("source.waveguide_mode_eig")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeEigOverlap(optplan.EmOverlap):
    """Represents a photonic crystal waveguide mode.

    The waveguide does NOT have to be axis-aligned. The waveguide mode is
    computed as a 3D eigenmode solve.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("overlap.waveguide_mode_eig")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class FdfdSimulation(optplan.Function):
    """Defines a FDFD simulation.

    Attributes:
        type: Must be "function.fdfd_simulation".
        name: Name of simulation.
        simulation_space: Simulation space name.
        source: Source name.
        wavelength: Wavelength at which to simulate.
        solver: Name of solver to use.
        bloch_vector: bloch optplan.vector at which to simulate.
    """
    type = schema_utils.polymorphic_model_type("function.fdfd_simulation")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    epsilon = optplan.ReferenceType(optplan.Function)
    source = optplan.ReferenceType(optplan.EmSource)
    wavelength = types.FloatType()
    solver = types.StringType(
        choices=("maxwell_bicgstab", "maxwell_cg", "local_direct"))
    bloch_vector = types.ListType(types.FloatType())


@optplan.register_node_type()
class Epsilon(optplan.Function):
    """Defines a Epsilon Grid.

    Attributes:
        type: Must be "function.epsilon".
        name: Name of epsilon.
        simulation_space: Simulation space name.
        wavelength: Wavelength at which to calculate epsilon.
    """
    type = schema_utils.polymorphic_model_type("function.epsilon")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    wavelength = types.FloatType()
    structure = optplan.ReferenceType(optplan.Parametrization)


@optplan.register_node_type()
class Overlap(optplan.Function):
    """Defines an overlap integral.

    Attributes:
        type: Must be "function.overlap".
        simulation: Simulation from which electric fields are obtained.
        overlap: Overlap type to use.
    """
    type = schema_utils.polymorphic_model_type("function.overlap")
    simulation = optplan.ReferenceType(optplan.Function)
    overlap = optplan.ReferenceType(optplan.EmOverlap)


@optplan.register_node_type()
class OverlapIntensity(optplan.Function):
    """Defines an overlap integral of the intensity of the field |E|^2.

    Attributes:
        type: Must be "function.overlap_intensity".
        simulation: Simulation from which electric fields are obtained.
        overlap: Overlap type to use.
    """
    type = schema_utils.polymorphic_model_type("function.overlap_intensity")
    simulation = optplan.ReferenceType(optplan.Function)
    overlap = optplan.ReferenceType(optplan.EmOverlap)


@optplan.register_node_type()
class WaveguidePhase(optplan.Function):
    """Defines a function that returns the phase difference between a waveguide mode
    and the source.

    Attributes:
        type: Must be "function.phase_absolute".
        simulation: Simulation from which electric fields are obtained.
        overlap: Overlap type to use.
        path: path from the source to the measurement waveguide mode region
    """
    type = schema_utils.polymorphic_model_type("function.waveguide_phase")
    simulation = optplan.ReferenceType(optplan.Function)
    overlap_in = optplan.ReferenceType(optplan.EmOverlap)
    overlap_out = optplan.ReferenceType(optplan.EmOverlap)
    path = optplan.ReferenceType(optplan.EmRegion)


@optplan.register_node_type()
class PhaseAbsolute(optplan.Function):
    """Defines a function that returns the absolute phase of the field.

    Attributes:
        type: Must be "function.phase_absolute".
        simulation: Simulation from which electric fields are obtained.
    """
    type = schema_utils.polymorphic_model_type("function.phase_absolute")
    simulation = optplan.ReferenceType(optplan.Function)
    region = optplan.ReferenceType(optplan.EmRegion)
    path = optplan.ReferenceType(optplan.EmRegion)


@optplan.register_node_type()
class DiffEpsilon(optplan.Function):
    """Defines a function that finds the L1 norm between two permittivities.

    Specifially, the function is defined as `sum(|epsilon - epsilon_ref|)`.

    Attributes:
        type: Must be "function.diff_epsilon".
        epsilon: Permittivity.
        epsilon_ref: Base permittivity to compare to.
    """
    type = schema_utils.polymorphic_model_type("function.diff_epsilon")
    epsilon = optplan.ReferenceType(optplan.Function)
    epsilon_ref = types.PolyModelType(EpsilonSpec)
