import os
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from spins import gds as gdslib
from spins import fdfd_tools
from spins import gridlock
from spins import material
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace

NM_PER_UM = 1000


class SimulationSpaceInstance(
        NamedTuple("SimulationSpaceInstance",
                   [("eps_bg", gridlock.Grid),
                    ("selection_matrix", np.ndarray)])):
    """Represents simulation space at a particular operating condition.

    Attributes:
        eps_bg: Permittivity distribution of the background.
        selection_matrix: The selection matrix.
    """


class SimulationSpace:

    def __init__(self, params: optplan.SimulationSpace, filepath: str):
        if params.mesh.type != "uniform":
            raise ValueError("Non-uniform meshing not yet supported.")

        # Setup the grid.
        self._dx = params.mesh.dx
        self._edge_coords = _create_edge_coords(params.sim_region, self._dx)
        self._ext_dir = gridlock.Direction.z  # Currently always extrude in z.
        # TODO(logansu): Factor out grid functionality and drawing.
        # Create a grid object just so we can calculate dxes.
        self._grid = gridlock.Grid(
            self._edge_coords, ext_dir=self._ext_dir, num_grids=3)

        self._pml_layers = params.pml_thickness

        self._eps_bg = params.eps_bg
        self._eps_fg = params.eps_fg
        self._selmat_type = params.selection_matrix_type

        self._filepath = filepath

        # Cache the simulation space instances since they are expensive to make.
        self._cache = {}

        # TODO(logansu): Remove this hack.
        # Call itself to set `self._design_area`.
        self.__call__(1500)

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dxes(self) -> fdfd_tools.GridSpacing:
        return [self._grid.dxyz, self._grid.autoshifted_dxyz()]

    @property
    def pml_layers(self) -> fdfd_tools.PmlLayers:
        return self._pml_layers

    @property
    def dims(self) -> Tuple[int, int, int]:
        return [
            len(self._edge_coords[0]) - 1,
            len(self._edge_coords[1]) - 1,
            len(self._edge_coords[2]) - 1
        ]

    @property
    def design_dims(self) -> Tuple[int, int]:
        return self._design_dims

    @property
    def edge_coords(self) -> fdfd_tools.GridSpacing:
        return self._edge_coords

    def __call__(self, wlen: float) -> SimulationSpaceInstance:
        """Creates the background permittivity and selection matrix at `wlen`.

        Since creating grids can be expensive, this function caches all the
        simulation space instances generated.

        Args:
            wlen: Wavelength to instantiate simulation space.

        Returns:
            Instantiated simulation space.
        """
        # Round the wavelength to avoid floating point discrepancies.
        wlen = _round(wlen, digits=6)

        if wlen not in self._cache:
            eps_bg = _create_grid(self._eps_bg, self._edge_coords, wlen,
                                  self._ext_dir, self._filepath)
            eps_fg = _create_grid(self._eps_fg, self._edge_coords, wlen,
                                  self._ext_dir, self._filepath)

            # Create the selection matrix.
            if self._selmat_type == optplan.SelectionMatrixType.DIRECT.value:
                # TODO(logansu): Create selection matrix from design dims.
                selection_mat, self._design_dims = (
                    gridlock.create_selection_matrix(
                        eps_bg.grids,
                        eps_fg.grids,
                        return_param_dims=True,
                    ))
            elif self._selmat_type == optplan.SelectionMatrixType.REDUCED.value:
                # TODO(logansu): Create selection matrix from design dims.
                selection_mat, self._design_dims = (
                    gridlock.create_selection_matrix(
                        eps_bg.grids,
                        eps_fg.grids,
                        reduced=True,
                        return_param_dims=True,
                    ))
            else:
                raise NotImplementedError(
                    "Selection matrix type {} not yet implemented".format(
                        self._selmat_type))

            self._cache[wlen] = SimulationSpaceInstance(
                eps_bg=eps_bg, selection_matrix=selection_mat)

        return self._cache[wlen]


@optplan.register_node(optplan.SimulationSpace)
def create_simulation_space(params: optplan.SimulationSpace,
                            work: workspace.Workspace) -> SimulationSpace:
    return SimulationSpace(params, work.filepath)


def _round(value: float, digits: int = 6) -> float:
    """Returns a rounded value with a requested number of significant digits.

    Args:
        value: Value that needs to be rounds.
        digits: Number of significant digits.

    Returns:
        Rounded number.
    """
    # TODO(vcruysse): Remove this and replace this with a check if a change in
    # wavelength results in a significant change in index.
    return round(value, -int(np.floor(np.log10(abs(value)))) + digits)


def _create_edge_coords(sim_region: optplan.Box3d,
                        dx: float) -> fdfd_tools.EdgeCoords:
    """Creates the edge coordinates of the grid for a uniform grid.

    Args:
        sim_region: The box defining the simulation region.
        dx: The grid spacing.

    Returns:
        Tuple where each element corresponds to one axis and contains an array
        that has the coordinates for the grid along that axis.
    """
    xyz_min = np.array(sim_region.center) - np.array(sim_region.extents) / 2
    xyz_max = np.array(sim_region.center) + np.array(sim_region.extents) / 2

    edge_coords = []
    for i in range(3):
        edge_coords.append(
            np.arange(xyz_min[i] - dx / 2, xyz_max[i] + dx / 2, dx))

    return edge_coords


def _create_grid(eps_spec: optplan.EpsilonSpec,
                 edge_coords: fdfd_tools.EdgeCoords, wlen: float,
                 ext_dir: gridlock.Direction, filepath: str) -> gridlock.Grid:
    if eps_spec.type != "gds":
        raise NotImplementedError(
            "Epsilon spec not implemented for type {}".format(eps_spec.type))

    # Make grid object.
    grid = gridlock.Grid(
        edge_coords,
        ext_dir=ext_dir,
        initial=_get_mat_index(eps_spec.mat_stack.background, wlen)**2,
        num_grids=3)

    # Draw layers.
    _draw_gds_on_grid(
        gds_stack=eps_spec.mat_stack.stack,
        grid=grid,
        gds_path=os.path.join(filepath, eps_spec.gds),
        wlen=wlen)

    # Make epsilon.
    grid.render()

    # Return epsilon and dxes.
    return grid


def _draw_gds_on_grid(gds_stack: List[optplan.GdsMaterialStackLayer],
                      grid: gridlock.Grid,
                      gds_path: str,
                      wlen: Optional[float] = None) -> None:
    """Draws onto a `Grid` based on a GDS file.

    Args:
        gds_stack: Stack element info on the layer, extent and refractive index.
        grid: Grid object to draw on.
        gds_path: Path of the gds.
        wlen: Wavelength required for index calculations.
    """

    # Load GDS.
    with open(gds_path, "rb") as gds_file:
        gds = gdslib.GDSImport(gds_file)

    # Draw layers
    for stack_element in gds_stack:
        layer = tuple(stack_element.gds_layer)
        polygons = gds.get_polygons(layer)
        extents = np.array(stack_element.extents)
        center = extents.mean()
        thickness = np.diff(extents)[0]
        polygon_center = np.zeros(3)
        polygon_center[grid.ext_dir] = center
        perm_bg = _get_mat_index(stack_element.background, wlen)**2
        perm_fg = _get_mat_index(stack_element.foreground, wlen)**2

        # Draw background
        grid.draw_slab(
            dir_slab=grid.ext_dir,
            center=center,
            thickness=thickness,
            eps=perm_bg)
        for polygon in polygons:
            polygon = np.around(polygon * NM_PER_UM)
            grid.draw_polygon(
                center=polygon_center,
                polygon=polygon,
                thickness=thickness,
                eps=perm_fg)


def _get_mat_index(index_element: optplan.Material,
                   wlen: Optional[float] = None) -> complex:
    """Return the refective index of an index element for a certain wavelength.

    Args:
        index_element: Material information.
        wlen: Wavelength to evaluate material index.

    Returns:
        Refractive index.

    Raises.
        ValueError: If index_element has no index or if the mat_name is invalid.
    """

    # TODO(logansu): Deal with naming.
    # Push the actual lookup for materials to the json prep size.
    if index_element.index is not None:
        index_real = index_element.index.get("real", default=0)
        index_imag = index_element.index.get("imag", default=0)
        index = index_real + 1j * index_imag
    elif index_element.mat_name is not None:
        name = index_element["mat_name"]
        if name in ["Air", "air"]:
            index = material.Air.refractive_index(np.array(wlen))[0]
        elif name in ["SiO2", "sio2", "sio"]:
            index = material.SiO2.refractive_index(np.array(wlen))[0]
        elif name in ["Si", "si"]:
            index = material.Si.refractive_index(np.array(wlen))[0]
        elif name in ["Si3N4", "sin", "si3n4", "SiN"]:
            index = material.Si3N4.refractive_index(np.array(wlen))[0]
        elif name.lower() in ["ta2o5", "tao", "tantalum pentoxide"]:
            index = material.Ta2O5.refractive_index(np.array(wlen))[0]
        else:
            raise ValueError("No valid material name.")
    else:
        raise ValueError("No valid material.")

    return index
