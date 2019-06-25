import os
import shutil

import phase
from monitor_plot import plot
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _copyfiles(src_folder, dest_folder, filenames):
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_folder, filename),
            os.path.join(dest_folder, filename))


def test_phase(tmpdir):
    folder = os.path.join(tmpdir, 'phase_test_results')
    _copyfiles(CUR_DIR, folder, ["sim_fg_kerr.gds", "sim_bg_kerr.gds"])

    sim_space = phase.create_sim_space("sim_fg_kerr.gds", "sim_bg_kerr.gds")
    obj, monitors = phase.create_objective(sim_space)
    trans_list = phase.create_transformations(
        obj, monitors, sim_space, cont_iters=40, min_feature=50, num_stages=4)
    plan = optplan.OptimizationPlan(transformations=trans_list)
    problem_graph.run_plan(plan, folder)


test_phase(CUR_DIR)
plot()
