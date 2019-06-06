import os
import shutil

import kerr
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _copyfiles(src_folder, dest_folder, filenames):
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_folder, filename),
            os.path.join(dest_folder, filename))


def test_kerr(tmpdir):
    folder = os.path.join(tmpdir, 'kerr_test_results')
    _copyfiles(CUR_DIR, folder, ["sim_fg.gds", "sim_bg.gds"])

    sim_space = kerr.create_sim_space("sim_fg.gds", "sim_bg.gds")
    obj, monitors = kerr.create_objective(sim_space)
    trans_list = kerr.create_transformations(
        obj, monitors, sim_space, cont_iters=8, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)
    problem_graph.run_plan(plan, folder)


test_kerr(CUR_DIR)
