import os
import shutil

import wdm2
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _copyfiles(src_folder, dest_folder, filenames):
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_folder, filename),
            os.path.join(dest_folder, filename))


def test_wdm2(tmpdir):
    folder = os.path.join(tmpdir, 'wdm2_test_1')
    _copyfiles(CUR_DIR, folder, ["sim_fg.gds", "sim_bg.gds"])

    sim_space = wdm2.create_sim_space("sim_fg.gds", "sim_bg.gds")
    obj, monitors = wdm2.create_objective(sim_space)
    trans_list = wdm2.create_transformations(
        obj, monitors, sim_space, cont_iters=6, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)
    problem_graph.run_plan(plan, folder)


test_wdm2(CUR_DIR)
