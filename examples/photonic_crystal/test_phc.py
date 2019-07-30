import os
import shutil

import phc
from monitor_plot import plot
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _copyfiles(src_folder, dest_folder, filenames):
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_folder, filename),
            os.path.join(dest_folder, filename))


def test_phc(tmpdir):
    folder = os.path.join(tmpdir, 'phc_ideal')
    fg = "sim_fg_phc.gds"
    bg = "sim_bg_phc.gds"
    _copyfiles(CUR_DIR, folder, [fg, bg])

    sim_space = phc.create_sim_space(fg, bg)
    obj, monitors = phc.create_objective(sim_space)
    trans_list = phc.create_transformations(
        obj, monitors, sim_space, cont_iters=40, min_feature=60, num_stages=5)
    plan = optplan.OptimizationPlan(transformations=trans_list)
    problem_graph.run_plan(plan, folder)


test_phc(CUR_DIR)
plot()
