# Import libraries necessary for monitor data processing.
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from spins.invdes.problem_graph import log_tools


def plot():
    # Define filenames.

    # `save_folder` is the full path to the directory containing the Pickle (.pkl) log files from the optimization.
    save_folder = os.path.join(os.getcwd(), 'GVD_test_wg')
    spec_folder = os.getcwd()

    # Load the logged monitor data and monitor spec information.

    # `df` is a pandas dataframe containing all the data loaded from the log Pickle (.pkl) files.
    df = log_tools.create_log_data_frame(log_tools.load_all_logs(save_folder))

    # `monitor_spec_filename` is the full path to the monitor spec yml file.
    monitor_spec_filename = os.path.join(spec_folder, "monitor_spec_dynamic.yml")

    # `monitor_descriptions` now contains the information from the monitor_spec.yml file. It follows the format of
    # the schema found in `log_tools.monitor_spec`.
    monitor_descriptions = log_tools.load_from_yml(monitor_spec_filename)

    # Plot all monitor data and save into a pdf file in the project folder.

    # `summary_out_name` is the full path to the pdf that will be generated containing plots of all the log data.
    summary_out_name = os.path.join(save_folder, "summary.pdf")

    # This command plots all the monitor data contained in the log files, saves it to the specified pdf file, and
    # displays to the screen.
    log_tools.plot_monitor_data(df, monitor_descriptions, None)


def main():
    plot()


if __name__ == "__main__":
    main()
