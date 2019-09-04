#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la

import csv
import os


# Whether to generate a PDF file. If False, will generate pgf.
GENERATE_PDF = True


def switch_matplotlib_to_agg():
    import matplotlib
    matplotlib.use("pgf")


switch_matplotlib_to_agg()
import matplotlib.pyplot as plt  # noqa


FONTSIZE = 10


def initialize_matplotlib():
    plt.rc("font", family="serif")
    plt.rc("text", usetex=False)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("axes", labelsize=0.01)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("pgf", rcfonts=True)
    # Needed on porter, which does not have xelatex
    plt.rc("pgf", texsystem="pdflatex")


initialize_matplotlib()


DATA_DIR = "raw-data"
OUTPUT_DIR = "out"


def open_data_file(filename, **kwargs):
    return open(os.path.join(DATA_DIR, filename), "r", **kwargs)


def open_output_file(filename, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **kwargs)


# {{{ utils

def print_table(table, headers, outf_name, column_formats=None):
    if column_formats is None:
        column_formats = "c" * len(headers)
    with open_output_file(outf_name) as outfile:
        def my_print(s):
            print(s, file=outfile)
        my_print(r"\begin{tabular}{%s}" % column_formats)
        my_print(r"\toprule")
        my_print(" & ".join(headers) + r"\\")
        my_print(r"\midrule")
        for row in table:
            my_print(" & ".join(row) + r"\\")
        my_print(r"\bottomrule")
        my_print(r"\end{tabular}")
    print(f"Wrote output {outf_name}")


def fmt(val):
    """Format a numerical table cell."""
    if isinstance(val, str):
        return val
    return f"{val:e}"

# }}}


# {{{ green error table

def generate_green_error_table(infile, scheme_name):
    def converged_fmt(item, converged):
        if not converged:
            return item
        return r"\converged{%s}" % item

    def is_converged(err, ref):
        # Converged if err is within 1% of ref or lower.
        return err <= ref * 1.01

    results_l2 = {}
    results_linf = {}

    fmm_orders = set()
    qbx_orders = set()

    reader = csv.DictReader(infile)

    for row in reader:
        fmm_order = row["fmm_order"]
        if fmm_order != "inf":
            fmm_order = int(fmm_order)

        fmm_orders.add(fmm_order)

        qbx_order = int(row["qbx_order"])
        qbx_orders.add(qbx_order)

        results_l2[fmm_order, qbx_order] = float(row["err_l2"])
        results_linf[fmm_order, qbx_order] = float(row["err_linf"])

    fmm_orders = sorted(fmm_orders, key=lambda order: -1 if order == "inf" else order)
    qbx_orders = sorted(qbx_orders)

    headers = (
            [r"{$(1/2)^{\pfmm+1}$}", r"{$\pfmm$}"]
            + [r"{$\pqbx=%d$}" % p for p in qbx_orders])

    column_formats = "".join([
            "S[table-format = 1e-1, round-precision = 0]",
            "c"] + ["S"] * len(qbx_orders))

    converged_values_l2 = []
    converged_values_linf = []

    table_l2 = []
    table_linf = []

    for fmm_order in fmm_orders:
        errs_l2 = []
        errs_linf = []

        converged_l2 = []
        converged_linf = []

        if fmm_order is False:
            fmm_error = "{0}"
        else:
            fmm_error = 2 ** -(fmm_order + 1)

        for iqbx_order, qbx_order in enumerate(qbx_orders):
            err_l2 = results_l2[fmm_order, qbx_order]
            err_linf = results_linf[fmm_order, qbx_order]

            if (
                    len(converged_values_l2) > iqbx_order
                    and is_converged(err_l2, converged_values_l2[iqbx_order])):
                converged_l2.append(True)
            else:
                converged_l2.append(False)

            if fmm_order is False:
                converged_values_l2.append(err_l2)

            errs_linf.append(err_linf)

            if (
                    len(converged_values_linf) > iqbx_order
                    and is_converged(err_linf, converged_values_linf[iqbx_order])):
                converged_linf.append(True)
            else:
                converged_linf.append(False)

            if fmm_order is False:
                converged_values_linf.append(err_linf)

        row_l2 = [fmt(fmm_error)]
        row_l2.append(str(fmm_order) if fmm_order != "inf" else "(direct)")
        for e, c in zip(errs_l2, converged_l2):
            row_l2.append(converged_fmt(fmt(e), c))
        table_l2.append(row_l2)

        row_linf = [fmt(fmm_error)]
        row_linf.append(
                str(fmm_order) if fmm_order != "inf" else "(direct)")
        for e, c in zip(errs_linf, converged_linf):
            row_linf.append(converged_fmt(fmt(e), c))
        table_linf.append(row_linf)

    print_table(table_l2, headers, f"green-error-l2-{scheme_name}.tex",
                column_formats)
    print_table(table_linf, headers, f"green-error-linf-{scheme_name}.tex",
                column_formats)

# }}}


def main():
    with open_data_file("green-error-results-gigaqbx-65.csv", "r", newline="") as infile:
        generate_green_error_table(infile, scheme_name="gigaqbx")

    with open_data_file("green-error-results-qbxfmm-65.csv", "r", newline="") as infile:
        generate_green_error_table(infile, scheme_name="qbxfmm")

    # run_bvp_error_experiment(use_gigaqbx_fmm=True)
    # run_particle_distributions_experiment()
    # run(run_complexity_experiment, use_gigaqbx_fmm=True, compute_wall_times=False)
    # run(run_complexity_experiment, use_gigaqbx_fmm=False)
    # run(run_from_sep_smaller_threshold_complexity_experiment)

    # run(run_wall_time_experiment, use_gigaqbx_fmm=True)
    # run(run_wall_time_experiment, use_gigaqbx_fmm=False)
    # run(run_level_restriction_experiment)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
