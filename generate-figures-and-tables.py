#!/usr/bin/env python3
import numpy as np

import csv
import os


# Whether to generate a PDF file. If False, will generate pgf.
GENERATE_PDF = True


def switch_matplotlib_to_agg():
    import matplotlib
    matplotlib.use("pgf")


switch_matplotlib_to_agg()
import matplotlib.pyplot as plt  # noqa


SMALLFONTSIZE = 8
FONTSIZE = 10
LINEWIDTH = 0.5


def initialize_matplotlib():
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    # https://stackoverflow.com/questions/40424249/vertical-alignment-of-matplotlib-legend-labels-with-latex-math
    plt.rc(("text.latex",), preview=True)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("axes", labelsize=1)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", linewidth=LINEWIDTH)
    plt.rc("pgf", rcfonts=True)
    plt.rc("lines", linewidth=LINEWIDTH)
    plt.rc("patch", linewidth=LINEWIDTH)
    plt.rc("legend", fancybox=False)
    plt.rc("legend", framealpha=1)
    plt.rc("legend", frameon=False)
    plt.rc("savefig", dpi=150)


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


def converged_fmt(item, converged):
    if not converged:
        return item
    return r"\converged{%s}" % item


def is_converged(err, ref):
    # Converged if err is within 1% of ref or lower.
    return err <= ref * 1.01

# }}}


# {{{ green error table

def generate_green_error_table(infile, scheme_name):
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


# {{{ bvp table

def generate_bvp_error_table(infile_bvp, infile_green):
    # {{{ read inputs

    bvp_error_l2 = {}
    bvp_error_linf = {}
    bvp_gmres_nit = {}

    fmm_orders = set()
    qbx_orders = set()

    for row in csv.DictReader(infile_bvp):
        fmm_order = int(row["fmm_order"])
        fmm_orders.add(fmm_order)

        qbx_order = int(row["qbx_order"])
        qbx_orders.add(qbx_order)

        bvp_error_l2[fmm_order, qbx_order] = float(row["err_l2"])
        bvp_error_linf[fmm_order, qbx_order] = float(row["err_linf"])
        bvp_gmres_nit[fmm_order, qbx_order] = row["gmres_nit"]

    fmm_orders = sorted(fmm_orders)
    qbx_orders = sorted(qbx_orders)

    green_error_l2 = {}
    green_error_linf = {}

    for row in csv.DictReader(infile_green):
        fmm_order = int(row["fmm_order"])
        qbx_order = int(row["qbx_order"])
        green_error_l2[fmm_order, qbx_order] = float(row["err_l2"])
        green_error_linf[fmm_order, qbx_order] = float(row["err_linf"])

    del fmm_order
    del qbx_order

    # }}}

    # {{{ generate convergence data

    def estimate_convergence(results_table):
        converged = {}

        for p, next_p in zip(fmm_orders, fmm_orders[1:]):
            for q in qbx_orders:
                converged[p, q] = False
                converged[next_p, q] = False

                if is_converged(results_table[p, q], results_table[next_p, q]):
                    converged[p, q] = True
                    converged[next_p, q] = True

        return converged

    converged_bvp_error_l2 = estimate_convergence(bvp_error_l2)
    converged_bvp_error_linf = estimate_convergence(bvp_error_linf)

    converged_green_error_l2 = estimate_convergence(green_error_l2)
    converged_green_error_linf = estimate_convergence(green_error_linf)

    # }}}

    from pytools import flatten

    headers = list(flatten(
        [[r"{$(1/2)^{\pfmm+1}$}", r"{$\pfmm$}"]]
        + [[r"{$\pqbx=%d$}" % p, r"\#it"] for p in qbx_orders]))

    column_formats = "".join([
        "S[table-format = 1e-1, round-precision = 0]",
        "c"] + ["S", "c"] * len(qbx_orders))

    def build_table(bvp_errors, converged_bvp_errors, green_errors, converged_green_errors):
        table = []

        for p in fmm_orders:
            fmm_error = ""
            if p != fmm_orders[0]:
                fmm_error += r"\cmidrule{1-%d} " % (2 * len(qbx_orders) + 2)
            fmm_error += fmt(2 ** - (p + 1))
            row = [fmm_error]
            row.append(str(p))
            for q in qbx_orders:
                err = fmt(bvp_errors[p, q])
                row.append(converged_fmt(err, converged_bvp_errors[p, q]))
                nit = str(bvp_gmres_nit[p, q])
                row.append(nit)
            table.append(row)

            row = ["", ""]
            for q in qbx_orders:
                err = fmt(green_errors[p, q])
                row.append(converged_fmt(err, converged_green_errors[p, q]))
                row.append("")
            table.append(row)

        return table

    table_l2 = build_table(bvp_error_l2, converged_bvp_error_l2, green_error_l2,
                           converged_green_error_l2)

    table_linf = build_table(bvp_error_linf, converged_bvp_error_linf,
                             green_error_linf, converged_green_error_linf)

    print_table(table_l2, headers, "bvp-l2.tex", column_formats)
    print_table(table_linf, headers, "bvp-linf.tex", column_formats)

# }}}


# {{{ particle distribution table

PERCENTILES = (20, 40, 60, 80, 100)


def generate_particle_distribution_table(infile):
    rows = []

    headers = [
            r"\multirow{2}{*}{$n$}",
            r"\cellcenter{\multirow{2}{*}{$N_S$}}",
            r"\cellcenter{\multirow{2}{*}{$M_C$}}",
            r"\multicolumn{5}{c}{Percentiles}"]

    for entry in csv.DictReader(infile):
        row = [
                str(entry["n_arms"]),
                str(entry["nsources"]),
                "%.1f" % float(entry["avg"]),
                ]
        row.extend(
                "%.1f" % float(entry["percentile_%d" % pct]) for pct in PERCENTILES)
        rows.append(row)

    rows.sort(key=lambda row: int(row[0]))
    rows.insert(
            0,
            [
                r"\cmidrule{4-8}",
                "",
                "",
            ] + [
                r"\cellcenter{%d\%%}" % pct for pct in PERCENTILES])

    print_table(rows, headers, "particle-distributions.tex", "r" * len(rows[0]))

# }}}


# {{{ complexity results

_colors = plt.cm.Paired.colors  # pylint:disable=no-member


class Colors(object):
    LIGHT_BLUE  = _colors[0]
    BLUE        = _colors[1]
    LIGHT_GREEN = _colors[2]
    GREEN       = _colors[3]
    RED         = _colors[5]
    ORANGE      = _colors[7]
    PURPLE      = _colors[9]


class QBXPerfLabelingBase(object):

    silent_summed_features = (
            "form_multipoles",
            "coarsen_multipoles",
            "eval_direct",
            "eval_multipoles",
            "refine_locals",
            "eval_locals",
            "translate_box_local_to_qbx_local",
            "eval_qbx_expansions",
            )
    
    perf_line_styles = ("x-", "+-", ".-", "*-", "s-", "d-")

    summary_line_style = ".--"

    summary_label = "all"

    summary_label = "all"

    summary_color = Colors.RED


class GigaQBXPerfLabeling(QBXPerfLabelingBase):

    perf_features = (
            "form_global_qbx_locals_list1",
            "multipole_to_local",
            "form_global_qbx_locals_list3",
            "translate_box_multipoles_to_qbx_local",
            "form_global_qbx_locals_list4",
            "form_locals")

    perf_labels = (
            r"$U_b$",
            r"$V_b$",
            r"$W_b^\mathrm{close}$",
            r"$W_b^\mathrm{far}$",
            r"$X_b^\mathrm{close}$",
            r"$X_b^\mathrm{far}$")

    perf_colors = (
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.LIGHT_BLUE,
            Colors.BLUE,
            Colors.LIGHT_GREEN,
            Colors.GREEN)


class QBXFMMPerfLabeling(QBXPerfLabelingBase):

    perf_features = (
            "form_global_qbx_locals_list1",
            "multipole_to_local",
            "translate_box_multipoles_to_qbx_local",
            "form_locals")

    perf_labels = (
            r"$U_b$",
            r"$V_b$",
            r"$W_b$",
            r"$X_b$")

    perf_colors = (
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.BLUE,
            Colors.GREEN)
    

def generate_complexity_figure(input_files, input_order_pairs, use_gigaqbx_fmm):
    subtitles = []
    for fmm_order, qbx_order in input_order_pairs:
        subtitles.append(rf"$pqbx = {qbx_order}$, $pfmm = {fmm_order}")

    labeling = GigaQBXPerfLabeling if use_gigaqbx_fmm else QBXFMMPerfLabeling

    x_values = [[] for _ in range(len(input_files))]
    y_values = [[] for _ in range(len(input_files))]

    for i, input_file in enumerate(input_files):
        for row in csv.DictReader(input_file)
            x_values[i].append(int(row["nparticles"]))
            y_values[i].append(row)

    ylabel = r"Cost $\sim$ Number of Flops"
    xlabel = "Number of Particles"
    name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    make_complexity_figure(
            subtitles, x_values, y_values, labeling, ylabel, xlabel, name,
            size_inches=(7, 2))


def initialize_axes(ax, title, xlabel=None, ylabel=None, grid_axes=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict={"size": FONTSIZE})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict={"size": FONTSIZE})
    if grid_axes is None:
        grid_axes = "both"
    ax.grid(lw=LINEWIDTH / 2, which="major", axis=grid_axes)


def make_complexity_figure(subtitles, x_values, y_values, labeling, ylabel,
                          xlabel, name, ylimits=None, size_inches=None,
                          subplots_adjust=None, plot_title=None,
                          plot_kind="loglog", postproc_func=None,
                          summary_labels=None, grid_axes=None):
    """Generate a figure with *n* subplots

    Parameters:
        subtitles: List of *n* subtitles
        x_values: List of *n* lists of x values
        y_values: List of *n* lists of results, which are dictionaries mapping
            stage names to costs
        labeling: Subclass of *QBXPerfLabelingBase*
        ylabel: Label for y axis
        xlabel: Label for x axis
        ylimits: List of *n* y limits, or *None*
        name: Output file name
        size_inches: Passed to Figure.set_size_inches()
        subplots_adjust: Passed to Figure.subplots_adjust()
        plot_title: Plot tile

    """
    fig, axes = plt.subplots(1, len(subtitles))

    if size_inches:
        fig.set_size_inches(*size_inches)

    if len(subtitles) == 1:
        axes = [axes]

    if ylimits is None:
        ylimits = (None,) * len(subtitles)

    plot_options = dict(linewidth=LINEWIDTH, markersize=3)

    for iax, (ax, axtitle) in enumerate(zip(axes, subtitles)):
        if iax > 0:
            ylabel = None
        initialize_axes(ax, axtitle, xlabel=xlabel, ylabel=ylabel, grid_axes=grid_axes)

    # Generate results.
    for xs, ys, lim, ax in zip(x_values, y_values, ylimits, axes):
        if lim:
            ax.set_ylim(*lim)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_kind == "loglog":
            plotter = ax.loglog
        elif plot_kind == "semilogy":
            plotter = ax.semilogy
        else:
            raise ValueError("unknown plot kind")

        # features
        labels = []
        for feature, label, style, color in zip(
                labeling.perf_features,
                labeling.perf_labels,
                labeling.perf_line_styles,
                labeling.perf_colors):
            ylist = [y[feature] for y in ys]
            l, = plotter(xs, ylist, style, color=color, label=label, **plot_options)
            labels.append(l)

        summary_values = [
                sum(val for key, val in y.items()
                    if key in
                        labeling.perf_features + labeling.silent_summed_features)
                for y in ys]

        # summary
        l, = plotter(xs,
                summary_values,
                labeling.summary_line_style,
                label=labeling.summary_label,
                color=labeling.summary_color,
                **plot_options)

        labels.append(l)

        if summary_labels:
            # label
            for x, y, l in zip(xs, summary_values, summary_labels):
                ax.text(
                        x, y * 1.3, l, ha="center", va="bottom",
                        fontsize=SMALLFONTSIZE)
            miny, maxy = ax.get_ylim()
            ax.set_ylim([miny, maxy * 1.7])

    if postproc_func:
        postproc_func(fig)

    fig.legend(labels, labeling.perf_labels + ("all",), loc="center right",
               fontsize=SMALLFONTSIZE)

    suffix = "pdf" if GENERATE_PDF else "pgf"

    if plot_title:
        fig.suptitle(plot_title, fontsize=FONTSIZE)

    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    outfile = f"{OUTPUT_DIR}/complexity-{name}.{suffix}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")

    print(f"Wrote {outfile}")

# }}}


def main():
    with open_data_file("green-error-results-gigaqbx.csv", newline="") as infile:
        generate_green_error_table(infile, scheme_name="gigaqbx")

    with open_data_file("green-error-results-qbxfmm.csv", newline="") as infile:
        generate_green_error_table(infile, scheme_name="qbxfmm")

    """
    with open_data_file("green-error-results-gigaqbx-25.csv", newline="") as infile_green,\
            open_data_file("bvp-results.csv", newline="") as infile_bvp:
        generate_bvp_error_table(infile_bvp, infile_green)

    with open_data_file("particle-distributions.csv", newline="") as infile:
        generate_particle_distribution_table(infile)
    """

    """
    with open_data_file("complexity-results-fmm7-qbx3-gigaqbx.csv", newline="") as infile_fmm7_qbx3,\
             open_data_file("complexity-results-fmm15-qbx10-gigaqbx.csv", newline="") as infile_fmm15_qbx10:
        input_files = (infile_fmm7_qbx3, infile_fmm15_qbx10)
        input_order_pairs = ((7, 3), (15, 10))
        generate_complexity_figure(input_files, input_order_pairs, use_gigaqbx_fmm=True)
    """

    with open_data_file("complexity-results-fmm15-qbx3-qbxfmm.csv", newline="") as infile_fmm15_qbx3,\
             open_data_file("complexity-results-fmm30-qbx7-qbxfmm.csv", newline="") as infile_fmm30_qbx7:
        input_files = (infile_fmm15_qbx3, infile_fmm30_qbx7)
        input_order_pairs = ((15, 3), (30, 7))
        generate_complexity_figure(input_files, input_order_pairs, use_gigaqbx_fmm=False)

    # run(run_complexity_experiment, use_gigaqbx_fmm=True, compute_wall_times=False)
    # run(run_complexity_experiment, use_gigaqbx_fmm=False)
    # run(run_from_sep_smaller_threshold_complexity_experiment)

    # run(run_wall_time_experiment, use_gigaqbx_fmm=True)
    # run(run_wall_time_experiment, use_gigaqbx_fmm=False)
    # run(run_level_restriction_experiment)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
