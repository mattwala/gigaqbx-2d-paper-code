#!/usr/bin/env python3

import argparse
import collections
import contextlib
import csv
import functools
import logging
import re
import os

import numpy as np

# Whether to generate a PDF file. If False, will generate pgf.
GENERATE_PDF = True


import matplotlib  # noqa
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa


SMALLFONTSIZE = 8
FONTSIZE = 10
LINEWIDTH = 0.5


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_matplotlib():
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    plt.rc("pgf", texsystem="pdflatex")
    plt.rc("pgf", preamble=[
            r"\usepackage{amsmath}",
            r"\providecommand{\pqbx}{p_{\text{QBX}}}",
            r"\providecommand{\pfmm}{p_{\text{FMM}}}"])
    # https://stackoverflow.com/questions/40424249/vertical-alignment-of-matplotlib-legend-labels-with-latex-math
    plt.rc(("text.latex",), preview=True)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("axes", labelsize=1)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", linewidth=LINEWIDTH)
    plt.rc("pgf", rcfonts=False)
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


my_open = functools.partial(open_data_file, newline="")


def open_output_file(filename, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **kwargs)


# {{{ utils

OutputTable = collections.namedtuple(
        "OutputTable", "rows, headers, column_formats, filename")


def print_table(table):
    with open_output_file(table.filename) as outfile:
        def my_print(s):
            print(s, file=outfile)
        my_print(r"\begin{tabular}{%s}" % table.column_formats)
        my_print(r"\toprule")
        if isinstance(table.headers[0], (list, tuple)):
            for header_row in table.headers:
                my_print(" & ".join(header_row) + r"\\")
        else:
            my_print(" & ".join(table.headers) + r"\\")
        my_print(r"\midrule")
        for row in table.rows:
            my_print(" & ".join(row) + r"\\")
        my_print(r"\bottomrule")
        my_print(r"\end{tabular}")
    logger.info("Wrote %s", os.path.join(OUTPUT_DIR, table.filename))


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
    # Converged if err is within 0.1% of ref or lower.
    return err <= ref * 1.001

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

    fmm_orders = sorted(
            fmm_orders, key=lambda order: -1 if order == "inf" else order)
    qbx_orders = sorted(qbx_orders)

    # {{{ generate convergence data

    def estimate_convergence(results_table):
        converged = {}

        for q in qbx_orders:
            converged["inf", q] = False

        for p, next_p in zip(fmm_orders[1:], fmm_orders[2:] + [fmm_orders[0]]):
            for q in qbx_orders:
                converged[p, q] = False
                if next_p != "inf":
                    converged[next_p, q] = False

                if is_converged(results_table[p, q], results_table[next_p, q]):
                    converged[p, q] = True
                    if next_p != "inf":
                        converged[next_p, q] = True

        return converged

    converged_l2 = estimate_convergence(results_l2)
    converged_linf = estimate_convergence(results_linf)

    # }}}

    headers = (
            [r"{$(1/2)^{\pfmm+1}$}", r"{$\pfmm$}"]
            + [r"{$\pqbx=%d$}" % p for p in qbx_orders])

    column_formats = "".join([
            "S[table-format = 1e-1, round-precision = 0]",
            "c"] + ["S"] * len(qbx_orders))

    # {{{ build table

    def build_table(results, converged):
        table = []

        for p in fmm_orders:
            if p == "inf":
                fmm_error = "{0}"
            else:
                fmm_error = 2 ** - (p + 1)
            row = [fmt(fmm_error)]
            if p == "inf":
                row.append("(direct)")
            else:
                row.append(str(p))
            for q in qbx_orders:
                err = fmt(results[p, q])
                row.append(converged_fmt(err, converged[p, q]))
            table.append(row)

        return table

    table_l2 = build_table(results_l2, converged_l2)
    table_linf = build_table(results_linf, converged_linf)

    # }}}

    return (
        OutputTable(
            table_l2, headers, column_formats,
            f"green-error-l2-{scheme_name}.tex"),
        OutputTable(
            table_linf, headers, column_formats,
            f"green-error-linf-{scheme_name}.tex"))

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

    def build_table(
            bvp_errors, converged_bvp_errors, green_errors,
            converged_green_errors):
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

    table_l2 = build_table(
            bvp_error_l2, converged_bvp_error_l2, green_error_l2,
            converged_green_error_l2)

    table_linf = build_table(
            bvp_error_linf, converged_bvp_error_linf,
            green_error_linf, converged_green_error_linf)

    return (
            OutputTable(table_l2, headers, column_formats, "bvp-l2.tex"),
            OutputTable(table_linf, headers, column_formats, "bvp-linf.tex"))

# }}}


# {{{ particle distribution table

PERCENTILES = (20, 40, 60, 80, 100)


def generate_particle_distribution_table(infile):
    rows = []

    headers = [
            [
                r"\cellcenter{\multirow{2}{*}{$n$}}",
                r"\cellcenter{\multirow{2}{*}{$N_S$}}",
                r"\cellcenter{\multirow{2}{*}{$M_C$}}",
                r"\multicolumn{5}{c}{Percentiles}",
                ],
            [
                r"\cmidrule(lr){4-8}",
                "",
                "",
                ] + [r"\cellcenter{%d\%%}" % pct for pct in PERCENTILES]]

    for entry in csv.DictReader(infile):
        row = [
                str(entry["n_arms"]),
                str(entry["nsources"]),
                "%.1f" % float(entry["avg"]),
                ]
        row.extend(
                "%.1f" % float(entry["percentile_%d" % pct])
                for pct in PERCENTILES)
        rows.append(row)

    rows.sort(key=lambda row: int(row[0]))

    return OutputTable(
            rows, headers, "r" * len(rows[0]), "particle-distributions.tex")

# }}}


# {{{ complexity results

_colors = plt.cm.Paired.colors  # pylint:disable=no-member


class Colors(object):
    LIGHT_BLUE = _colors[0]
    BLUE = _colors[1]
    LIGHT_GREEN = _colors[2]
    GREEN = _colors[3]
    RED = _colors[5]
    ORANGE = _colors[7]
    PURPLE = _colors[9]


class QBXPerfLabelingBase(object):

    summary_line_style = ".--"

    summary_label = "all"

    summary_color = Colors.RED


class QBXPerfLabelingSummaryOnly(QBXPerfLabelingBase):

    silent_summed_features = ("op_count",)

    perf_features = ()


class QBXPerfLabelingWithStages(QBXPerfLabelingBase):

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


class GigaQBXPerfLabeling(QBXPerfLabelingWithStages):

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

    perf_line_styles = ("x-", "+-", ".-", "*-", "s-", "d-")

    perf_colors = (
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.LIGHT_BLUE,
            Colors.BLUE,
            Colors.LIGHT_GREEN,
            Colors.GREEN)


class QBXFMMPerfLabeling(QBXPerfLabelingWithStages):

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

    perf_line_styles = ("x-", "+-", "*-", "d-")

    perf_colors = (
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.BLUE,
            Colors.GREEN)


def generate_complexity_figure(
        input_file, input_order_pairs, use_gigaqbx_fmm,
        summary_labels=None, postproc_func=lambda fig: None):
    subtitles = []
    for fmm_order, qbx_order in input_order_pairs:
        subtitles.append(rf"$\pqbx = {qbx_order}$, $\pfmm = {fmm_order}")

    labeling = GigaQBXPerfLabeling if use_gigaqbx_fmm else QBXFMMPerfLabeling

    x_values = [[] for _ in range(len(input_order_pairs))]
    y_values = [[] for _ in range(len(input_order_pairs))]

    rows = list(csv.DictReader(input_file))

    for i, order_pair in enumerate(input_order_pairs):
        for row in rows:
            if (int(row["fmm_order"]), int(row["qbx_order"])) != order_pair:
                continue
            x_values[i].append(int(row["nparticles"]))
            result = {}
            for key in row:
                try:
                    result[key] = int(row[key])
                except ValueError:
                    pass
            y_values[i].append(result)

    ylabel = r"Cost $\sim$ Number of Flops"
    xlabel = "Number of Particles"
    name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    make_complexity_figure(
            subtitles, x_values, y_values, labeling, ylabel, xlabel, name,
            size_inches=(7, 2.5), summary_labels=summary_labels,
            postproc_func=postproc_func)


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


def make_complexity_figure(
        subtitles, x_values, y_values, labeling, ylabel,
        xlabel, name, ylimits=None, size_inches=None, subplots_adjust=None,
        plot_title=None, plot_kind="loglog", postproc_func=None,
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
        initialize_axes(
                ax, axtitle, xlabel=xlabel, ylabel=ylabel, grid_axes=grid_axes)

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
            l, = plotter(
                    xs, ylist, style,
                    color=color, label=label, **plot_options)
            labels.append(l)

        summary_values = [
                sum(val for key, val in y.items()
                    if key in labeling.perf_features
                    + labeling.silent_summed_features)
                for y in ys]

        # summary
        l, = plotter(
                xs,
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

    outfile = os.path.join(OUTPUT_DIR, f"complexity-{name}.{suffix}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    logger.info("Wrote %s", outfile)

# }}}


# {{{ green error summary table

def generate_green_error_summary_table(
        infile, fmm_and_qbx_order_pairs, scheme_name):
    rows_by_arms = {}

    for row in csv.DictReader(infile):
        n_arms = int(row["n_arms"])
        fmm_order = int(row["fmm_order"])
        qbx_order = int(row["qbx_order"])
        if n_arms not in rows_by_arms:
            rows_by_arms[n_arms] = [None] * len(fmm_and_qbx_order_pairs)
        result = rows_by_arms[n_arms]
        index = fmm_and_qbx_order_pairs.index((fmm_order, qbx_order))
        result[index] = row["err_linf"]

    table = []
    headers = ["$n$"] + [
            r"{{$\pfmm={p}$, $\pqbx={q}$}}".format(p=p, q=q)
            for p, q in fmm_and_qbx_order_pairs]

    for n_arms in sorted(rows_by_arms):
        row = [str(n_arms)] + rows_by_arms[n_arms]
        table.append(row)

    mean_row = [r"\cmidrule{1-%d}(avg.)" % (1 + len(fmm_and_qbx_order_pairs))]
    for i in range(1, 1 + len(fmm_and_qbx_order_pairs)):
        mean_row.append(
                "%.17e" % np.mean([float(row[i]) for row in table]))

    table.append(mean_row)
    column_formats = "r" + "S" * len(fmm_and_qbx_order_pairs)

    return OutputTable(
            table, headers, column_formats,
            f"complexity-green-errors-{scheme_name}.tex")

# }}}


# {{{ complexity comparison table

Column = collections.namedtuple(
        "Column", "label, order_pair, input_source, fmt")

ComparisonColumn = collections.namedtuple(
        "ComparisonColumn", "label, numerator_column, denominator_column, fmt")

ColumnGroup = collections.namedtuple("ColumnGroup", "label, columns")


def generate_complexity_comparison_table(
        input_files, perf_labelings, spec, include_averages, label):
    """
    Inputs:

        input_files: List of input sources

        perf_labelings: A PerfLabeling for each input source describing how to
            summarize data

        spec: A tuple of elements describing the column specification

        include_averages: Whether to include a row at the bottom summarizing
            average ratios

        scheme_name: A label for the output file
    """

    # {{{ gather results

    results_by_file = [dict() for _ in range(len(input_files))]
    arms = set()

    for infile, perf_labeling, results in \
            zip(input_files, perf_labelings, results_by_file):
        for row in csv.DictReader(infile):
            n_arms = int(row["n_arms"])
            arms.add(n_arms)
            fmm_order = int(row["fmm_order"])
            qbx_order = int(row["qbx_order"])
            value = sum(
                    int(val) for key, val in row.items()
                    if key in perf_labeling.perf_features
                    + perf_labeling.silent_summed_features)
            results[n_arms, fmm_order, qbx_order] = value

    # }}}

    # {{{ construct rows

    rows = []

    def build_row(row, n_arms, spec):
        for item in spec:
            if isinstance(item, ColumnGroup):
                build_row(row, n_arms, item.columns)

            elif isinstance(item, Column):
                key = (n_arms,) + item.order_pair
                row.append(results_by_file[item.input_source][key])

            elif isinstance(item, ComparisonColumn):
                row.append(
                        row[item.numerator_column]
                        / row[item.denominator_column])

    for n_arms in sorted(arms):
        row = [n_arms]
        build_row(row, n_arms, spec)
        rows.append(row)

    if include_averages:
        avg_row = [r"\cmidrule{1-%d}(avg.)" % len(rows[0])]

        def build_avg_row(spec):
            for item in spec:
                if isinstance(item, ColumnGroup):
                    build_avg_row(item.columns)

                elif isinstance(item, Column):
                    avg_row.append(None)

                elif isinstance(item, ComparisonColumn):
                    i = len(avg_row)
                    avg_row.append(sum(row[i] for row in rows) / len(rows))

        build_avg_row(spec)
        rows.append(avg_row)

    formats = ["%d"]

    def get_formats(spec):
        for item in spec:
            if isinstance(item, ColumnGroup):
                get_formats(item.columns)

            elif isinstance(item, (Column, ComparisonColumn)):
                formats.append(item.fmt)

    get_formats(spec)

    def stringify(args):
        index, item = args
        if item is None:
            return "---"
        elif isinstance(item, str):
            return item
        else:
            return formats[index] % item

    for row in rows:
        row[:] = map(stringify, enumerate(row))

    # }}}

    # {{{ construct headers

    headers = [[""]]
    lengths = [1]

    def build_header(irow, spec):
        for item in spec:
            if isinstance(item, ColumnGroup):
                if len(headers) == 1 + irow:
                    headers.append([""])
                    lengths.append(1)

                ncols = len(item.columns)
                label = r"\multicolumn{%d}{c}{%s}" % (ncols, item.label)
                headers[irow].append(label)
                lengths[irow] += ncols

                rule_start = 1 + lengths[irow + 1]
                rule_end = rule_start + ncols - 1
                headers[irow + 1][0] += (
                        r"\cmidrule(lr){%d-%d}" % (rule_start, rule_end))
                lengths[irow + 1] = rule_end

                build_header(irow + 1, item.columns)

            elif isinstance(item, (Column, ComparisonColumn)):
                headers[irow].append(r"\cellcenter{%s}" % item.label)

    build_header(0, spec)
    headers[0][0] = r"\cellcenter{\multirow{%d}{*}{$n$}}" % len(headers)

    # }}}

    return OutputTable(
            rows, headers, "r" * len(rows[0]),
            f"complexity-summary-{label}.tex")

# }}}


# {{{ wall time comparison table

def generate_wall_time_comparison_table(
        input_files, order_pairs, input_labels, comparison_columns,
        comparison_labels, scheme_name):
    rows_by_arms = {}

    for i, (infile, order_pair) in enumerate(zip(input_files, order_pairs)):
        for row in csv.DictReader(infile):
            n_arms = int(row["n_arms"])
            fmm_order = int(row["fmm_order"])
            qbx_order = int(row["qbx_order"])
            if (fmm_order, qbx_order) != order_pair:
                continue
            if n_arms not in rows_by_arms:
                rows_by_arms[n_arms] = [[] for _ in range(len(input_files))]
            result = rows_by_arms[n_arms][i]
            result.append(float(row["time"]))

    for row in rows_by_arms.values():
        row[:] = [np.mean(times) for times in row]

    totals = [0] * len(comparison_columns)

    for i, (numerator_col, denominator_col) in enumerate(comparison_columns):
        for row in rows_by_arms.values():
            val = float(row[numerator_col]) / float(row[denominator_col])
            row.append(val)
            totals[i] += val

    for row in rows_by_arms.values():
        row[:] = ["%.2f" % item for item in row]

    table = []
    headers = ["$n$"] + list(input_labels) + list(comparison_labels)

    for n_arms in sorted(rows_by_arms):
        row = [str(n_arms)] + rows_by_arms[n_arms]
        table.append(row)

    nrows = len(rows_by_arms)
    ncols = 1 + len(input_files) + len(comparison_columns)
    mean_row = [r"\cmidrule{1-%d}(avg.)" % ncols]
    for _ in range(len(input_files)):
        mean_row.append("---")
    for i, val in enumerate(totals):
        mean_row.append("%.2f" % (val / nrows))
    table.append(mean_row)

    ncols = 1 + len(input_labels) + len(comparison_labels)
    column_formats = "r" * ncols

    return OutputTable(
            table, headers, column_formats,
            f"wall-time-summary-{scheme_name}.tex")

# }}}


# {{{ data comparison table

def generate_data_comparison_table(
        old_table, new_table, cols_to_compare, comparison_func,
        formatter, output_fname, column_formats=None):

    rows = []

    for old_row, new_row in zip(old_table.rows, new_table.rows):
        assert len(old_row) == len(new_row)

        row = []
        corrected_row = []

        for icol, (old_item, new_item) in enumerate(zip(old_row, new_row)):
            if icol not in cols_to_compare:
                assert 0 == comparison_func(icol, old_item, new_item)
                row.append(formatter(icol, old_item))
                corrected_row.append("")
                continue

            result = comparison_func(icol, old_item, new_item)

            if 0 == result:
                row.append(formatter(icol, old_item))
                corrected_row.append("")
            else:
                label = "correctedHigher" if result < 0 else "correctedLower"
                row.append(
                        r"\uncorrected{%s}" % formatter(icol, old_item))
                corrected_row.append(
                        r"\%s{%s}" % (label, formatter(icol, new_item)))

        if len(rows) and not row[0].startswith(r"\cmid"):
            row[0] = r"\hdashline{}" + row[0]

        rows.append(row)
        if not all(item == "" for item in corrected_row):
            rows.append(corrected_row)

    assert old_table.headers == new_table.headers

    if column_formats is None:
        assert old_table.column_formats == new_table.column_formats
        column_formats = old_table.column_formats

    return OutputTable(
            rows,
            old_table.headers,
            column_formats,
            output_fname)

# }}}


EXPERIMENTS = (
        "wall-time",
        "green-error",
        "bvp",
        "particle-distributions",
        "green-error-comparison",
        "bvp-comparison",
        "particle-distributions-comparison",
        "complexity",
        "from-sep-smaller-threshold",
        "op-count-comparison")


GIGAQBX_ORDER_PAIRS = ((10, 3), (15, 7))
QBXFMM_ORDER_PAIRS = ((15, 3), (30, 7))


def generate_wall_time_outputs():
    wall_time_comparison_files = (
            "wall-time-results-qbxfmm.csv",
            "wall-time-results-gigaqbx.csv",
    )

    for order_pairs in zip(QBXFMM_ORDER_PAIRS, GIGAQBX_ORDER_PAIRS):
        with contextlib.ExitStack() as stack:
            input_files = [
                    stack.enter_context(my_open(fname))
                    for fname in wall_time_comparison_files]

            table = generate_wall_time_comparison_table(
                    input_files=input_files,
                    order_pairs=order_pairs,
                    input_labels=(
                        r"$t_\text{qbxfmm}$", r"$t_\text{giga}$"),
                    comparison_columns=((1, 0),),
                    comparison_labels=(
                        r"$t_\text{giga} / t_\text{qbxfmm}$",),
                    scheme_name="qbx%d" % order_pairs[0][1])
            print_table(table)


def generate_green_error_outputs():
    with my_open("green-error-results-gigaqbx.csv") as infile:
        table_l2, table_linf = (
                generate_green_error_table(infile, scheme_name="gigaqbx"))
        print_table(table_l2)
        print_table(table_linf)

    with my_open("green-error-results-qbxfmm.csv") as infile:
        table_l2, table_linf = (
                generate_green_error_table(infile, scheme_name="qbxfmm"))
        print_table(table_l2)
        print_table(table_linf)


def compare_vals(a, b):
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1


def generate_green_error_comparison_outputs():
    with my_open("old-green-error-results-gigaqbx.csv") as infile:
        _, old_table_linf = (
                generate_green_error_table(infile, scheme_name="gigaqbx"))

    with my_open("green-error-results-gigaqbx.csv") as infile:
        _, new_table_linf = (
                generate_green_error_table(infile, scheme_name="gigaqbx"))

    def cmp(col, old_val, new_val):
        if col in (0, 1):
            return compare_vals(old_val, new_val)
        old_val = float(old_val.strip(r"\converged{}"))
        new_val = float(new_val.strip(r"\converged{}"))
        if abs(old_val - new_val) < 0.01 * old_val:
            return 0
        else:
            return -1 if old_val < new_val else 1

    def formatter(col, val):
        if col == 1:
            return val
        spec = "%.0e" if col == 0 else "%.2e"
        val = val.strip(r"\converged{}")
        if float(val) == 0:
            return "0"
        val = spec % float(val)
        eindex = val.index("e")
        base = val[:eindex]
        exp = int(val[eindex+1:])
        result = fr"{base} \times 10^{{{exp}}}"
        result = f"${result}$"
        return result

    table = generate_data_comparison_table(
            old_table_linf, new_table_linf, (2, 3, 4, 5),
            cmp, formatter,
            "green-error-results-old-vs-new-gigaqbx.tex",
            column_formats="c" * 6)

    print_table(table)


def generate_particle_distributions_comparison_outputs():
    with my_open("particle-distributions.csv") as infile:
        new_table = generate_particle_distribution_table(infile)

    with my_open("old-particle-distributions.csv") as infile:
        old_table = generate_particle_distribution_table(infile)

    def cmp(col, old_val, new_val):
        return compare_vals(float(old_val), float(new_val))

    def formatter(col, val):
        return val

    table = generate_data_comparison_table(
            old_table, new_table, (2, 3, 4, 5, 6, 7),
            cmp, formatter,
            "particle-distributions-results-old-vs-new.tex")

    print_table(table)


def generate_bvp_comparison_outputs():
    with my_open("bvp-green-error-results-gigaqbx.csv") as infile_green,\
            my_open("bvp-results.csv") as infile_bvp:
        _, new_table_linf = (
                generate_bvp_error_table(infile_bvp, infile_green))

    with my_open("old-bvp-green-error-results-gigaqbx.csv") as infile_green,\
            my_open("old-bvp-results.csv") as infile_bvp:
        _, old_table_linf = (
                generate_bvp_error_table(infile_bvp, infile_green))

    def cmp(col, old_val, new_val):
        if col in (0, 1) or "" in (old_val, new_val):
            return compare_vals(old_val, new_val)
        old_val = float(old_val.strip(r"\converged{}"))
        new_val = float(new_val.strip(r"\converged{}"))
        if abs(old_val - new_val) < 0.01 * old_val:
            return 0
        return -1 if old_val < new_val else 1

    def formatter(col, val):
        if not val:
            return val
        if col in (1, 3, 5, 7, 9):
            return val
        has_midrule = val.startswith(r"\cmidrule")
        spec = "%.0e" if col == 0 else "%.2e"
        val = re.findall(r"[0-9]+\.[0-9]*e[+-][0-9]+", val)[0]
        val = spec % float(val)
        eindex = val.index("e")
        base = val[:eindex]
        exp = int(val[eindex+1:])
        result = fr"{base} \times 10^{{{exp}}}"
        result = f"${result}$"
        if has_midrule:
            result = r"\cmidrule{1-10}" + result
        return result

    table = generate_data_comparison_table(
            old_table_linf, new_table_linf, list(range(2, 10)),
            cmp, formatter,
            "bvp-results-old-vs-new-gigaqbx.tex",
            column_formats="c" * 10)

    print_table(table)


def generate_bvp_outputs():
    with my_open("bvp-green-error-results-gigaqbx.csv") as infile_green,\
            my_open("bvp-results.csv") as infile_bvp:
        table_l2, table_linf = (
                generate_bvp_error_table(infile_bvp, infile_green))
        print_table(table_l2)
        print_table(table_linf)


def generate_particle_distributions_outputs():
    with my_open("particle-distributions.csv") as infile:
        table = generate_particle_distribution_table(infile)
        print_table(table)


def generate_complexity_outputs():
    def postproc_func(fig):
        for ax in fig.get_axes():
            ax.set_xlim(1e4, 2*1e6)
            ax.set_ylim(1e5, 2e9)

    summary_labels = tuple(r"$\gamma_{%d}" % n for n in range(5, 66, 10))

    with my_open("complexity-results-gigaqbx-threshold15.csv")\
            as input_file:
        generate_complexity_figure(
                input_file, GIGAQBX_ORDER_PAIRS, use_gigaqbx_fmm=True,
                summary_labels=summary_labels, postproc_func=postproc_func)

    with my_open("complexity-results-qbxfmm-threshold15.csv")\
            as input_file:
        generate_complexity_figure(
                input_file, QBXFMM_ORDER_PAIRS, use_gigaqbx_fmm=False,
                summary_labels=summary_labels, postproc_func=postproc_func)

    complexity_comparison_files = (
            "complexity-results-qbxfmm-threshold15.csv",
            "complexity-results-gigaqbx-threshold15.csv",
    )

    complexity_comparison_labelings = (
            QBXFMMPerfLabeling,
            GigaQBXPerfLabeling)

    for order_pairs in zip(QBXFMM_ORDER_PAIRS, GIGAQBX_ORDER_PAIRS):
        with contextlib.ExitStack() as stack:
            input_files = [
                    stack.enter_context(open_data_file(fname, newline=""))
                    for fname in complexity_comparison_files]

            label = "qbx%d-qbxfmm-vs-gigaqbx" % order_pairs[0][1]

            spec = (
                    Column(r"\#Ops (QBX FMM)", order_pairs[0], 0, "%d"),
                    Column(r"\#Ops (GIGAQBX)", order_pairs[1], 1, "%d"),
                    ComparisonColumn("Ratio", 2, 1, "%.2f"))

            table = generate_complexity_comparison_table(
                    input_files,
                    complexity_comparison_labelings,
                    spec,
                    include_averages=True,
                    label=label)
            print_table(table)

    # Green error summaries for complexity experiment
    with my_open("complexity-green-error-results-gigaqbx.csv") as infile:
        table = generate_green_error_summary_table(
                infile, GIGAQBX_ORDER_PAIRS, scheme_name="gigaqbx")
        print_table(table)

    with my_open("complexity-green-error-results-qbxfmm.csv") as infile:
        table = generate_green_error_summary_table(
                infile, QBXFMM_ORDER_PAIRS, scheme_name="qbxfmm")
        print_table(table)


def generate_from_sep_smaller_threshold_outputs():
    complexity_comparison_files = (
            "complexity-results-gigaqbx-threshold0.csv",
            "complexity-results-gigaqbx-threshold15.csv",
    )

    for order_pair in GIGAQBX_ORDER_PAIRS:
        with contextlib.ExitStack() as stack:
            input_files = [
                    stack.enter_context(open_data_file(fname, newline=""))
                    for fname in complexity_comparison_files]
            label = (
                    "fmm%d-qbx%d-gigaqbx-threshold0-vs-threshold15"
                    % order_pair)
            table = generate_complexity_comparison_table(
                    input_files,
                    (GigaQBXPerfLabeling,) * 2,
                    spec=(
                        Column(r"\#Ops (thresh.=0)", order_pair, 0, "%d"),
                        Column(r"\#Ops (thresh.=15)", order_pair, 1, "%d"),
                        ComparisonColumn("Ratio", 2, 1, "%.2f")),
                    include_averages=True,
                    label=label)
            print_table(table)


def generate_op_count_comparison_outputs():
    gigaqbx_complexity_results = (
            "old-summary-op-counts-gigaqbx.csv",
            "complexity-results-gigaqbx-threshold15.csv",

    )

    qbxfmm_complexity_results = (
            "old-summary-op-counts-qbxfmm.csv",
            "complexity-results-qbxfmm-threshold15.csv",
    )

    for label, perf_labeling, input_file_names, order_pairs in \
            zip(
                ("gigaqbx", "qbxfmm"),
                (GigaQBXPerfLabeling, QBXFMMPerfLabeling),
                (gigaqbx_complexity_results, qbxfmm_complexity_results),
                (GIGAQBX_ORDER_PAIRS, QBXFMM_ORDER_PAIRS)):

        def make_group_label(order_pair):
            fmm_order, qbx_order = order_pair
            return fr"$\pqbx={qbx_order}$, $\pfmm={fmm_order}$"

        old = r"\#Ops (Old)"
        new = r"\#Ops (New)"

        spec = (
            ColumnGroup(make_group_label(order_pairs[0]), (
                Column(old, order_pairs[0], 0, "%d"),
                Column(new, order_pairs[0], 1, "%d"),
                ComparisonColumn("Ratio", 2, 1, "%.2f"))),
            ColumnGroup(make_group_label(order_pairs[1]), (
                Column(old, order_pairs[1], 0, "%d"),
                Column(new, order_pairs[1], 1, "%d"),
                ComparisonColumn("Ratio", 5, 4, "%.2f"))))

        with contextlib.ExitStack() as stack:
            input_files = [
                    stack.enter_context(open_data_file(fname, newline=""))
                    for fname in input_file_names]

            table = generate_complexity_comparison_table(
                    input_files,
                    (QBXPerfLabelingSummaryOnly, perf_labeling),
                    spec,
                    include_averages=False,
                    label=f"{label}-old-vs-new")
            print_table(table)


def gen_figures_and_tables(experiments):
    # Wall time comparison
    if "wall-time" in experiments:
        generate_wall_time_outputs()

    # Green error tables
    if "green-error" in experiments:
        generate_green_error_outputs()

    if "green-error-comparison" in experiments:
        generate_green_error_comparison_outputs()

    # BVP error table
    if "bvp" in experiments:
        generate_bvp_outputs()

    if "bvp-comparison" in experiments:
        generate_bvp_comparison_outputs()

    # Particle distributions table
    if "particle-distributions" in experiments:
        generate_particle_distributions_outputs()

    # Green error tables
    if "particle-distributions-comparison" in experiments:
        generate_particle_distributions_comparison_outputs()

    # Complexity result graphs
    if "complexity" in experiments:
        generate_complexity_outputs()

    # Effect of threshold on operation counts
    if "from-sep-smaller-threshold" in experiments:
        generate_from_sep_smaller_threshold_outputs()

    if "op-count-comparison" in experiments:
        generate_op_count_comparison_outputs()


def main():
    names = ["'%s'" % name for name in EXPERIMENTS]
    names[-1] = "and " + names[-1]

    description = (
            "This script postprocesses results for one or more experiments. "
            " The names of the experiments are: " + ", ".join(names)
            + ".")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
            "-x",
            metavar="experiment-name",
            action="append",
            dest="experiments",
            default=[],
            help="Postprocess results for an experiment "
                 "(may be specified multiple times)")

    parser.add_argument(
            "--all",
            action="store_true",
            dest="run_all",
            help="Postprocess results for all available experiments")

    parser.add_argument(
            "--except",
            action="append",
            metavar="experiment-name",
            dest="run_except",
            default=[],
            help="Do not postprocess results for an experiment "
                 "(may be specified multiple times)")

    result = parser.parse_args()

    experiments = set()

    if result.run_all:
        experiments = set(EXPERIMENTS)
    experiments |= set(result.experiments)
    experiments -= set(result.run_except)

    gen_figures_and_tables(experiments)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
