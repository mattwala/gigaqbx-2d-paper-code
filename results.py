#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from boxtree.area_query import AreaQueryElementwiseTemplate
from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, NArmedStarfish, drop, n_gon, qbx_peanut,
        WobblyCircle, make_curve_mesh, starfish)
from pytential import bind, sym, norm
from pytools import memoize

import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# tree parameters
# * gigaqbx accuracy
# * qbxfmm accuracy
# * particle distribution
# * gigaqbx complexity
# * qbxfmm complexity

# Whether to generate a PDF file. If False, will generate pgf.
GENERATE_PDF = True


def switch_matplotlib_to_agg():
    import matplotlib
    matplotlib.use("pgf")


switch_matplotlib_to_agg()


class GigaQBXPaperTranslationCostModel(object):

    def __init__(self, dim, nlevels):
        from pymbolic import var
        p_qbx = var("p_qbx")
        p_fmm_by_level = np.array([var("p_fmm_lev%d" % i) for i in range(nlevels)])

        # Note: This means order n has n coeffs in 2D, not (n + 1) coeffs.
        # This is consistent with the published version.
        self.ncoeffs_fmm_by_level = p_fmm_by_level ** (dim - 1)
        self.ncoeffs_qbx = p_qbx ** (dim - 1)

    @staticmethod
    def direct():
        return 1

    def p2qbxl(self):
        return self.ncoeffs_qbx

    def p2p_tsqbx(self):
        # This term should be linear in the QBX order, which is the
        # square root of the number of QBX coefficients.
        return self.ncoeffs_qbx

    def qbxl2p(self):
        return self.ncoeffs_qbx

    def p2l(self, level):
        return self.ncoeffs_fmm_by_level[level]

    def l2p(self, level):
        return self.ncoeffs_fmm_by_level[level]

    def p2m(self, level):
        return self.ncoeffs_fmm_by_level[level]

    def m2p(self, level):
        return self.ncoeffs_fmm_by_level[level]

    def m2m(self, src_level, tgt_level):
        return self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def l2l(self, src_level, tgt_level):
        return self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2l(self, src_level, tgt_level):
        return self.e2e_cost(
                self.ncoeffs_fmm_by_level[src_level],
                self.ncoeffs_fmm_by_level[tgt_level])

    def m2qbxl(self, level):
        return self.e2e_cost(
                self.ncoeffs_fmm_by_level[level],
                self.ncoeffs_qbx)

    def l2qbxl(self, level):
        return self.e2e_cost(
                self.ncoeffs_fmm_by_level[level],
                self.ncoeffs_qbx)

    def e2e_cost(self, nsource_coeffs, ntarget_coeffs):
        return nsource_coeffs * ntarget_coeffs



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


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 4
PANELS_PER_ARM = 50
N_ARMS_FOR_GREEN_EXPERIMENT = 65
N_ARMS_FOR_BVP_EXPERIMENT = 25
if 0:
    ARMS = [3, 4, 5]
else:
    ARMS = list(range(5, 70, 10))
TCF = 0.9
DEFAULT_GIGAQBX_BACKEND = "fmmlib"
DEFAULT_QBXFMM_BACKEND = "fmmlib"
TIMING_ROUNDS = 3

# }}}


# {{{ general utils

@memoize
def get_geometry(ctx, n_arms, use_gigaqbx_fmm, fmm_backend=None,
                 from_sep_smaller_threshold=None):
    if use_gigaqbx_fmm:
        well_sep_is_n_away = 2
        stickout_factor = TCF
        with_extents = True
        max_leaf_refine_weight = 64
        if fmm_backend is None:
            fmm_backend = DEFAULT_GIGAQBX_BACKEND
    else:
        well_sep_is_n_away = 1
        stickout_factor = 0
        with_extents = False
        max_leaf_refine_weight = 128
        if fmm_backend is None:
            fmm_backend = DEFAULT_QBXFMM_BACKEND

    with cl.CommandQueue(ctx) as queue:
        return get_lpot_source(
                queue,
                partial(get_starfish_mesh, n_arms),
                PANELS_PER_ARM * n_arms,
                stickout_factor,
                well_sep_is_n_away,
                with_extents,
                max_leaf_refine_weight,
                fmm_backend=fmm_backend,
                from_sep_smaller_threshold=from_sep_smaller_threshold)


def get_starfish_mesh(n_arms, nelements, target_order):
    return make_curve_mesh(
            NArmedStarfish(n_arms, 0.8),
            np.linspace(0, 1, nelements+1),
            target_order)


def get_lpot_source(queue, mesh_getter, nelements,
                    stickout_factor, well_sep_is_n_away,
                    with_extents, max_leaf_refine_weight,
                    k=0, fmm_backend="sumpy",
                    from_sep_smaller_threshold=None):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    mesh = mesh_getter(nelements, target_order)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    refiner_extra_kwargs = {}

    if k != 0:
        refiner_extra_kwargs["kernel_length_scale"] = 5/k

    """
    from pytential.qbx.cost import CostModel
    perf_model = CostModel(
            translation_cost_model_factory=GigaQBXPaperTranslationCostModel)
    """

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            fmm_backend=fmm_backend,
            _well_sep_is_n_away=well_sep_is_n_away,
            _expansions_in_tree_have_extent=with_extents,
            _expansion_stick_out_factor=stickout_factor,
            _max_leaf_refine_weight=max_leaf_refine_weight,
            target_association_tolerance=1e-3,
            fmm_order=10, qbx_order=10,
            _from_sep_smaller_min_nsources_cumul=from_sep_smaller_threshold,
            )

    lpot_source, _ = lpot_source.with_refinement(**refiner_extra_kwargs)

    return lpot_source


def print_table(table, headers, outf, column_formats=None):
    if column_formats is None:
        column_formats = "c" * len(headers)
    with open(outf, "w") as outfile:
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
    print(f"Output written to {outf}.")

# }}}


# {{{ slp wall time

def get_slp_wall_time(lpot_source, fmm_order, qbx_order, k=0):
    queue = cl.CommandQueue(lpot_source.cl_context)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    lpot_source = lpot_source.copy(
            qbx_order=qbx_order,
            fmm_level_to_order=(
                False if fmm_order is False else lambda *args: fmm_order))

    d = lpot_source.ambient_dim

    u_sym = sym.var("u")

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    lap_k_sym = LaplaceKernel(d)
    if k == 0:
        k_sym = lap_k_sym
        knl_kwargs = {}
    else:
        k_sym = HelmholtzKernel(d)
        knl_kwargs = {"k": sym.var("k")}

    S = sym.S(k_sym, u_sym, qbx_forced_limit=-1, **knl_kwargs)  # noqa: N806

    density_discr = lpot_source.density_discr

    u = cl.array.empty(queue, density_discr.nnodes, np.float64)
    u.fill(1)

    op = bind(lpot_source, S)

    # Warmup
    op(queue, u=u, k=k)

    times = []
    from time import time as curr_time

    for i in range(TIMING_ROUNDS):
        t_start = curr_time()
        op(queue, u=u, k=k)
        t_end = curr_time()
        times.append(t_end - t_start)

    return np.mean(times)

# }}}


# {{{ green error experiment

def get_green_error(lpot_source, fmm_order, qbx_order, k=0, time=None):
    queue = cl.CommandQueue(lpot_source.cl_context)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    lpot_source = lpot_source.copy(
            qbx_order=qbx_order,
            fmm_level_to_order=(
                False if fmm_order is False else lambda *args: fmm_order))

    d = lpot_source.ambient_dim

    u_sym = sym.var("u")
    dn_u_sym = sym.var("dn_u")

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    lap_k_sym = LaplaceKernel(d)
    if k == 0:
        k_sym = lap_k_sym
        knl_kwargs = {}
    else:
        k_sym = HelmholtzKernel(d)
        knl_kwargs = {"k": sym.var("k")}

    S_part = (  # noqa: N806
            sym.S(k_sym, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs))

    D_part = (  # noqa: N806
            sym.D(k_sym, u_sym, qbx_forced_limit="avg", **knl_kwargs))

    density_discr = lpot_source.density_discr

    # {{{ compute values of a solution to the PDE

    nodes_host = density_discr.nodes().get(queue)
    normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
    normal_host = [normal[j].get() for j in range(d)]

    if k != 0:
        if d == 2:
            angle = 0.3
            wave_vec = np.array([np.cos(angle), np.sin(angle)])
            u = np.exp(1j*k*np.tensordot(wave_vec, nodes_host, axes=1))
            grad_u = 1j*k*wave_vec[:, np.newaxis]*u
        else:
            center = np.array([3, 1, 2])
            diff = nodes_host - center[:, np.newaxis]
            r = la.norm(diff, axis=0)
            u = np.exp(1j*k*r) / r
            grad_u = diff * (1j*k*u/r - u/r**2)
    else:
        center = np.array([2, 1, 2])[:d]
        diff = nodes_host - center[:, np.newaxis]
        dist_squared = np.sum(diff**2, axis=0)
        dist = np.sqrt(dist_squared)
        if d == 2:
            u = np.log(dist)
            grad_u = diff/dist_squared
        elif d == 3:
            u = 1/dist
            grad_u = -diff/dist**3
        else:
            assert False

    dn_u = 0
    for i in range(d):
        dn_u = dn_u + normal_host[i]*grad_u[i]

    # }}}

    u_dev = cl.array.to_device(queue, u)
    dn_u_dev = cl.array.to_device(queue, dn_u)
    grad_u_dev = cl.array.to_device(queue, grad_u)

    bound_S_part = bind(lpot_source, S_part)  # noqa: N806
    bound_D_part = bind(lpot_source, D_part)  # noqa: N806

    from time import time as curr_time
    t_start = curr_time()
    S_result = bound_S_part(  # noqa: N806
            queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)
    t_end = curr_time()
    if time is not None:
        time[0] = t_end - t_start

    D_result = bound_D_part(  # noqa: N806
            queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)

    scaling_l2 = 1 / norm(density_discr, queue, u_dev, p=2)
    scaling_linf = 1 / norm(density_discr, queue, u_dev, p="inf")

    error = S_result - D_result - 0.5 * u_dev

    return (
            scaling_l2 * norm(density_discr, queue, error, p=2),
            scaling_linf * norm(density_discr, queue, error, p="inf"))


def fmt(val):
    """Format a numerical table cell."""
    if isinstance(val, str):
        return val
    return f"{val:e}"


def _green_error_experiment_body(use_gigaqbx_fmm, k, fmm_and_qbx_order_pair):
    # This returns a tuple (errs_l2, errs_linf).
    cl_ctx = cl.create_some_context(interactive=False)

    fmm_order, qbx_order = fmm_and_qbx_order_pair

    if fmm_order is False:
        # The number of particles is impractical for direct eval.
        # GIGAQBX + fmmlib appears to be the most reliable fast
        # eval option.
        lpot_source = get_geometry(cl_ctx,
                                   N_ARMS_FOR_GREEN_EXPERIMENT,
                                   use_gigaqbx_fmm=True,
                                   fmm_backend="fmmlib")
    else:
        lpot_source = get_geometry(cl_ctx,
                                   N_ARMS_FOR_GREEN_EXPERIMENT,
                                   use_gigaqbx_fmm=use_gigaqbx_fmm)

    print("#" * 80)
    print("fmm_order %s, qbx_order %d" % (fmm_order, qbx_order))
    print("#" * 80)

    true_fmm_order = 30 if fmm_order is False else fmm_order
    return get_green_error(lpot_source, true_fmm_order, qbx_order, k=k)


def run_green_error_experiment(use_gigaqbx_fmm):
    def converged_fmt(item, converged):
        if not converged:
            return item
        return r"\converged{%s}" % item

    def is_converged(err, ref):
        # Converged if err is within 1% of ref or lower.
        return err <= ref * 1.01

    k = 0
    qbx_orders = [3, 5, 7, 9]
    fmm_orders = [False, 3, 5, 10, 15, 20]

    POOL_WORKERS = 12

    with multiprocessing.Pool(POOL_WORKERS) as p:
        from itertools import product
        results = p.map(
                partial(_green_error_experiment_body, use_gigaqbx_fmm, k),
                product(fmm_orders, qbx_orders))
        with open("green-error-results-%s.csv" % use_gigaqbx_fmm, "w") as outfile:
        writer = csv.DictWriter(["fmm_order", "qbx_order", "err_l2", "err_linf"])

    headers = (
        [r"{$(1/2)^{\pfmm+1}$}", r"{$\pfmm$}"]
        + [r"{$\pqbx=%d$}" % p for p in qbx_orders])

    column_formats = "".join([
        "S[table-format = 1e-1, round-precision = 0]",
        "c"] + ["S"] * len(qbx_orders))

    results_iter = iter(results)

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
            err_l2, err_linf = next(results_iter)

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
        row_l2.append(str(fmm_order) if fmm_order is not False else "(direct)")
        for e, c in zip(errs_l2, converged_l2):
            row_l2.append(converged_fmt(fmt(e), c))
        table_l2.append(row_l2)

        row_linf = [fmt(fmm_error)]
        row_linf.append(
                str(fmm_order) if fmm_order is not False else "(direct)")
        for e, c in zip(errs_linf, converged_linf):
            row_linf.append(converged_fmt(fmt(e), c))
        table_linf.append(row_linf)

    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    print_table(table_l2, headers, f"green-error-l2-{scheme_name}.tex",
                column_formats)
    print_table(table_linf, headers, f"green-error-linf-{scheme_name}.tex",
                column_formats)

# }}}


# {{{ BVP error experiment

def get_bvp_error(lpot_source, fmm_order, qbx_order, k=0, time=None):
    queue = cl.CommandQueue(lpot_source.cl_context)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    lpot_source = lpot_source.copy(
            qbx_order=qbx_order,
            fmm_level_to_order=(
                False if fmm_order is False else lambda *args: fmm_order))

    d = lpot_source.ambient_dim

    assert k == 0  # Helmholtz would require a different representation

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    lap_k_sym = LaplaceKernel(d)
    if k == 0:
        k_sym = lap_k_sym
        knl_kwargs = {}
    else:
        k_sym = HelmholtzKernel(d)
        knl_kwargs = {"k": sym.var("k")}

    density_discr = lpot_source.density_discr

    # {{{ find source and target points

    source_angles = (
            np.pi/2 + np.linspace(
                0, 2*np.pi * N_ARMS_FOR_BVP_EXPERIMENT,
                N_ARMS_FOR_BVP_EXPERIMENT, endpoint=False)
            ) / N_ARMS_FOR_BVP_EXPERIMENT
    source_points = 0.75 * np.array([
        np.cos(source_angles),
        np.sin(source_angles),
        ])
    target_angles = (
            np.pi + np.pi/2 + np.linspace(
                0, 2*np.pi * N_ARMS_FOR_BVP_EXPERIMENT,
                N_ARMS_FOR_BVP_EXPERIMENT, endpoint=False)
            ) / N_ARMS_FOR_BVP_EXPERIMENT
    target_points = 1.5 * np.array([
        np.cos(target_angles),
        np.sin(target_angles),
        ])

    if 0:
        from meshmode.discretization.visualization import draw_curve
        draw_curve(density_discr)
        import matplotlib.pyplot as plt
        plt.plot(source_points[0], source_points[1], "go")
        plt.plot(target_points[0], target_points[1], "ro")
        plt.show()

    np.random.seed(17)
    source_charges = np.random.randn(N_ARMS_FOR_BVP_EXPERIMENT)

    source_points_dev = cl.array.to_device(queue, source_points)
    target_points_dev = cl.array.to_device(queue, target_points)
    source_charges_dev = cl.array.to_device(queue, source_charges)

    from pytential.source import PointPotentialSource
    from pytential.target import PointsTarget

    point_source = PointPotentialSource(
            lpot_source.cl_context, source_points_dev)

    pot_src = sym.IntG(
        # FIXME: qbx_forced_limit--really?
        k_sym, sym.var("charges"), qbx_forced_limit=None, **knl_kwargs)

    ref_direct = bind(
            (point_source, PointsTarget(target_points_dev)), pot_src)(
            queue, charges=source_charges_dev, **knl_kwargs).get()

    sym_sqrt_j = sym.sqrt_jac_q_weight(density_discr.ambient_dim)

    bc = bind(
            (point_source, density_discr),
            sym.normal_derivative(
                density_discr.ambient_dim, pot_src, where=sym.DEFAULT_TARGET)
            )(queue, charges=source_charges_dev, **knl_kwargs)

    rhs = bind(density_discr, sym.var("bc")*sym_sqrt_j)(queue, bc=bc)

    # }}}

    # {{{ solve

    bound_op = bind(
            lpot_source,
            -0.5*sym.var("u")
            + sym_sqrt_j*sym.Sp(
                k_sym, sym.var("u")/sym_sqrt_j, qbx_forced_limit="avg",
                **knl_kwargs))

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "u", np.float64, **knl_kwargs),
            rhs,
            tol=1e-10,
            stall_iterations=100,
            progress=True,
            hard_failure=True)

    print("gmres state:", gmres_result.state)
    u = gmres_result.solution

    # }}}

    points_target = PointsTarget(target_points_dev)
    bound_tgt_op = bind(
            (lpot_source, points_target),
            sym.S(k_sym, sym.var("u")/sym_sqrt_j, qbx_forced_limit=None))

    test_via_bdry = bound_tgt_op(queue, u=u).get()

    err = ref_direct-test_via_bdry

    err_l2 = la.norm(err, 2) / la.norm(ref_direct, 2)
    err_linf = la.norm(err, np.inf) / la.norm(ref_direct, np.inf)
    return err_l2, err_linf, gmres_result.iteration_count


def run_bvp_error_experiment(use_gigaqbx_fmm):
    cl_ctx = cl.create_some_context(interactive=False)

    table_linf = []
    table_l2 = []

    qbx_orders = [3, 5, 7, 9]
    from pytools import flatten
    headers = list(flatten(
        [[r"{$(1/2)^{\pfmm+1}$}", r"{$\#it$}"]]
        + [[r"{$\pqbx=%d$}" % p, "\#it"] for p in qbx_orders]))

    column_formats = "".join([
        "S[table-format = 1e-1, round-precision = 0]",
        "c"] + ["S", "c"] * len(qbx_orders))

    k = 0

    for fmm_order in [3, 5, 10, 15, 20]:
        errs_l2 = []
        errs_linf = []
        num_iterations = []

        for ip, p in enumerate(qbx_orders):
            if fmm_order is False:
                fmm_error = "{0}"
                # The number of particles is impractical for direct eval.
                # GIGAQBX + fmmlib appears to be the most reliable fast
                # eval option.
                lpot_source = get_geometry(cl_ctx,
                                           N_ARMS_FOR_BVP_EXPERIMENT,
                                           use_gigaqbx_fmm=True,
                                           fmm_backend="fmmlib")
            else:
                fmm_error = 2 ** -(fmm_order + 1)
                lpot_source = get_geometry(cl_ctx,
                                           N_ARMS_FOR_BVP_EXPERIMENT,
                                           use_gigaqbx_fmm=use_gigaqbx_fmm)

                lpot_source = lpot_source.copy(fmm_backend="sumpy")

            print("#" * 80)
            print("fmm_order %s, qbx_order %d" % (fmm_order, p))
            print("#" * 80)
            true_fmm_order = 30 if fmm_order is False else fmm_order
            err_l2, err_linf, num_it = get_bvp_error(
                    lpot_source, true_fmm_order, p, k=k)

            print("errors: l^2: %g - l^inf: %g" % (err_l2, err_linf))

            errs_l2.append(err_l2)
            errs_linf.append(err_linf)
            num_iterations.append(num_it)

        row_l2 = [fmt(fmm_error)]
        row_l2.append(str(fmm_order) if fmm_order is not False else "(direct)")
        for e, num_it in zip(errs_l2, num_iterations):
            row_l2.append(fmt(e))
            row_l2.append(str(num_it))

        table_l2.append(row_l2)

        row_linf = [fmt(fmm_error)]
        row_linf.append(
                str(fmm_order) if fmm_order is not False else "(direct)")
        for e, num_it in zip(errs_linf, num_iterations):
            row_linf.append(fmt(e))
            row_linf.append(str(num_it))

        table_linf.append(row_linf)

    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    print_table(table_l2, headers, f"bvp-error-l2-{scheme_name}.tex",
                column_formats)
    print_table(table_linf, headers, f"bvp-error-linf-{scheme_name}.tex",
                column_formats)

# }}}


# {{{ wall time experiment

def run_wall_time_experiment(use_gigaqbx_fmm):
    cl_ctx = cl.create_some_context(interactive=False)

    if use_gigaqbx_fmm:
        qbx_orders = [3, 7]
        fmm_orders = [10, 15]
    else:
        qbx_orders = [3, 7]
        fmm_orders = [15, 30]

    times_by_arms = {}

    for n_arms in ARMS:
        lpot_source = get_geometry(cl_ctx, n_arms, use_gigaqbx_fmm)

        times = []
        for fmm_order, qbx_order in zip(fmm_orders, qbx_orders):
            times.append(get_slp_wall_time(lpot_source, fmm_order, qbx_order))

        times_by_arms[n_arms] = times

    headers = ["n"] + [
            f"Time: $\pqbx = {qbx_order}$, $pfmm = {fmm_order}$"
            for qbx_order, fmm_order in zip(qbx_orders, fmm_orders)]

    table = []

    for n_arms in ARMS:
        table.append([str(n_arms)] +
                     [str(t) for t in times_by_arms[n_arms]])

    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"
    print_table(table, headers, f"slp-wall-times-{scheme_name}.tex")

# }}}


# {{{ particle distributions experiment

NeighborhoodCounter = AreaQueryElementwiseTemplate(
    extra_args=r"""
    /* input */
    particle_id_t *box_source_starts,
    particle_id_t *box_source_counts,
    coord_t *search_radii,

    /* output */
    int *sizes,

    /* input, dim-dependent length */
    %for ax in AXIS_NAMES[:dimensions]:
        coord_t *sources_${ax},
    %endfor

    %for ax in AXIS_NAMES[:dimensions]:
        coord_t *centers_${ax},
    %endfor
    """,
    ball_center_and_radius_expr=r"""
    %for ax in AXIS_NAMES[:dimensions]:
        ${ball_center}.${ax} = centers_${ax}[i];
    %endfor
    ${ball_radius} = search_radii[i];
    """,
    leaf_found_op=r"""
    for (int j = 0; j < box_source_counts[${leaf_box_id}]; ++j)
    {
        particle_id_t source_idx = box_source_starts[${leaf_box_id}] + j;

        coord_t dist = 0;
        %for ax in AXIS_NAMES[:dimensions]:
            dist = fmax(dist, sources_${ax}[source_idx] - centers_${ax}[i]);
        %endfor

        if (dist > search_radii[i]) {
            continue;
        }

        ++sizes[i];
    }
    """,
    name="count_neighborhood_size")


def get_qbx_center_neighborhood_sizes(lpot_source, radius):
    queue = cl.CommandQueue(lpot_source.cl_context)

    def inspect_geo_data(insn, bound_expr, geo_data):
        nonlocal sizes, nsources, ncenters
        tree = geo_data.tree().with_queue(queue)

        from boxtree.area_query import PeerListFinder
        plf = PeerListFinder(queue.context)
        pl, evt = plf(queue, tree)

        # Perform an area query around each QBX center, counting the
        # neighborhood sizes.
        knl = NeighborhoodCounter.generate(
            queue.context,
            tree.dimensions,
            tree.coord_dtype,
            tree.box_id_dtype,
            tree.box_id_dtype,
            tree.nlevels,
            extra_type_aliases=(('particle_id_t', tree.particle_id_dtype),))

        centers = geo_data.centers()
        search_radii = radius * geo_data.expansion_radii().with_queue(queue)

        ncenters = len(search_radii)
        nsources = tree.nsources
        sizes = cl.array.zeros(queue, ncenters, np.int32)

        assert nsources == lpot_source.quad_stage2_density_discr.nnodes

        coords = []
        coords.extend(tree.sources)
        coords.extend(centers)

        evt = knl(
            *NeighborhoodCounter.unwrap_args(
                tree,
                pl,
                tree.box_source_starts,
                tree.box_source_counts_cumul,
                search_radii,
                sizes,
                *coords),
            range=slice(ncenters),
            queue=queue,
            wait_for=[evt])

        cl.wait_for_events([evt])

        return False  # no need to do the actual FMM

    sizes = None
    nsources = None
    ncenters = None

    lpot_source = lpot_source.copy(geometry_data_inspector=inspect_geo_data)
    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    # The kernel doesn't really matter here
    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var('sigma')
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    bound_op = bind(lpot_source, sym_op)
    bound_op(queue, sigma=sigma)

    return (sizes.get(), nsources, ncenters)


def run_particle_distributions_experiment():
    cl_ctx = cl.create_some_context(interactive=False)

    PERCENTILES = [20, 40, 60, 80, 100]  # noqa: N806

    headers = [r"$n$", r"N_S", r"$M_C$"] + [str(i) for i in PERCENTILES]
    table = []

    for n_arms in ARMS:
        row = []
        lpot_source = get_geometry(cl_ctx, n_arms, True)
        sizes, ns, nc = get_qbx_center_neighborhood_sizes(
                lpot_source, radius=8/TCF)
        row.append(str(n_arms))
        row.append(str(ns))
        row.append("%.1f" % np.mean(sizes))
        for percentile in PERCENTILES:
            row.append("%.1f" % np.percentile(sizes, percentile))
        table.append(row)

    print_table(table, headers, "particle-distributions.tex")

# }}}


# {{{ complexity experiment

def get_fmm_cost(lpot_source):
    queue = cl.CommandQueue(lpot_source.cl_context)

    costs = None
    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    # The kernel doesn't really matter here
    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var('sigma')
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    bound_op = bind(lpot_source, sym_op)

    perf = bound_op.get_modeled_cost(queue, sigma=sigma)
    from pytools import one
    perf_result = one(perf.values())

    return perf_result


def run_complexity_experiment(use_gigaqbx_fmm, fmm_backend=None,
                              compute_wall_times=True):
    cl_ctx = cl.create_some_context(interactive=False)

    if use_gigaqbx_fmm:
        qbx_orders = [3, 7]
        fmm_orders = [10, 15]

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

    else:
        qbx_orders = [3, 7]
        fmm_orders = [15, 30]

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

    perf_line_styles = ("x-", "+-", ".-", "*-", "s-", "d-")

    def initialize_axis(ax, title, xlabel=None, ylabel=None):
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontdict={"size": FONTSIZE})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontdict={"size": FONTSIZE})
        ax.grid("on")
        ax.set_xlim(1e4, 2*1e6)
        ax.set_ylim(1e5, 2e9)

    # {{{ load / generate data

    import pickle

    try:
        with open("costs-%s.pkl" % use_gigaqbx_fmm, "rb") as inf:
            costs_by_arms = pickle.load(inf)
        if compute_wall_times:
            with open("errors-%s.pkl" % use_gigaqbx_fmm, "rb") as inf:
                errors_by_arms = pickle.load(inf)
            with open("times-%s.pkl" % use_gigaqbx_fmm, "rb") as inf:
                times_by_arms = pickle.load(inf)
        else:
            errors_by_arms = dict()
            times_by_arms = dict()
        print("Note: Generating figures from pickled results.")
    except Exception:
        costs_by_arms = {}
        errors_by_arms = {}
        times_by_arms = {}

        # Generate costs, errors.
        for n_arms in ARMS:
            lpot_source = get_geometry(cl_ctx, n_arms,
                                       use_gigaqbx_fmm,
                                       fmm_backend=fmm_backend)
            costs_by_arms[n_arms] = get_fmm_cost(lpot_source)
            errors = []
            times = []

            if compute_wall_times:
                for fmm_order, qbx_order in zip(fmm_orders, qbx_orders):
                    time = np.empty(1,)
                    errors.append(get_green_error(lpot_source, fmm_order,
                                                  qbx_order,
                                                  time=time)[1])
                    times.append(time[0])

                errors_by_arms[n_arms] = tuple(errors)
                times_by_arms[n_arms] = tuple(times)

        with open("costs-%s.pkl" % use_gigaqbx_fmm, "wb") as outf:
            pickle.dump(costs_by_arms, outf)

        if compute_wall_times:
            with open("errors-%s.pkl" % use_gigaqbx_fmm, "wb") as outf:
                pickle.dump(errors_by_arms, outf)

            with open("times-%s.pkl" % use_gigaqbx_fmm, "wb") as outf:
                pickle.dump(times_by_arms, outf)

    # }}}

    # {{{ gen figure

    fig, axes = plt.subplots(1, len(qbx_orders))
    fig.set_size_inches(6.5, 2)

    plot_options = dict(linewidth=1, markersize=4)

    for iax, (ax, qbx_order, fmm_order) in enumerate(
            zip(axes, qbx_orders, fmm_orders)):
        if iax == 0:
            ylabel = r"Cost $\sim$ Number of Flops"
        else:
            ylabel = None
        title = (
                f"$p_\\textrm{{QBX}}={qbx_order}$, "
                + f"$p_\\textrm{{FMM}}={fmm_order}$"
                if not GENERATE_PDF
                else f"QBX order {qbx_order}, FMM order {fmm_order}")
        initialize_axis(ax, title, xlabel="Number of Particles", ylabel=ylabel)

    # Generate results.
    for qbx_order, fmm_order, ax in zip(qbx_orders, fmm_orders, axes):
        xs = []
        nlevels_max = 0
        for arm in ARMS:
            xs.append(costs_by_arms[arm].params["nparticles"])
            nlevels_max = max(nlevels_max, costs_by_arms[arm].params["nlevels"])

        ev_dict = {"p_qbx": qbx_order}
        for i in range(nlevels_max):
            ev_dict["p_fmm_lev%d" % i] = fmm_order

        evaluated_costs_by_arms = {}

        for n_arms in ARMS:
            evaluated_costs_by_arms[n_arms] = (
                    costs_by_arms[n_arms]
                    .with_params(ev_dict)
                    .get_predicted_times(merge_close_lists=False))

        labels = []
        for feature, label, style in zip(
                perf_features, perf_labels, perf_line_styles):
            costs = [evaluated_costs_by_arms[n_arms][feature] for n_arms in ARMS]
            if feature != "form_global_qbx_locals_list1":
                l, = ax.loglog(xs, costs, style, label=label, **plot_options)
            else:
                l, = ax.loglog(xs, costs, style, label=label, zorder=100, **plot_options)
            labels.append(l)

        # summary
        l, = ax.loglog(xs, [
                sum(evaluated_costs_by_arms[n_arms].values())
                for n_arms in ARMS], ".--", label="all")

        labels.append(l)

    fig.legend(labels, perf_labels + ("all",), loc="center right",
               prop={"size": FONTSIZE})
    fig.subplots_adjust(right=0.85)

    suffix = "pdf" if GENERATE_PDF else "pgf"
    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    fig.savefig(f"complexity-{scheme_name}.{suffix}", bbox_inches="tight")

    # {{{ op counts

    """ FIXME
    headers = ["n"] + [
            f"Op count: $\pqbx = {qbx_order}$, $pfmm = {fmm_order}$"
            for qbx_order, fmm_order in zip(qbx_orders, fmm_orders)]

    table = []

    for n_arms in ARMS:
        row = [str(n_arms)]
        for qbx_order, fmm_order in zip(qbx_orders, fmm_orders):
            def ev(expr):
                from pymbolic import evaluate_kw
                return evaluate_kw(expr, p_fmm=fmm_order, p_qbx=qbx_order)
            row.append(str(
                sum(ev(val) for val in costs_by_arms[n_arms].values())))
        table.append(row)

    print_table(table, headers, f"complexity-op-counts-{scheme_name}.tex")
    """

    # }}}

    # {{{ particle counts

    headers = ["n", "Particles"]

    table = []

    for n_arms in ARMS:
        table.append([str(n_arms),
                      str(costs_by_arms[n_arms].params["nparticles"])])

    print_table(table, headers, f"complexity-nparticles-{scheme_name}.tex")

    # }}}

    if compute_wall_times:

        headers = ["n"] + [
                f"Error: $\pqbx = {qbx_order}$, $pfmm = {fmm_order}$"
                for qbx_order, fmm_order in zip(qbx_orders, fmm_orders)]

        table = []

        for n_arms in ARMS:
            table.append([str(n_arms)] +
                         [fmt(e) for e in errors_by_arms[n_arms]])

        means = [
            np.mean([errors_by_arms[n_arms][i] for n_arms in ARMS])
            for i in range(len(qbx_orders))]

        table.append([str("(mean)")] + [fmt(e) for e in means])

        print_table(table, headers, f"complexity-errors-{scheme_name}.tex")

        headers = ["n"] + [
                f"Time: $\pqbx = {qbx_order}$, $pfmm = {fmm_order}$"
                for qbx_order, fmm_order in zip(qbx_orders, fmm_orders)]

        table = []

        for n_arms in ARMS:
            table.append([str(n_arms)] +
                         [str(t) for t in times_by_arms[n_arms]])

        print_table(table, headers, f"complexity-times-{scheme_name}.tex")

    # }}}


def run_from_sep_smaller_threshold_complexity_experiment():
    cl_ctx = cl.create_some_context(interactive=False)

    qbx_orders = [3, 7]
    fmm_orders = [10, 15]
    from_sep_smaller_threshold_values = [0, 15]

    headers = ["n arms"] + [f"pqbx={q}, pfmm={p}" for q, p in
                            zip(qbx_orders, fmm_orders)]

    for threshold in from_sep_smaller_threshold_values:
        table = []
        for n_arms in ARMS:
            row = [f"{n_arms}"]

            lpot_source = get_geometry(
                cl_ctx, n_arms,
                use_gigaqbx_fmm=True,
                from_sep_smaller_threshold=threshold)
            cost = get_fmm_cost(lpot_source)

            for fmm_order, qbx_order in zip(fmm_orders, qbx_orders):
                def ev(expr):
                    from pymbolic import evaluate_kw
                    return evaluate_kw(expr, p_fmm=fmm_order, p_qbx=qbx_order)
                row.append(str(sum(ev(feature) for feature in cost.values())))

            table.append(row)

        print_table(table, headers,
                    f"total-cost-from-sep-smaller-threshold-{threshold}.tex")

# }}}


# {{{ level restriction

def run_level_restriction_experiment():
    """This verifies that, if the tree is level restricted, then List 3 close and
    List 3 far of a leaf box contain boxes at most 2 levels lower than the box.
    """

    def inspect_geo_data(insn, bound_expr, geo_data):
        with cl.CommandQueue(cl_ctx) as queue:
            trav = geo_data.traversal(merge_close_lists=False).get(queue)
            tree = geo_data.tree().get(queue)

        from boxtree.tree import box_flags_enum

        logger.info("checking list 3 far")

        # Check List 3 far
        for src_level in range(tree.nlevels):
            for box in (
                    trav.target_boxes_sep_smaller_by_source_level[src_level]):
                assert 0 <= box < tree.nboxes
                level = tree.box_levels[box]

                if tree.box_flags[box] & box_flags_enum.HAS_CHILDREN:
                    # Target must be a leaf
                    continue

                assert level < src_level <= level + 2, (level, src_level)

        logger.info("done checking list 3 far")

        logger.info("checking list 3 close")

        # Check List 3 close
        for ibox, box in enumerate(trav.target_boxes):
            start, end = trav.from_sep_close_smaller_starts[ibox:ibox+2]
            level = tree.box_levels[box]

            if tree.box_flags[box] & box_flags_enum.HAS_CHILDREN:
                # Target must be a leaf
                continue

            for src_box in trav.from_sep_close_smaller_lists[start:end]:
                assert 0 <= box < tree.nboxes
                assert tree.box_flags[src_box] & ~box_flags_enum.HAS_CHILDREN
                assert level < tree.box_levels[src_box] <= level + 2

        logger.info("done checking list 3 close")

        print("Verified List 3 close and far.")

    cl_ctx = cl._csc(0)
    queue = cl.CommandQueue(cl_ctx)
    n_arms = 65

    lpot_source = get_geometry(cl_ctx, n_arms, True,
                               from_sep_smaller_threshold=0)
    lpot_source = lpot_source.copy(
        geometry_data_inspector=inspect_geo_data,
        _tree_kind="adaptive-level-restricted")
    density_discr = lpot_source.density_discr

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    # The kernel doesn't really matter here
    from sumpy.kernel import LaplaceKernel
    sigma_sym = sym.var('sigma')
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    bound_op = bind(lpot_source, sym_op)
    bound_op(queue, sigma=sigma)

# }}}


def main():
    import os
    os.nice(1)
    
    multiprocessing.set_start_method("spawn")

    # Uncomment everything to gather all experiment data for the paper.
    run_green_error_experiment(use_gigaqbx_fmm=True)
    run_green_error_experiment(use_gigaqbx_fmm=False)
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
