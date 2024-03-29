#!/usr/bin/env python3
"""This script generates experimental data."""

import csv
import collections
import logging
import multiprocessing
import os

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import utils

from functools import partial
from itertools import product
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, NArmedStarfish, drop, n_gon, qbx_peanut,
        WobblyCircle, make_curve_mesh, starfish)
from pytential import bind, sym, norm


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


OUTPUT_DIR = "raw-data"


def make_output_file(filename, **flags):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **flags)


# {{{ global params

TARGET_ORDER = 8
OVSMP_FACTOR = 4
PANELS_PER_ARM = 50
ARMS = list(range(5, 70, 10))
TCF = 0.9
DEFAULT_GIGAQBX_BACKEND = "fmmlib"
DEFAULT_QBXFMM_BACKEND = "fmmlib"
TIMING_ROUNDS = 3
POOL_WORKERS = multiprocessing.cpu_count() + 1
FLOAT_OUTPUT_FMT = "%.17e"

# }}}


GreenErrorParams = collections.namedtuple(
        "GreenErrorParams",
        "n_arms, fmm_order, qbx_order")


# {{{ green error experiment params

GREEN_ERROR_EXPERIMENT_N_ARMS = 65
GREEN_ERROR_EXPERIMENT_FMM_ORDERS = ["inf", 3, 5, 10, 15, 20]
GREEN_ERROR_EXPERIMENT_QBX_ORDERS = [3, 5, 7, 9]
GREEN_ERROR_EXPERIMENT_PARAMS = [
        GreenErrorParams(GREEN_ERROR_EXPERIMENT_N_ARMS, *p)
        for p in product(
            GREEN_ERROR_EXPERIMENT_FMM_ORDERS,
            GREEN_ERROR_EXPERIMENT_QBX_ORDERS)]

# }}}


# {{{ bvp experiment params

BVP_EXPERIMENT_N_ARMS = 25
BVP_EXPERIMENT_FMM_ORDERS = [3, 5, 10, 15, 20]
BVP_EXPERIMENT_QBX_ORDERS = [3, 5, 7, 9]

BVP_EXPERIMENT_GREEN_ERROR_PARAMS = [
        GreenErrorParams(BVP_EXPERIMENT_N_ARMS, *p)
        for p in product(BVP_EXPERIMENT_FMM_ORDERS, BVP_EXPERIMENT_QBX_ORDERS)]

# }}}


# {{{ particle distribution experiment params

PARTICLE_DISTRIBUTION_EXPERIMENT_PERCENTILES = [20, 40, 60, 80, 100]
PARTICLE_DISTRIBUTION_EXPERIMENT_N_ARMS_LIST = ARMS

# }}}


# {{{ complexity experiment params

COMPLEXITY_EXPERIMENT_N_ARMS_LIST = ARMS

COMPLEXITY_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS = [(10, 3), (15, 7)]
COMPLEXITY_EXPERIMENT_GIGAQBX_GREEN_ERROR_PARAMS = [
        GreenErrorParams(n_arms, fmm_order, qbx_order)
        for n_arms in COMPLEXITY_EXPERIMENT_N_ARMS_LIST
        for fmm_order, qbx_order
        in COMPLEXITY_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS]

COMPLEXITY_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS = [(15, 3), (30, 7)]
COMPLEXITY_EXPERIMENT_QBXFMM_GREEN_ERROR_PARAMS = [
        GreenErrorParams(n_arms, fmm_order, qbx_order)
        for n_arms in COMPLEXITY_EXPERIMENT_N_ARMS_LIST
        for fmm_order, qbx_order
        in COMPLEXITY_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS]

# }}}


# {{{ wall time experiment params

WALL_TIME_EXPERIMENT_TIMING_ROUNDS = 3
WALL_TIME_EXPERIMENT_N_ARMS_LIST = ARMS
WALL_TIME_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS = \
        COMPLEXITY_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS
WALL_TIME_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS = \
        COMPLEXITY_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS

# }}}


# {{{ general utils

def get_geometry(ctx, n_arms, use_gigaqbx_fmm, fmm_backend=None,
                 from_sep_smaller_threshold=None):
    """Return a QBXLayerPotentialSource representing an n-armed starfish
    source geometry."""

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
            _from_sep_smaller_crit="static_linf",
            _box_extent_norm="linf",
            )

    lpot_source, _ = lpot_source.with_refinement(**refiner_extra_kwargs)

    return lpot_source

# }}}


# {{{ green error experiment

def get_green_error(lpot_source, fmm_order, qbx_order, k=0, time=None):
    queue = cl.CommandQueue(lpot_source.cl_context)

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

# }}}


# {{{ green error experiment

GREEN_ERROR_FIELDS = ("n_arms", "fmm_order", "qbx_order", "err_l2", "err_linf")


def run_green_error_experiment(use_gigaqbx_fmm, params, label):
    with multiprocessing.Pool(POOL_WORKERS) as pool:
        results = pool.map(
                partial(_green_error_experiment_body, use_gigaqbx_fmm),
                params)

    output_path = "%s-results-%s.csv" % (
            label, "gigaqbx" if use_gigaqbx_fmm else "qbxfmm")

    with make_output_file(output_path, newline="") as outfile:
        writer = csv.DictWriter(outfile, GREEN_ERROR_FIELDS)
        writer.writeheader()
        for param, result in zip(params, results):
            row = dict(
                    n_arms=param.n_arms,
                    fmm_order=param.fmm_order,
                    qbx_order=param.qbx_order,
                    err_l2=FLOAT_OUTPUT_FMT % result[0],
                    err_linf=FLOAT_OUTPUT_FMT % result[1])
            writer.writerow(row)


def _green_error_experiment_body(use_gigaqbx_fmm, params):
    # This returns a tuple (err_l2, err_linf).
    cl_ctx = cl.create_some_context(interactive=False)

    fmm_order = params.fmm_order
    qbx_order = params.qbx_order

    if fmm_order == "inf":
        # The number of particles is impractical for direct eval.
        # GIGAQBX + fmmlib appears to be the most reliable fast
        # eval option.
        lpot_source = get_geometry(cl_ctx,
                                   params.n_arms,
                                   use_gigaqbx_fmm=True,
                                   fmm_backend="fmmlib")
    else:
        lpot_source = get_geometry(cl_ctx,
                                   params.n_arms,
                                   use_gigaqbx_fmm=use_gigaqbx_fmm)

    print("#" * 80)
    print("fmm_order %s, qbx_order %d" % (fmm_order, qbx_order))
    print("#" * 80)

    true_fmm_order = 30 if fmm_order == "inf" else fmm_order
    return get_green_error(lpot_source, true_fmm_order, qbx_order, k=0)

# }}}


# {{{ bvp experiment

def get_bvp_error(lpot_source, fmm_order, qbx_order, k=0):
    # This returns a tuple (err_l2, err_linf, nit).
    queue = cl.CommandQueue(lpot_source.cl_context)

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
                0, 2*np.pi * BVP_EXPERIMENT_N_ARMS,
                BVP_EXPERIMENT_N_ARMS, endpoint=False)
            ) / BVP_EXPERIMENT_N_ARMS
    source_points = 0.75 * np.array([
        np.cos(source_angles),
        np.sin(source_angles),
        ])
    target_angles = (
            np.pi + np.pi/2 + np.linspace(
                0, 2*np.pi * BVP_EXPERIMENT_N_ARMS,
                BVP_EXPERIMENT_N_ARMS, endpoint=False)
            ) / BVP_EXPERIMENT_N_ARMS
    target_points = 1.5 * np.array([
        np.cos(target_angles),
        np.sin(target_angles),
        ])

    np.random.seed(17)
    source_charges = np.random.randn(BVP_EXPERIMENT_N_ARMS)

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


BVP_FIELDS = ("fmm_order", "qbx_order", "err_l2", "err_linf", "gmres_nit")


def run_bvp_experiment():
    fmm_orders = BVP_EXPERIMENT_FMM_ORDERS
    qbx_orders = BVP_EXPERIMENT_QBX_ORDERS

    order_pairs = list(product(fmm_orders, qbx_orders))

    with multiprocessing.Pool(POOL_WORKERS) as pool:
        results = pool.map(
                _bvp_experiment_body,
                order_pairs)

    output_path = "bvp-results.csv"

    with make_output_file(output_path, newline="") as outfile:
        writer = csv.DictWriter(outfile, BVP_FIELDS)
        writer.writeheader()
        for order_pair, result in zip(order_pairs, results):
            row = dict(
                    fmm_order=order_pair[0],
                    qbx_order=order_pair[1],
                    err_l2=FLOAT_OUTPUT_FMT % result[0],
                    err_linf=FLOAT_OUTPUT_FMT % result[1],
                    gmres_nit=result[2])
            writer.writerow(row)


def _bvp_experiment_body(fmm_and_qbx_order_pair):
    # This returns a tuple (err_l2, err_linf, gmres_nit).
    cl_ctx = cl.create_some_context(interactive=False)

    fmm_order, qbx_order = fmm_and_qbx_order_pair
    lpot_source = (
            get_geometry(cl_ctx, BVP_EXPERIMENT_N_ARMS, use_gigaqbx_fmm=True))

    print("#" * 80)
    print("BVP fmm_order %d, qbx_order %d" % (fmm_order, qbx_order))
    print("#" * 80)

    return get_bvp_error(lpot_source, fmm_order, qbx_order, k=0)

# }}}


# {{{ particle distributions

PARTICLE_DISTRIBUTION_FIELDS = (
        ("n_arms", "nsources", "avg")
        + tuple(
            "percentile_%d" % pct
            for pct in PARTICLE_DISTRIBUTION_EXPERIMENT_PERCENTILES))


def run_particle_distributions_experiment():
    cl_ctx = cl.create_some_context(interactive=False)

    rows = []

    for n_arms in PARTICLE_DISTRIBUTION_EXPERIMENT_N_ARMS_LIST:
        lpot_source = get_geometry(cl_ctx, n_arms, use_gigaqbx_fmm=True)
        neighborhood_sizes, nsources, _ = (
                utils.get_qbx_center_neighborhood_sizes(lpot_source, 8 / TCF))

        row = dict(
                n_arms=n_arms,
                nsources=nsources,
                avg=FLOAT_OUTPUT_FMT % np.mean(neighborhood_sizes))

        percentiles = np.percentile(
                neighborhood_sizes,
                PARTICLE_DISTRIBUTION_EXPERIMENT_PERCENTILES)

        for pct, val in zip(
                PARTICLE_DISTRIBUTION_EXPERIMENT_PERCENTILES,
                percentiles):
            row["percentile_%d" % pct] = FLOAT_OUTPUT_FMT % val

        rows.append(row)

    with make_output_file("particle-distributions.csv", newline="") as outfile:
        writer = csv.DictWriter(outfile, PARTICLE_DISTRIBUTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# }}}


# {{{ fmm cost

class GigaQBXPaperTranslationCostModel(object):

    def __init__(self, dim, nlevels):
        from pymbolic import var
        p_qbx = var("p_qbx")
        p_fmm_by_level = np.array(
                [var("p_fmm_lev%d" % i) for i in range(nlevels)])

        # Note: This means order n is modeled as having n coeffs in 2D, instead
        # of (n + 1) coeffs.
        # This is consistent with the model in the 2D paper.
        self.ncoeffs_fmm_by_level = p_fmm_by_level ** (dim - 1)
        self.ncoeffs_qbx = p_qbx ** (dim - 1)

    @staticmethod
    def direct():
        return 1

    def p2qbxl(self):
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


CONSTANT_ONE_PARAMS = dict(
        c_l2l=1,
        c_l2p=1,
        c_l2qbxl=1,
        c_m2l=1,
        c_m2m=1,
        c_m2p=1,
        c_m2qbxl=1,
        c_p2l=1,
        c_p2m=1,
        c_p2p=1,
        c_p2qbxl=1,
        c_qbxl2p=1,
        )


def get_fmm_cost(lpot_source):
    """Return the modeled cost of on-surface evaluation for the single-layer
    potential."""

    from sumpy.kernel import LaplaceKernel
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    sym_op_S = sym.S(k_sym, sym.var("sigma"), qbx_forced_limit=+1)

    inspect_geo_data_result = []

    def inspect_geo_data(insn, bound_expr, geo_data):
        del bound_expr

        from pytential.qbx.cost import CostModel
        cost_model = CostModel(
                translation_cost_model_factory=(
                    GigaQBXPaperTranslationCostModel),
                calibration_params=CONSTANT_ONE_PARAMS)

        kernel = lpot_source.get_fmm_kernel(insn.kernels)
        kernel_arguments = insn.kernel_arguments

        result = cost_model(geo_data, kernel, kernel_arguments)
        inspect_geo_data_result.append(result)

        return False

    lpot_source = lpot_source.copy(geometry_data_inspector=inspect_geo_data)
    op_S = bind(lpot_source, sym_op_S)

    with cl.CommandQueue(lpot_source.cl_context) as queue:
        density_discr = lpot_source.density_discr
        nodes = density_discr.nodes().with_queue(queue)
        sigma = cl.clmath.sin(10 * nodes[0])
        op_S(queue, sigma=sigma)

    return inspect_geo_data_result[0]


STAGES = (
        "form_multipoles",
        "coarsen_multipoles",
        "eval_direct_list1",
        "eval_direct_list3",
        "eval_direct_list4",
        "multipole_to_local",
        "eval_multipoles",
        "form_locals",
        "refine_locals",
        "eval_locals",
        "form_global_qbx_locals_list1",
        "form_global_qbx_locals_list3",
        "form_global_qbx_locals_list4",
        "translate_box_multipoles_to_qbx_local",
        "translate_box_local_to_qbx_local",
        "eval_qbx_expansions")


COMPLEXITY_FIELDS = ("n_arms", "nparticles", "fmm_order", "qbx_order") + STAGES


def run_complexity_experiment(use_gigaqbx_fmm):
    _get_complexity_experiment_results(
            use_gigaqbx_fmm, from_sep_smaller_threshold=15)


def _get_complexity_experiment_results(
        use_gigaqbx_fmm, from_sep_smaller_threshold):
    with multiprocessing.Pool(POOL_WORKERS) as pool:
        results = pool.map(
                partial(
                        _complexity_experiment_body,
                        use_gigaqbx_fmm,
                        from_sep_smaller_threshold),
                COMPLEXITY_EXPERIMENT_N_ARMS_LIST)

    fmm_and_qbx_order_pairs = (
            COMPLEXITY_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS
            if use_gigaqbx_fmm
            else COMPLEXITY_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS)

    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    outfile = make_output_file(
            f"complexity-results-{scheme_name}"
            f"-threshold{from_sep_smaller_threshold}.csv",
            newline="")

    writer = csv.DictWriter(outfile, COMPLEXITY_FIELDS)
    writer.writeheader()

    for fmm_order, qbx_order in fmm_and_qbx_order_pairs:
        for n_arms, result in zip(COMPLEXITY_EXPERIMENT_N_ARMS_LIST, results):
            # Adjust QBX and FMM order params.
            new_params = result.params.copy()
            new_params["p_qbx"] = qbx_order
            for lev in range(new_params["nlevels"]):
                new_params["p_fmm_lev%d" % lev] = fmm_order

            costs = result.with_params(new_params).get_predicted_times()
            row = {}
            row["n_arms"] = n_arms
            row["nparticles"] = new_params["nsources"] + new_params["ntargets"]
            row["fmm_order"] = fmm_order
            row["qbx_order"] = qbx_order
            for stage in STAGES:
                row[stage] = costs[stage]
            writer.writerow(row)

    outfile.close()


def _complexity_experiment_body(
        use_gigaqbx_fmm, from_sep_smaller_threshold, n_arms):
    cl_ctx = cl.create_some_context(interactive=False)
    lpot_source = get_geometry(
            cl_ctx,
            n_arms,
            use_gigaqbx_fmm=use_gigaqbx_fmm,
            from_sep_smaller_threshold=from_sep_smaller_threshold)
    return get_fmm_cost(lpot_source)

# }}}


# {{{ from_sep_smaller_threshold experiment

def run_from_sep_smaller_threshold_complexity_experiment():
    _get_complexity_experiment_results(
            use_gigaqbx_fmm=True, from_sep_smaller_threshold=0)

# }}}


# {{{ wall time experiment

WALL_TIME_EXPERIMENT_FIELDS = ("n_arms", "fmm_order", "qbx_order", "time")


def run_wall_time_experiment(use_gigaqbx_fmm):
    cl_ctx = cl.create_some_context(interactive=False)

    if use_gigaqbx_fmm:
        fmm_and_qbx_orders = \
                WALL_TIME_EXPERIMENT_GIGAQBX_FMM_AND_QBX_ORDER_PAIRS
    else:
        fmm_and_qbx_orders = \
                WALL_TIME_EXPERIMENT_QBXFMM_FMM_AND_QBX_ORDER_PAIRS

    scheme_name = "gigaqbx" if use_gigaqbx_fmm else "qbxfmm"

    outfile = make_output_file(
            f"wall-time-results-{scheme_name}.csv", newline="")

    writer = csv.DictWriter(outfile, WALL_TIME_EXPERIMENT_FIELDS)
    writer.writeheader()

    for n_arms in WALL_TIME_EXPERIMENT_N_ARMS_LIST:
        lpot_source = get_geometry(cl_ctx, n_arms, use_gigaqbx_fmm)

        for fmm_order, qbx_order in fmm_and_qbx_orders:
            row = dict(n_arms=n_arms, fmm_order=fmm_order, qbx_order=qbx_order)

            wall_times = get_slp_wall_times(lpot_source, fmm_order, qbx_order)
            for wall_time in wall_times:
                row[f"time"] = FLOAT_OUTPUT_FMT % wall_time
                writer.writerow(row)


def get_slp_wall_times(lpot_source, fmm_order, qbx_order):
    queue = cl.CommandQueue(lpot_source.cl_context)

    lpot_source = lpot_source.copy(
            qbx_order=qbx_order,
            fmm_level_to_order=(
                False if fmm_order is False else lambda *args: fmm_order))

    d = lpot_source.ambient_dim

    u_sym = sym.var("u")

    from sumpy.kernel import LaplaceKernel
    S = sym.S(LaplaceKernel(d), u_sym, qbx_forced_limit=-1)
    density_discr = lpot_source.density_discr

    u = cl.array.empty(queue, density_discr.nnodes, np.float64)
    u.fill(1)

    op = bind(lpot_source, S)

    # Warmup
    op(queue, u=u)

    times = []
    from time import perf_counter as curr_time

    for _ in range(WALL_TIME_EXPERIMENT_TIMING_ROUNDS):
        t_start = curr_time()
        op(queue, u=u)
        t_end = curr_time()
        times.append(t_end - t_start)

    return times

# }}}


EXPERIMENTS = (
        "wall-time",
        "green-error",
        "bvp",
        "particle-distributions",
        "complexity",
        "from-sep-smaller-threshold")


def run_experiments(experiments):
    # Wall time experiment
    if "wall-time" in experiments:
        run_wall_time_experiment(use_gigaqbx_fmm=True)
        run_wall_time_experiment(use_gigaqbx_fmm=False)

    # Green error experiment (Tables 3 and 4 in the paper)
    if "green-error" in experiments:
        run_green_error_experiment(use_gigaqbx_fmm=True,
                                   params=GREEN_ERROR_EXPERIMENT_PARAMS,
                                   label="green-error")
        run_green_error_experiment(use_gigaqbx_fmm=False,
                                   params=GREEN_ERROR_EXPERIMENT_PARAMS,
                                   label="green-error")

    # BVP error (Table 5 in the paper)
    if "bvp" in experiments:
        run_green_error_experiment(use_gigaqbx_fmm=True,
                                   params=BVP_EXPERIMENT_GREEN_ERROR_PARAMS,
                                   label="bvp-green-error")
        run_bvp_experiment()

    # Particle distributions (Table 6 in the paper)
    if "particle-distributions" in experiments:
        run_particle_distributions_experiment()

    # Complexity (Figures 14 and 15 in the paper)
    #
    # Also includes the Green errors associated with complexity (reported in
    # Figures 14, 15, and Section 5.2.2 in the paper)
    if "complexity" in experiments:
        run_complexity_experiment(use_gigaqbx_fmm=True)
        run_complexity_experiment(use_gigaqbx_fmm=False)

        run_green_error_experiment(
                use_gigaqbx_fmm=True,
                params=COMPLEXITY_EXPERIMENT_GIGAQBX_GREEN_ERROR_PARAMS,
                label="complexity-green-error")
        run_green_error_experiment(
                use_gigaqbx_fmm=False,
                params=COMPLEXITY_EXPERIMENT_QBXFMM_GREEN_ERROR_PARAMS,
                label="complexity-green-error")

    # From-sep-smaller threshold impact (reported in Section 5.2.2 in the
    # paper)
    if "from-sep-smaller-threshold" in experiments:
        run_from_sep_smaller_threshold_complexity_experiment()


def main():
    description = "This script collects data from one or more experiments."
    experiments = utils.parse_args(description, EXPERIMENTS)
    run_experiments(experiments)


if __name__ == "__main__":
    # Mixing calls to fork() with OpenCL context creation is a bad idea. This
    # call makes it safe to create an OpenCL context in the main
    # thread. (Workers spawned by multiprocessing always use their own
    # separately created contexts.)
    multiprocessing.set_start_method("spawn")
    main()

# vim: foldmethod=marker
