#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

from boxtree.area_query import AreaQueryElementwiseTemplate
from functools import partial
from meshmode.mesh.generation import make_curve_mesh, NArmedStarfish
from pytential import bind, sym, norm
from pytools import memoize

import logging
import multiprocessing

# logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# {{{ neighborhood counter

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
            dist = fmax(dist, fabs(sources_${ax}[source_idx] - centers_${ax}[i]));
        %endfor

        if (dist > search_radii[i]) {
            continue;
        }

        ++sizes[i];
    }
    """,
    name="count_neighborhood_size")

# }}}


# {{{ get qbx center neighborhood sizes

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

    return (sizes.get(queue), nsources, ncenters)

# }}}


# {{{ get qbx center neighborhood sizes, directly evaluated version

def get_qbx_center_neighborhood_sizes_direct(lpot_source, radius):
    queue = cl.CommandQueue(lpot_source.cl_context)

    def inspect_geo_data(insn, bound_expr, geo_data):
        nonlocal sizes, nsources, ncenters
        tree = geo_data.tree().with_queue(queue)

        centers = np.array([axis.get(queue) for axis in geo_data.centers()])
        search_radii = radius * geo_data.expansion_radii().get(queue)
        sources = np.array([axis.get(queue) for axis in tree.sources])

        ncenters = len(search_radii)
        nsources = tree.nsources

        assert nsources == lpot_source.quad_stage2_density_discr.nnodes

        center_to_source_dists = (
                la.norm(
                    (centers[:, np.newaxis, :] - sources[:, :, np.newaxis]).T,
                    ord=np.inf,
                    axis=-1))

        sizes = np.count_nonzero(
                center_to_source_dists <= search_radii[:, np.newaxis],
                axis=1)

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

    return (sizes, nsources, ncenters)

# }}}


# {{{ neighborhood size counter test

def test_get_qbx_center_neighborhood_sizes(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = 8
    n_arms = 5
    nelements = 50 * n_arms

    mesh = make_curve_mesh(
            NArmedStarfish(n_arms, 0.8),
            np.linspace(0, 1, nelements+1),
            target_order)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source, _ = QBXLayerPotentialSource(
            pre_density_discr, 4 * target_order,
            _max_leaf_refine_weight=64,
            target_association_tolerance=1e-3,
            ).with_refinement()


    t_f = 0.9
    
    result_direct = get_qbx_center_neighborhood_sizes_direct(lpot_source, 8/t_f)
    result_aq = get_qbx_center_neighborhood_sizes(lpot_source, 8/t_f)

    assert (result_direct[0] == result_aq[0]).all()
    assert result_direct[1] == result_aq[1]
    assert result_direct[2] == result_aq[2]

# }}}


if __name__ == "__main__":
    pytest.main([__file__])


# vim: foldmethod=marker
