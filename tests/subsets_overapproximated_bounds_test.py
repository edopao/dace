# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Code moving or iterating over a subset must use its exact extent, not the over-approximation of its bounds.

Strip mining (e.g., ``MapTiling``) writes the end of a tile as ``SymExpr(Min(N - 1, tile_i + 4), tile_i + 4)``: the
exact bound is clamped to the array, the approximation assumes a full tile. On the last, partial tile, using the
approximation moves or reads elements past the end of the subset.
"""

import numpy as np
import pytest
import sympy as sp

import dace
from dace import subsets, symbolic
from dace.libraries.standard import Reduce
from dace.sdfg import memlet_utils

N = dace.symbol('N')
tile_i = dace.symbol('tile_i')

# With these values, the tile `[tile_i, tile_i + 4]` has a single element in an array of size `_N`.
_N = 16
_TILE_I = 15
# The arrays are larger than `_N`, so that an overrun lands in memory that can be checked.
_PADDING = 4


def _partial_tile() -> subsets.Range:
    return subsets.Range([(tile_i, symbolic.SymExpr(sp.Min(N - 1, tile_i + 4), tile_i + 4), 1)])


def _make_sdfg(name: str):
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('tile_i', dace.int64)
    sdfg.add_array('A', [N + _PADDING], dace.int32)
    return sdfg, sdfg.add_state()


def _make_copy_sdfg(name: str):
    sdfg, state = _make_sdfg(name)
    sdfg.add_array('B', [N + _PADDING], dace.int32)
    edge = state.add_edge(state.add_access('A'), None, state.add_access('B'), None,
                          dace.Memlet(data='A', subset=_partial_tile(), other_subset=_partial_tile()))
    return sdfg, state, edge


def _run_copy(sdfg: dace.SDFG) -> np.ndarray:
    A = np.arange(_N + _PADDING, dtype=np.int32)
    B = np.full(_N + _PADDING, -1, dtype=np.int32)
    sdfg(A=A, B=B, N=_N, tile_i=_TILE_I)
    return B


def _expected_copy() -> np.ndarray:
    expected = np.full(_N + _PADDING, -1, dtype=np.int32)
    expected[_TILE_I] = _TILE_I
    return expected


def test_bounding_box_size_exact():
    rng = _partial_tile()
    assert rng.bounding_box_size() == [5]
    assert [s.subs({N: _N, tile_i: _TILE_I}) for s in rng.bounding_box_size_exact()] == [1]


def test_num_elements_exact():
    rng = _partial_tile()
    assert rng.num_elements() == 5
    assert rng.num_elements_exact().subs({N: _N, tile_i: _TILE_I}) == 1


def test_copy():
    sdfg, _, _ = _make_copy_sdfg('overapproximated_bounds_copy')
    assert np.array_equal(_run_copy(sdfg), _expected_copy())


def test_memlet_to_map():
    sdfg, state, edge = _make_copy_sdfg('overapproximated_bounds_memlet_to_map')
    map_entry, _ = memlet_utils.memlet_to_map(edge=edge, state=state, sdfg=sdfg)
    ((_, end, _), ) = map_entry.map.range
    assert end.subs({N: _N, tile_i: _TILE_I}) == 0
    assert np.array_equal(_run_copy(sdfg), _expected_copy())


@pytest.mark.parametrize('implementation', ['pure', 'pure-seq', 'OpenMP'])
def test_reduce(implementation: str):
    sdfg, state = _make_sdfg(f'overapproximated_bounds_reduce_{implementation.replace("-", "_")}')
    sdfg.add_array('S', [1], dace.int32)
    red = Reduce('sum', wcr='lambda a, b: a + b', axes=None, identity=0)
    red.implementation = implementation
    state.add_node(red)
    state.add_edge(state.add_read('A'), None, red, None, dace.Memlet(data='A', subset=_partial_tile()))
    state.add_edge(red, None, state.add_write('S'), None, dace.Memlet('S[0]'))

    A = np.arange(_N + _PADDING, dtype=np.int32)
    S = np.zeros(1, dtype=np.int32)
    sdfg(A=A, S=S, N=_N, tile_i=_TILE_I)
    assert S[0] == _TILE_I


if __name__ == '__main__':
    test_bounding_box_size_exact()
    test_num_elements_exact()
    test_copy()
    test_memlet_to_map()
    test_reduce('pure')
    test_reduce('pure-seq')
    test_reduce('OpenMP')
