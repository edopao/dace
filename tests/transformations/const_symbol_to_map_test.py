import unittest

import numpy as np

import dace
from dace.libraries.blas import MatMul
from dace.sdfg import nodes
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate.const_symbol_to_map import ConstSymbolToMap


def make_spmv_sdfg():
    sdfg = dace.SDFG(f"spmv")
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    state = sdfg.add_state("state")
    body = sdfg.add_state("body")
    after = sdfg.add_state("after")

    M, N, nnz = (dace.symbol(s, dtype=dace.uint32) for s in ('M', 'N', 'nnz'))

    sdfg.add_array('A_row', [M + 1], dace.uint32)
    sdfg.add_array('A_col', [nnz], dace.uint32)
    sdfg.add_array('A_val', [nnz], dace.float64)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('y', [M], dace.float64)

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "0"}))
    sdfg.add_edge(guard, state, dace.InterstateEdge(condition="i < M"))
    sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i >= M"))
    sdfg.add_edge(state, body, dace.InterstateEdge(
        assignments={"__start": "A_row[i]", "__stop": "A_row[i + 1]"}))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"i": "i + 1"}))

    # Create access nodes for reading and writing
    A_col_node = body.add_access('A_col')
    A_val_node = body.add_access('A_val')
    x_node = body.add_access('x')
    y_node = body.add_access('y')

    # Create a map in which a tasklet will reside
    me, mx = body.add_map('indirect_x_slice', dict(__i0='__start:__stop'))

    # Create transient nodes for temporary storage
    x_slice_node = body.add_transient('x_slice', ('__stop - __start',), dace.float64)

    # Create tasklet for indirection of x values
    indirection_tasklet = body.add_tasklet('indirection', {'__arr', '__inp0'}, {'__out'}, '__out = __arr[__inp0]')
    # input path (src->me->tasklet[a])
    body.add_memlet_path(x_node, me, indirection_tasklet, dst_conn='__arr',
                         memlet=dace.Memlet(data='x', subset='0:N'))
    body.add_memlet_path(A_col_node, me, indirection_tasklet, dst_conn='__inp0',
                         memlet=dace.Memlet(data='A_col', subset='__i0'))
    # output path (tasklet[b]->mx->dst)
    body.add_memlet_path(indirection_tasklet, mx, x_slice_node, src_conn='__out',
                         memlet=dace.Memlet(data='x_slice', subset='__i0 - __start'))

    matmult_tasklet = MatMul('_MatMult_')
    body.add_node(matmult_tasklet)
    body.add_memlet_path(A_val_node, matmult_tasklet, dst_conn='_a',
                         memlet=dace.Memlet(data='A_val', subset='__start:__stop'))
    body.add_edge(x_slice_node, None, matmult_tasklet, '_b',
                  dace.Memlet(data='x_slice', subset='0:__stop - __start'))
    body.add_edge(matmult_tasklet, '_c', y_node, None, dace.Memlet(data='y', subset='i'))

    return sdfg

def make_sparse_matrix(M, N, nnz):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    from scipy.sparse import random

    matrix = random(M,
                    N,
                    density=nnz / (M * N),
                    format='csr',
                    dtype=np.float64,
                    random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = np.copy(matrix.data)

    return rows, cols, vals, x

def spmv_ref(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y

class TestConstSymbolToMap(unittest.TestCase):

    def test_const_symbol_to_map(self):
        sdfg = make_spmv_sdfg()
        expected_arglist = ['A_col', 'A_row', 'A_val', 'x', 'y', 'M', 'N', 'nnz']

        num_transformations = sdfg.apply_transformations([LoopToMap])
        self.assertEqual(1, num_transformations)
        # run simplify to eliminate the exit state, so that the sink node can be detected
        sdfg.simplify()

        # at this state, the loop body should be implemented as a nested sdfg
        self.assertEqual(1, len(sdfg.nodes()))
        self.assertEqual(['loop_body'], [n.label for n in sdfg.nodes()[0] if isinstance(n, nodes.NestedSDFG)])

        # verify that symbols 'start' and 'stop' are not propagated out of map scope
        self.assertEqual(expected_arglist, list(sdfg.arglist().keys()))

        num_transformations = sdfg.apply_transformations([ConstSymbolToMap])
        self.assertEqual(1, num_transformations)
        sdfg.simplify()

        # test that after transformation there is no nested sdfg
        self.assertEqual(1, len(sdfg.nodes()))
        self.assertEqual([], [n.label for n in sdfg.nodes()[0] if isinstance(n, nodes.NestedSDFG)])

        # verify that symbols 'start' and 'stop' are not propagated out of map scope
        self.assertEqual(expected_arglist, list(sdfg.arglist().keys()))

        M = np.uint32(4096)
        N = np.uint32(8192)
        nnz = np.uint32(2048)

        A_row, A_col, A_val, x = make_sparse_matrix(M, N, nnz)
        y = np.empty(M, dtype=x.dtype)

        sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, y=y, M=M, N=N, nnz=nnz)

        if not all(np.isclose(y, spmv_ref(A_row, A_col, A_val, x))):
            raise ValueError("Validation failed.")


if __name__ == '__main__':
    unittest.main()
