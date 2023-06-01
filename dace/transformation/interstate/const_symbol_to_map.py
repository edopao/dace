import re

from dace.sdfg import graph
from dace.properties import make_properties, SubsetProperty
from dace import Memlet, subsets
from dace import nodes, sdfg as sd
from dace.transformation import transformation as xf


def reroute_edge_through_map(state, e: graph.Edge, m: nodes.Map):
    state.add_memlet_path(e.src, m, e.dst, src_conn=e.src_conn, dst_conn=e.dst_conn, memlet=e.data)
    state.remove_edge(e)

@make_properties
class ConstSymbolToMap(xf.MultiStateTransformation):
    symbol_state = xf.PatternNode(sd.SDFGState)
    compute_state = xf.PatternNode(sd.SDFGState)
    symbol_edge = None
    memlet_symbols = {}

    @classmethod
    def expressions(cls):
        maptoloop_sdfg = sd.graph.OrderedDiGraph()
        maptoloop_sdfg.add_nodes_from([cls.symbol_state, cls.compute_state])
        maptoloop_sdfg.add_edge(cls.symbol_state, cls.compute_state, sd.InterstateEdge())
        return [maptoloop_sdfg]

    # Some naming conventions:
    # - Source node is called symbol state, we check if there is an out-edge which defines some symbols
    #   as array access.
    # - The next state is called compute state, and for now has to be a sink state: this is a pure restriction
    #   to be on the safe side, and ensure that the interstate symbols are not used anywhere else;
    #   the strict requirement would just be that the symbols are not modified, so they can be treated as constants.
    def can_be_applied(self, graph: sd.SDFGState, expr_index: int, sdfg: sd.SDFG, permissive: bool = False):
        # For now put restriction that compute state is a sink node
        if len(graph.out_edges(self.compute_state)) != 0:
            return False
        # Find the edge that connects the symbol node to the compute node
        for e in graph.out_edges(self.symbol_state):
            if e.dst == self.compute_state:
                self.symbol_edge = e
                break
        assert(self.symbol_edge != None)
        # Find inter-state symbols on this edge used by the compute node
        # N.B. should be all edges as long as the compute node is a sink node
        for k, v in self.symbol_edge.data.assignments.items():
            if k in self.compute_state.free_symbols:
                # select only the assignments that contain array access
                try:
                    m = Memlet(v)
                except SyntaxError:
                    continue
                if m.subset != None and m.data in sdfg.arrays:
                    self.memlet_symbols[k] = m
        return len(self.memlet_symbols) != 0

    def apply(self, graph, sdfg: sd.SDFG):
        map = nodes.Map(self.compute_state.label + "_map", [], [])
        me = nodes.MapEntry(map)
        mx = nodes.MapExit(map)

        # replace inter-state symbols with map
        id_table = {}
        for s, m in self.memlet_symbols.items():
            # check that symbol is not defined yet
            assert (s not in map.params)
            # create in-connector with unique id
            if m.data not in id_table.keys():
                id_table[m.data] = 1
            else:
                id_table[m.data] += 1
            connector_name = m.data + '_' + str(id_table[m.data])
            connector_type = sdfg.arrays[m.data].dtype
            me.add_in_connector(connector_name, connector_type)
            # create access node, if not already existing
            access_node = None
            for n in self.compute_state.source_nodes():
                if isinstance(n, nodes.AccessNode) and n.data == m.data:
                    access_node = n
                    break
            if not access_node:
                access_node = nodes.AccessNode(m.data)
                self.compute_state.add_node(access_node)
            # create edge from access node to map entry node
            self.compute_state.add_edge(access_node, None, me, connector_name, m)
            # update table of symbols in map range
            map.params.append(s)
            map.range += subsets.Range(SubsetProperty.from_string(connector_name))

        # search for source nodes and corresponding edges
        for n in self.compute_state.source_nodes():
            # ignore the access nodes newly added for map symbols
            if n.data not in id_table.keys():
                for e in self.compute_state.edges():
                    if e.src == n:
                        reroute_edge_through_map(self.compute_state, e, me)

        # search for sink nodes and corresponding edges
        for n in self.compute_state.sink_nodes():
            # ignore the access nodes newly added for map symbols
            if n.data not in id_table.keys():
                for e in self.compute_state.edges():
                    if e.dst == n:
                        reroute_edge_through_map(self.compute_state, e, mx)

        # remove symbol assignments from inter-state edge
        for s in self.memlet_symbols.keys():
            self.symbol_edge.data.assignments.pop(s)
            if s in sdfg.symbols:
                sdfg.remove_symbol(s)

        # removal of symbol state
        if len(self.symbol_edge.data.assignments) == 0 and len(self.symbol_state.nodes()) == 0:
            sdfg.remove_node(self.symbol_state)
