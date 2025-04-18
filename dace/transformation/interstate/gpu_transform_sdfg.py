# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains inter-state transformations of an SDFG to run on the GPU. """

from dace import data, memlet, dtypes, sdfg as sd, subsets as sbs, propagate_memlets_sdfg
from dace.sdfg import nodes, scope
from dace.sdfg import utils as sdutil
from dace.sdfg.replace import replace_in_codeblock
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import transformation, helpers as xfh
from dace.properties import ListProperty, Property, make_properties
from collections import defaultdict
from copy import deepcopy as dc
from sympy import floor
from typing import Dict, List, Set, Tuple

gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]


def _recursive_out_check(node, state, gpu_scalars):
    """
    Recursively checks if the outputs of a node are scalars and if they are/should be stored in GPU memory.
    """
    scalset = set()
    scalout = True
    sdfg = state.parent
    for e in state.out_edges(node):
        last_edge = state.memlet_path(e)[-1]
        if isinstance(last_edge.dst, nodes.AccessNode):
            desc = sdfg.arrays[last_edge.dst.data]
            if isinstance(desc, data.Scalar):
                if desc.storage in gpu_storage or last_edge.dst.data in gpu_scalars:
                    scalout = False
                scalset.add(last_edge.dst.data)
                sset, ssout = _recursive_out_check(last_edge.dst, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.shape == (1, ):  # Pseudo-scalar
                scalout = False
                sset, ssout = _recursive_out_check(last_edge.dst, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.storage not in gpu_storage and last_edge.data.num_elements() == 1:
                sset, ssout = _recursive_out_check(last_edge.dst, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            scalout = False
    return scalset, scalout


def _recursive_in_check(node, state, gpu_scalars):
    """
    Recursively checks if the inputs of a node are scalars and if they are/should be stored in GPU memory.
    """
    scalset = set()
    scalout = True
    sdfg = state.parent
    for e in state.in_edges(node):
        last_edge = state.memlet_path(e)[0]
        if isinstance(last_edge.src, nodes.AccessNode):
            desc = sdfg.arrays[last_edge.src.data]
            if isinstance(desc, data.Scalar):
                if desc.storage in gpu_storage or last_edge.src.data in gpu_scalars:
                    scalout = False
                scalset.add(last_edge.src.data)
                sset, ssout = _recursive_in_check(last_edge.src, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.shape == (1, ):  # Pseudo-scalar
                scalout = False
                sset, ssout = _recursive_in_check(last_edge.src, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            if desc.storage not in gpu_storage and last_edge.data.num_elements() == 1:
                sset, ssout = _recursive_in_check(last_edge.src, state, gpu_scalars)
                scalset = scalset.union(sset)
                scalout = scalout and ssout
                continue
            scalout = False
    return scalset, scalout


@make_properties
@transformation.explicit_cf_compatible
class GPUTransformSDFG(transformation.MultiStateTransformation):
    """ Implements the GPUTransformSDFG transformation.

        Transforms a whole SDFG to run on the GPU:

            1. Acquire metadata about SDFG and arrays
            2. Replace all non-transients with their GPU counterparts
            3. Copy-in state from host to GPU
            4. Copy-out state from GPU to host
            5. Re-store Default-top/CPU_Heap transients as GPU_Global
            6. Global tasklets are wrapped with a map of size 1
            7. Global Maps are re-scheduled to use the GPU
            8. Make data ready for interstate edges that use them
            9. Re-apply simplification to get rid of extra states and transients
    """

    toplevel_trans = Property(desc="Make all GPU transients top-level", dtype=bool, default=True)

    register_trans = Property(desc="Make all transients inside GPU maps registers", dtype=bool, default=True)

    sequential_innermaps = Property(desc="Make all internal maps Sequential", dtype=bool, default=True)

    skip_scalar_tasklets = Property(desc="If True, does not transform tasklets "
                                    "that manipulate (Default-stored) scalars",
                                    dtype=bool,
                                    default=True)

    simplify = Property(desc='Reapply simplification after modifying graph', dtype=bool, default=True)

    exclude_copyin = Property(desc="Exclude these arrays from being copied into the device "
                              "(comma-separated)",
                              dtype=str,
                              default='')

    exclude_tasklets = Property(desc="Exclude these tasklets from being processed as CPU tasklets "
                                "(comma-separated)",
                                dtype=str,
                                default='')

    exclude_copyout = Property(desc="Exclude these arrays from being copied out of the device "
                               "(comma-separated)",
                               dtype=str,
                               default='')

    host_maps = ListProperty(desc='List of map GUIDs, the passed maps are not offloaded to the GPU',
                             element_type=str,
                             default=None,
                             allow_none=True)

    host_data = ListProperty(desc='List of data names, the passed data are not offloaded to the GPU',
                             element_type=str,
                             default=None,
                             allow_none=True)

    @staticmethod
    def annotates_memlets():
        # Skip memlet propagation for now
        return True

    @classmethod
    def expressions(cls):
        # Matches anything
        return [sd.SDFG('_')]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        for node, _ in sdfg.all_nodes_recursive():
            # Consume scopes are currently unsupported
            if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
                return False

        for state in sdfg.states():
            schildren = state.scope_children()
            for node in schildren[None]:
                # If two top-level tasklets are connected with a code->code
                # memlet, they will transform into an invalid SDFG
                if (isinstance(node, nodes.CodeNode)
                        and any(isinstance(e.dst, nodes.CodeNode) for e in state.out_edges(node))):
                    return False
        return True

    def _get_marked_inputs_and_outputs(self, state, entry_node) -> list:
        if not self.host_data and not self.host_maps:
            return []
        marked_sources = [state.memlet_tree(e).root().edge.src for e in state.in_edges(entry_node)]
        marked_sources = [
            sdutil.get_view_node(state, node) if isinstance(node, data.View) else node for node in marked_sources
        ]
        marked_destinations = [
            state.memlet_tree(e).root().edge.dst for e in state.in_edges(state.exit_node(entry_node))
        ]
        marked_destinations = [
            sdutil.get_view_node(state, node) if isinstance(node, data.View) else node for node in marked_destinations
        ]
        marked_accesses = [
            n.data for n in (marked_sources + marked_destinations)
            if n is not None and isinstance(n, nodes.AccessNode) and n.data in self.host_data
        ]
        return marked_accesses

    def _output_or_input_is_marked_host(self, state, entry_node) -> bool:
        marked_accesses = self._get_marked_inputs_and_outputs(state, entry_node)
        return len(marked_accesses) > 0

    def apply(self, _, sdfg: sd.SDFG):
        #######################################################
        # Step 0: SDFG metadata

        # Find all input and output data descriptors
        input_nodes: List[Tuple[str, data.Data]] = []
        output_nodes: List[Tuple[str, data.Data]] = []
        global_code_nodes: Dict[sd.SDFGState, nodes.Tasklet] = defaultdict(list)
        if self.host_maps is None:
            self.host_maps = []
        if self.host_data is None:
            self.host_data = []

        # Propagate memlets to ensure that we can find the true array subsets that are written.
        propagate_memlets_sdfg(sdfg)

        # Input and ouputs of all host_maps need to be marked as host_data
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.EntryNode) and node.guid in self.host_maps:
                    accesses = self._get_marked_inputs_and_outputs(state, node)
                    self.host_data.extend(accesses)

        for state in sdfg.states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient == False):
                    if (state.out_degree(node) > 0 and node.data not in input_nodes):
                        # Special case: nodes that lead to top-level dynamic
                        # map ranges must stay on host
                        for e in state.out_edges(node):
                            last_edge = state.memlet_path(e)[-1]
                            if (isinstance(last_edge.dst, nodes.EntryNode)
                                    and ((last_edge.dst_conn and not last_edge.dst_conn.startswith('IN_')
                                          and sdict[last_edge.dst] is None) or (last_edge.dst in self.host_maps))):
                                break
                        else:
                            input_nodes.append((node.data, node.desc(sdfg)))
                    if (state.in_degree(node) > 0 and node.data not in output_nodes
                            and node.data not in self.host_data):
                        output_nodes.append((node.data, node.desc(sdfg)))

            # Input nodes may also be nodes with WCR memlets and no identity
            for e in state.edges():
                if e.data.wcr is not None:
                    if (e.data.data not in input_nodes and sdfg.arrays[e.data.data].transient == False):
                        input_nodes.append((e.data.data, sdfg.arrays[e.data.data]))

        start_block = sdfg.start_block
        end_blocks = sdfg.sink_nodes()

        #######################################################
        # Step 1: Create cloned GPU arrays and replace originals

        data_already_on_gpu = {}

        cloned_arrays = {}
        for inodename, inode in set(input_nodes):
            if inode.storage == dtypes.StorageType.GPU_Global:
                data_already_on_gpu[inodename] = None
                continue
            if isinstance(inode, data.Scalar):  # Scalars can remain on host
                continue
            newdesc = inode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + inodename, newdesc, find_new_name=True)
            cloned_arrays[inodename] = name

        for onodename, onode in set(output_nodes):
            if onode.storage == dtypes.StorageType.GPU_Global:
                data_already_on_gpu[onodename] = None
                continue
            if onodename in cloned_arrays:
                continue
            newdesc = onode.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            name = sdfg.add_datadesc('gpu_' + onodename, newdesc, find_new_name=True)
            cloned_arrays[onodename] = name

            # The following ensures that when writing to a subset of an array, we don't overwrite the rest of the array
            # when copying back to the host. This is done by adding the array to the `inputs_nodes,` which will copy
            # the entire array to the GPU.
            if (onodename, onode) not in input_nodes:
                found_full_write = False
                full_subset = sbs.Range.from_array(onode)
                try:
                    for state in sdfg.states():
                        for node in state.nodes():
                            if (isinstance(node, nodes.AccessNode) and node.data == onodename):
                                for e in state.in_edges(node):
                                    if e.data.get_dst_subset(e, state) == full_subset:
                                        is_full = True
                                        for pe in state.memlet_tree(e):
                                            vol = pe.data.volume
                                            size = pe.data.get_dst_subset(pe, state).num_elements()
                                            if pe.data.dynamic or vol / size != floor(vol / size):
                                                is_full = False
                                                break
                                        if not is_full:
                                            continue
                                        found_full_write = True
                                        raise StopIteration
                except StopIteration:
                    assert found_full_write
                if not found_full_write:
                    input_nodes.append((onodename, onode))

        check_memlets: List[memlet.Memlet] = []
        for edge in sdfg.all_interstate_edges():
            check_memlets.extend(edge.data.get_read_memlets(sdfg.arrays))
        for blk in sdfg.all_control_flow_blocks():
            if isinstance(blk, AbstractControlFlowRegion):
                check_memlets.extend(blk.get_meta_read_memlets())
        for mem in check_memlets:
            if sdfg.arrays[mem.data].storage == dtypes.StorageType.GPU_Global:
                data_already_on_gpu[mem.data] = None

        # Replace nodes
        for state in sdfg.states():
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.data in cloned_arrays):
                    node.data = cloned_arrays[node.data]

        # Replace memlets
        for state in sdfg.states():
            for edge in state.edges():
                if edge.data.data in cloned_arrays:
                    edge.data.data = cloned_arrays[edge.data.data]

        #######################################################
        # Step 2: Create copy-in state
        excluded_copyin = self.exclude_copyin.split(',')

        copyin_state = sdfg.add_state(sdfg.label + '_copyin')
        sdfg.add_edge(copyin_state, start_block, sd.InterstateEdge())

        for nname, desc in dtypes.deduplicate(input_nodes):
            if nname in excluded_copyin or nname not in cloned_arrays:
                continue
            src_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(cloned_arrays[nname], debuginfo=desc.debuginfo)
            copyin_state.add_node(src_array)
            copyin_state.add_node(dst_array)
            copyin_state.add_nedge(src_array, dst_array, memlet.Memlet.from_array(src_array.data, src_array.desc(sdfg)))

        #######################################################
        # Step 3: Create copy-out state
        excluded_copyout = self.exclude_copyout.split(',')

        copyout_state = sdfg.add_state(sdfg.label + '_copyout')
        for state in end_blocks:
            sdfg.add_edge(state, copyout_state, sd.InterstateEdge())

        for nname, desc in dtypes.deduplicate(output_nodes):
            if nname in excluded_copyout or nname not in cloned_arrays:
                continue
            src_array = nodes.AccessNode(cloned_arrays[nname], debuginfo=desc.debuginfo)
            dst_array = nodes.AccessNode(nname, debuginfo=desc.debuginfo)
            copyout_state.add_node(src_array)
            copyout_state.add_node(dst_array)
            copyout_state.add_nedge(src_array, dst_array, memlet.Memlet.from_array(dst_array.data,
                                                                                   dst_array.desc(sdfg)))

        #######################################################
        # Step 4: Change all top-level maps and library nodes to GPU schedule

        gpu_nodes: Set[Tuple[SDFGState, nodes.Node]] = set()
        for state in sdfg.states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if sdict[node] is None:
                    if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                        if node.guid:
                            node.schedule = dtypes.ScheduleType.GPU_Default
                            gpu_nodes.add((state, node))
                    elif isinstance(node, nodes.EntryNode):
                        if node.guid not in self.host_maps and not self._output_or_input_is_marked_host(state, node):
                            node.schedule = dtypes.ScheduleType.GPU_Device
                            gpu_nodes.add((state, node))
                elif self.sequential_innermaps:
                    if isinstance(node, (nodes.EntryNode, nodes.LibraryNode)):
                        node.schedule = dtypes.ScheduleType.Sequential
                    elif isinstance(node, nodes.NestedSDFG):
                        for nnode, _ in node.sdfg.all_nodes_recursive():
                            if isinstance(nnode, (nodes.EntryNode, nodes.LibraryNode)):
                                nnode.schedule = dtypes.ScheduleType.Sequential

        # NOTE: The outputs of LibraryNodes, NestedSDFGs and Map that have GPU schedule must be moved to GPU memory.
        # TODO: Also use GPU-shared and GPU-register memory when appropriate.
        for state, node in gpu_nodes:
            if isinstance(node, (nodes.LibraryNode, nodes.NestedSDFG)):
                for e in state.out_edges(node):
                    dst = state.memlet_path(e)[-1].dst
                    if isinstance(dst, nodes.AccessNode):
                        desc = sdfg.arrays[dst.data]
                        desc.storage = dtypes.StorageType.GPU_Global
            if isinstance(node, nodes.EntryNode):
                for e in state.out_edges(state.exit_node(node)):
                    dst = state.memlet_path(e)[-1].dst
                    if isinstance(dst, nodes.AccessNode):
                        desc = sdfg.arrays[dst.data]
                        desc.storage = dtypes.StorageType.GPU_Global

        #######################################################
        # Step 5: Collect free tasklets and check for scalars that have to be moved to the GPU
        # Also recursively call GPUTransformSDFG on NestedSDFGs that have GPU device schedule but are not actually
        # inside a GPU kernel.

        gpu_scalars = {}
        nsdfgs: List[Tuple[nodes.NestedSDFG, SDFGState]] = []
        changed = True
        # Iterates over Tasklets that not inside a GPU kernel. Such Tasklets must be moved inside a GPU kernel only
        # if they write to GPU memory. The check takes into account the fact that GPU kernels can read host-based
        # Scalars, but cannot write to them.
        while changed:
            changed = False
            for state in sdfg.states():
                for node in state.nodes():
                    # Handle NestedSDFGs later.
                    if isinstance(node, nodes.NestedSDFG):
                        if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(
                                state.parent, state, node):
                            nsdfgs.append((node, state))
                    elif isinstance(node, nodes.Tasklet):
                        if node in global_code_nodes[state]:
                            continue
                        if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(
                                state.parent, state, node):
                            scalars, scalar_output = _recursive_out_check(node, state, gpu_scalars)
                            sset, ssout = _recursive_in_check(node, state, gpu_scalars)
                            scalars = scalars.union(sset)
                            scalar_output = scalar_output and ssout
                            csdfg = state.parent
                            # If the tasklet is not adjacent only to scalars or it is in a GPU scope.
                            # The latter includes NestedSDFGs that have a GPU-Device schedule but are not in a GPU kernel.
                            if (not scalar_output
                                    or (csdfg.parent is not None
                                        and csdfg.parent_nsdfg_node.schedule == dtypes.ScheduleType.GPU_Default)):
                                global_code_nodes[state].append(node)
                                gpu_scalars.update({k: None for k in scalars})
                                changed = True

        # Apply GPUTransformSDFG recursively to NestedSDFGs.
        for node, state in nsdfgs:
            excl_copyin = set()
            for e in state.in_edges(node):
                src = state.memlet_path(e)[0].src
                if isinstance(src, nodes.AccessNode) and sdfg.arrays[src.data].storage in gpu_storage:
                    excl_copyin.add(e.dst_conn)
                    node.sdfg.arrays[e.dst_conn].storage = sdfg.arrays[src.data].storage
            excl_copyout = set()
            for e in state.out_edges(node):
                dst = state.memlet_path(e)[-1].dst
                if isinstance(dst, nodes.AccessNode) and sdfg.arrays[dst.data].storage in gpu_storage:
                    excl_copyout.add(e.src_conn)
                    node.sdfg.arrays[e.src_conn].storage = sdfg.arrays[dst.data].storage
            # TODO: Do we want to copy here the options from the top-level SDFG?
            node.sdfg.apply_transformations(
                GPUTransformSDFG, {
                    'exclude_copyin': ','.join([str(n) for n in excl_copyin]),
                    'exclude_copyout': ','.join([str(n) for n in excl_copyout])
                })

        #######################################################
        # Step 6: Modify transient data storage

        const_syms = xfh.constant_symbols(sdfg)

        for state in sdfg.states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient:
                    nodedesc = node.desc(sdfg)

                    # Special case: nodes that lead to dynamic map ranges must stay on host
                    if any(isinstance(state.memlet_path(e)[-1].dst, nodes.EntryNode) for e in state.out_edges(node)):
                        continue

                    if sdict[node] is None and nodedesc.storage not in gpu_storage:

                        # Scalars were already checked.
                        if isinstance(nodedesc, data.Scalar) and not node.data in gpu_scalars:
                            continue

                        # NOTE: the cloned arrays match too but it's the same storage so we don't care
                        if node.data not in self.host_data:
                            nodedesc.storage = dtypes.StorageType.GPU_Global

                        # Try to move allocation/deallocation out of loops
                        dsyms = set(map(str, nodedesc.free_symbols))
                        if (self.toplevel_trans and not isinstance(nodedesc, (data.Stream, data.View))
                                and len(dsyms - const_syms) == 0):
                            nodedesc.lifetime = dtypes.AllocationLifetime.SDFG
                    elif nodedesc.storage not in gpu_storage:
                        # Make internal transients registers
                        if self.register_trans:
                            nodedesc.storage = dtypes.StorageType.Register

        #######################################################
        # Step 7: Wrap free tasklets and nested SDFGs with a GPU map

        # Extend global_code_nodes with tasklets that write/read from an array
        # Previous steps map all arrays to GPU storage, but only checks tasklets that write to/read from
        # Scalars to be wrapped in a GPU Map
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    if node in global_code_nodes[state]:
                        continue
                    if state.entry_node(node) is None and not scope.is_devicelevel_gpu_kernel(
                            state.parent, state, node):
                        memlet_path_roots = set()
                        memlet_path_roots = memlet_path_roots.union(
                            [state.memlet_tree(e).root().edge.src for e in state.in_edges(node)])
                        memlet_path_roots = memlet_path_roots.union(
                            [state.memlet_tree(e).root().edge.dst for e in state.out_edges(node)])
                        gpu_accesses = [
                            n.data for n in memlet_path_roots
                            if isinstance(n, nodes.AccessNode) and sdfg.arrays[n.data].storage in gpu_storage
                        ]
                        if len(gpu_accesses) > 0:
                            global_code_nodes[state].append(node)

        for state, gcodes in global_code_nodes.items():
            for gcode in gcodes:
                if gcode.label in self.exclude_tasklets.split(','):
                    continue
                # Create map and connectors
                me, mx = state.add_map(gcode.label + '_gmap', {gcode.label + '__gmapi': '0:1'},
                                       schedule=dtypes.ScheduleType.GPU_Device)
                # Store in/out edges in lists so that they don't get corrupted when they are removed from the graph.
                in_edges = list(state.in_edges(gcode))
                out_edges = list(state.out_edges(gcode))
                me.in_connectors = {('IN_' + e.dst_conn): None for e in in_edges}
                me.out_connectors = {('OUT_' + e.dst_conn): None for e in in_edges}
                mx.in_connectors = {('IN_' + e.src_conn): None for e in out_edges}
                mx.out_connectors = {('OUT_' + e.src_conn): None for e in out_edges}

                # Create memlets through map
                for e in in_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, me, 'IN_' + e.dst_conn, e.data)
                    state.add_edge(me, 'OUT_' + e.dst_conn, e.dst, e.dst_conn, dc(e.data))
                for e in out_edges:
                    state.remove_edge(e)
                    state.add_edge(e.src, e.src_conn, mx, 'IN_' + e.src_conn, e.data)
                    state.add_edge(mx, 'OUT_' + e.src_conn, e.dst, e.dst_conn, dc(e.data))

                # Map without inputs
                if len(in_edges) == 0:
                    state.add_nedge(me, gcode, memlet.Memlet())

        #######################################################
        # Step 8: Introduce copy-out if data used in outgoing interstate edges

        cloned_data = set(cloned_arrays.keys()).union(gpu_scalars.keys()).union(data_already_on_gpu.keys())

        def _create_copy_out(arrays_used: Set[str]) -> Dict[str, str]:
            # Add copy-out nodes
            name_mapping = {}
            for nname in arrays_used:
                # Handle GPU scalars
                if nname in gpu_scalars:
                    hostname = gpu_scalars[nname]
                    if not hostname:
                        desc = sdfg.arrays[nname].clone()
                        desc.storage = dtypes.StorageType.CPU_Heap
                        desc.transient = True
                        hostname = sdfg.add_datadesc('host_' + nname, desc, find_new_name=True)
                        gpu_scalars[nname] = hostname
                    else:
                        desc = sdfg.arrays[hostname]
                    devicename = nname
                elif nname in data_already_on_gpu:
                    hostname = data_already_on_gpu[nname]
                    if not hostname:
                        desc = sdfg.arrays[nname].clone()
                        desc.storage = dtypes.StorageType.CPU_Heap
                        desc.transient = True
                        hostname = sdfg.add_datadesc('host_' + nname, desc, find_new_name=True)
                        data_already_on_gpu[nname] = hostname
                    else:
                        desc = sdfg.arrays[hostname]
                    devicename = nname
                else:
                    desc = sdfg.arrays[nname]
                    hostname = nname
                    devicename = cloned_arrays[nname]

                src_array = nodes.AccessNode(devicename, debuginfo=desc.debuginfo)
                dst_array = nodes.AccessNode(hostname, debuginfo=desc.debuginfo)
                co_state.add_node(src_array)
                co_state.add_node(dst_array)
                co_state.add_nedge(src_array, dst_array, memlet.Memlet.from_array(dst_array.data, dst_array.desc(sdfg)))
                name_mapping[devicename] = hostname
            return name_mapping

        for block in list(sdfg.all_control_flow_blocks()):
            arrays_used = set()
            for e in block.parent_graph.out_edges(block):
                # Used arrays = intersection between symbols and cloned data
                arrays_used.update(set(e.data.free_symbols) & cloned_data)

            # Create a state and copy out used arrays
            if len(arrays_used) > 0:
                co_state = block.parent_graph.add_state(block.label + '_icopyout')

                # Reconnect outgoing edges to after interim copyout state
                for e in block.parent_graph.out_edges(block):
                    sdutil.change_edge_src(block.parent_graph, block, co_state)
                # Add unconditional edge to interim state
                block.parent_graph.add_edge(block, co_state, sd.InterstateEdge())
                mapping = _create_copy_out(arrays_used)
                for devicename, hostname in mapping.items():
                    for e in block.parent_graph.out_edges(co_state):
                        e.data.replace(devicename, hostname, False)

        for block in list(sdfg.all_control_flow_blocks()):
            arrays_used = set(block.used_symbols(all_symbols=True, with_contents=False)) & cloned_data

            # Create a state and copy out used arrays
            if len(arrays_used) > 0:
                co_state = block.parent_graph.add_state_before(block, block.label + '_icopyout')
                mapping = _create_copy_out(arrays_used)
                for devicename, hostname in mapping.items():
                    block.replace_meta_accesses({devicename: hostname})

        # Step 9: Simplify
        if not self.simplify:
            return

        sdfg.simplify()
