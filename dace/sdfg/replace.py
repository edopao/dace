# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to perform find-and-replace of symbols in SDFGs. """

import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

import sympy as sp

import dace
from dace import dtypes, properties, symbolic
from dace.codegen import cppunparse
from dace.frontend.python.astutils import ASTFindReplace

if TYPE_CHECKING:
    from dace.sdfg.state import StateSubgraphView

tokenize_cpp = re.compile(r'\b\w+\b')


def _internal_replace(sym, symrepl):
    if not isinstance(sym, sp.Basic):
        return sym

    # Filter out only relevant replacements
    fsyms = set(map(str, sym.free_symbols))
    # TODO/NOTE: Could we return the generated strings below as free symbols from Attr instead or ther will be issues?
    for s in set(fsyms):
        if '.' in s:
            tokens = s.split('.')
            for i in range(1, len(tokens)):
                fsyms.add('.'.join(tokens[:i]))
    newrepl = {k: v for k, v in symrepl.items() if str(k) in fsyms}
    if not newrepl:
        return sym

    return sym.subs(newrepl)


def _replsym(symlist, symrepl):
    """ Helper function to replace symbols in various symbolic expressions. """
    if symlist is None:
        return None
    if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
        return _internal_replace(symlist, symrepl)
    for i, dim in enumerate(symlist):
        try:
            symlist[i] = tuple(_internal_replace(d, symrepl) for d in dim)
        except TypeError:
            symlist[i] = _internal_replace(dim, symrepl)
    return symlist


def replace_dict(subgraph: 'StateSubgraphView',
                 repl: Dict[str, str],
                 symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
    """
    Finds and replaces all occurrences of a set of symbols/arrays in the given subgraph.

    :param subgraph: The given graph or subgraph to replace in.
    :param repl: Dictionary of replacements (key -> value).
    :param symrepl: Optional cached dictionary of ``repl`` as symbolic expressions.
    """
    symrepl = symrepl or {
        symbolic.pystr_to_symbolic(symname):
        symbolic.pystr_to_symbolic(new_name) if isinstance(new_name, str) else new_name
        for symname, new_name in repl.items()
    }

    # Replace in node properties
    for node in subgraph.nodes():
        replace_properties_dict(node, repl, symrepl)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data in repl:
            edge.data.data = str(repl[edge.data.data])
        if (edge.data.subset is not None and repl.keys() & edge.data.subset.free_symbols):
            edge.data.subset = _replsym(edge.data.subset, symrepl)
        if (edge.data.other_subset is not None and repl.keys() & edge.data.other_subset.free_symbols):
            edge.data.other_subset = _replsym(edge.data.other_subset, symrepl)
        if symrepl.keys() & edge.data.volume.free_symbols:
            edge.data.volume = _replsym(edge.data.volume, symrepl)


def replace(subgraph: 'StateSubgraphView', name: str, new_name: str):
    """
    Finds and replaces all occurrences of a symbol or array in the given subgraph.

    :param subgraph: The given graph or subgraph to replace in.
    :param name: Name to find.
    :param new_name: Name to replace.
    """
    if str(name) == str(new_name):
        return
    replace_dict(subgraph, {name: new_name})


def replace_in_codeblock(codeblock: properties.CodeBlock, repl: Dict[str, str], node: Optional[Any] = None):
    code = codeblock.code
    if isinstance(code, str) and code:
        lang = codeblock.language
        if lang is dtypes.Language.CPP:  # Replace in C++ code
            prefix = ''
            tokenized = tokenize_cpp.findall(code)
            active_replacements = set()
            for name, new_name in repl.items():
                if name not in tokenized:
                    continue
                # Use local variables and shadowing to replace
                replacement = f'auto {name} = {cppunparse.pyexpr2cpp(new_name)};\n'
                prefix = replacement + prefix
                active_replacements.add(name)

            if prefix:
                codeblock.code = prefix + code
                if node and isinstance(node, dace.nodes.Tasklet):
                    # Ignore replaced symbols since they no longer exist as reads
                    node.ignored_symbols = node.ignored_symbols.union(active_replacements)

        else:
            warnings.warn('Replacement of %s with %s was not made '
                          'for string tasklet code of language %s' % (name, new_name, lang))

    elif codeblock.code is not None:
        afr = ASTFindReplace(repl)
        for stmt in codeblock.code:
            afr.visit(stmt)


def replace_properties_dict(node: Any,
                            repl: Dict[str, str],
                            symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
    symrepl = symrepl or {
        symbolic.pystr_to_symbolic(symname):
        symbolic.pystr_to_symbolic(new_name) if isinstance(new_name, str) else new_name
        for symname, new_name in repl.items()
    }

    for propclass, propval in node.properties():
        if propval is None:
            continue
        pname = propclass.attr_name
        if isinstance(propclass, properties.SymbolicProperty):
            # NOTE: `propval` can be a numeric constant instead of a symbolic expression.
            if not symbolic.issymbolic(propval):
                setattr(node, pname, symbolic.pystr_to_symbolic(str(propval)).subs(symrepl))
            else:
                setattr(node, pname, propval.subs(symrepl))
        elif isinstance(propclass, properties.DataProperty):
            if propval in repl:
                setattr(node, pname, repl[propval])
        elif isinstance(propclass, (properties.RangeProperty, properties.ShapeProperty)):
            setattr(node, pname, _replsym(list(propval), symrepl))
        elif isinstance(propclass, properties.CodeProperty):
            # Don't replace variables that appear as an input or an output
            # connector, as this should shadow the outer declaration.
            reduced_repl = set(repl.keys())
            if hasattr(node, 'in_connectors'):
                reduced_repl -= set(node.in_connectors.keys()) | set(node.out_connectors.keys())
            reduced_repl = {k: repl[k] for k in reduced_repl}
            replace_in_codeblock(propval, reduced_repl, node)
        elif (isinstance(propclass, properties.DictProperty) and pname == 'symbol_mapping'):
            # Symbol mappings for nested SDFGs
            for symname, sym_mapping in propval.items():
                try:
                    propval[symname] = symbolic.pystr_to_symbolic(str(sym_mapping)).subs(symrepl)
                except AttributeError:  # If the symbolified value has no subs
                    pass


def replace_properties(node: Any, symrepl: Dict[symbolic.SymbolicType, symbolic.SymbolicType], name: str,
                       new_name: str):
    replace_properties_dict(node, {name: new_name}, symrepl)


def replace_datadesc_names(sdfg: 'dace.SDFG', repl: Dict[str, str]):
    """ Reduced form of replace which only replaces data descriptor names. """
    # Replace in descriptor repository
    for aname, aval in list(sdfg.arrays.items()):
        if aname in repl:
            del sdfg.arrays[aname]
            sdfg.arrays[repl[aname]] = aval
            if aname in sdfg.constants_prop:
                sdfg.constants_prop[repl[aname]] = sdfg.constants_prop[aname]
                del sdfg.constants_prop[aname]

    for cf in sdfg.all_control_flow_regions():
        # Replace in interstate edges
        for e in cf.edges():
            e.data.replace_dict(repl, replace_keys=False)

        for block in cf.nodes():
            if isinstance(block, dace.SDFGState):
                # Replace in access nodes
                for node in block.data_nodes():
                    if node.data in repl:
                        node.data = repl[node.data]
                    elif '.' in node.data:
                        # Handle structure member accesses where the structure name is being replaced.
                        parts = node.data.split('.')
                        if parts[0] in repl:
                            node.data = repl[parts[0]] + '.' + '.'.join(parts[1:])

                # Replace in memlets
                for edge in block.edges():
                    if edge.data.data is None:
                        continue
                    if edge.data.data in repl:
                        edge.data.data = repl[edge.data.data]
                    elif '.' in edge.data.data:
                        # Handle structure member accesses where the structure name is being replaced.
                        parts = edge.data.data.split('.')
                        if parts[0] in repl:
                            edge.data.data = repl[parts[0]] + '.' + '.'.join(parts[1:])

        # Replace in loop or branch conditions:
        cf.replace_meta_accesses(repl)
