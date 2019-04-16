#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import utils
import sys


def gen_ind_subsets(graph: nx.Graph):
    m = graph.number_of_edges()
    if m == 0:  # given graph is already ind. so all subsets are ind.
        for ss in utils.powerset(graph.nodes_iter()):
            yield set(ss)
    # at least one edge in graph
    u, v = utils.first(graph.edges_iter())
    stack = [((u, v), 0, set(), set())]
    while stack:
        # print("popped", stack[-1], "remaining", stack[:-1])
        cur_edge, state, discard, include = stack.pop()
        if state > 2:  # ignore invalid state
            continue
        u, v = cur_edge
        if state == 0:
            new_discard = discard | set(graph.neighbors_iter(u))
            new_include = include | {u}
        elif state == 1:
            new_discard = discard | set(graph.neighbors_iter(v))
            new_include = include | {v}
        else:
            new_discard = discard | {u, v}
            new_include = include
        nodes = set(graph.nodes_iter())
        # print("checking subgraph on", nodes - new_discard - new_include)
        new_graph = graph.subgraph(nodes - new_discard - new_include)
        # print("found", new_graph.number_of_edges(), "edges")
        stack.append((cur_edge, state + 1, discard, include))
        if new_graph.number_of_edges() == 0:  # found ind. set
            # print("yielding", new_include|new_graph.nodes)
            for subset in utils.powerset(new_graph.nodes_iter()):
                yield new_include | set(subset)
        else:
            new_edge = utils.first(new_graph.edges_iter())
            # print("appending", new_edge)
            stack.append((new_edge, 0, new_discard, new_include))


def brute_ind_subsets(graph: nx.Graph):
    for ss in utils.powerset(graph.nodes_iter()):
        edge_found = False
        for u in ss:
            for v in ss:
                if u == v: continue
                if v in graph[u]:  # edge between u and v
                    edge_found = True
                    break
            if edge_found: break  # break out of subset
        if edge_found:  # try next subset
            continue
        else:  # independent subset
            yield ss


def compress(graph: nx.Graph, vc, smart_subsets=True):
    """
    Given a vc tries to find a vc of strictly smaller size
    If unable to find, then provided vc is optimal.

    :param graph: Graph to find vc for
    :param vc: Supplied vc
    :param smart_subsets: generate subsets smartly by branching
    :return: 2-tuple (status, new_vc) where status is True if
    it found a smaller vc in which case new_vc contains this vc.
    Else status is false and new_vc is empty_set()
    """
    vc = set(vc)
    ctr = 0
    subset_func = gen_ind_subsets if smart_subsets else brute_ind_subsets
    for ss in subset_func(graph.subgraph(vc)):
        if ctr%1000==0: print(ctr,"done",file=sys.stderr)
        ctr += 1
        if len(ss) < 1:  # ignore empty subsets
            continue
        nbrs = set()
        for u in ss:  # accumulate neighborhood of subset
            nbrs.update(graph.neighbors_iter(u))
        if len(nbrs) < len(ss):  # found smaller neighborhood
            return True, vc - ss | nbrs
    return False, set()
