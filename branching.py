#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import networkx as nx
import heapq
from operator import itemgetter
from time import time
import random

import reduction
import utils


def print(*args, **kwargs):
    pass


def edge_branching(graph: nx.Graph, budget: int):
    nodes = set(graph.nodes_iter())
    num_edges = graph.number_of_edges()
    if budget <= 0 and num_edges:
        return False, set()
    elif num_edges == 0:
        return True, set()
    else:  # budget > 0 and num_edges > 0
        cur_edge = utils.first(graph.edges_iter())
        u, v = cur_edge
        for picked in [u, v]:
            res, vc = edge_branching(graph.subgraph(nodes - {picked}), budget - 1)
            if res:
                return True, vc | {picked}
        return False, set()


def vertex_branching(graph: nx.Graph, budget: int):
    #print("in branching start:", graph.number_of_nodes(), graph.number_of_edges(), "budget", budget, file=sys.stderr)
    flag, graph, vc, folded_verts = reduction.rrhandler(graph, budget)
    #print("in branching vc:", vc, "fold_verts:", folded_verts, "new graph", graph.number_of_nodes(), file=sys.stderr)
    if not flag:  # NO instance
        return False, set()
    budget -= len(vc) + len(folded_verts)
    num_edges = graph.number_of_edges()
    if budget < 0:
        return False, set()
    if budget <= 0 and num_edges > 0:
        return False, set()
    elif num_edges == 0:
        return True, utils.unfold(vc, folded_verts)
    else:  # budget > 0 and num_edges > 0
        cur_node, max_degree = max(graph.degree_iter(), key=itemgetter(1))
        # print("branching on vertex ", cur_node, "of degree", max_degree, file=sys.stderr)
        nbrs = set(graph.neighbors_iter(cur_node))
        flip = random.random() < 0.5  # flip coin to decide branch order
        if flip:
            branches = [(nbrs, len(nbrs)), (set(), 1)]
        else:
            branches = [(set(), 1), (nbrs, len(nbrs))]
        nodes = set(graph.nodes_iter())
        for branch, budget_change in branches:
            if budget_change > budget: continue
            res, vc_new = vertex_branching(graph.subgraph(nodes - {cur_node} - branch),
                                           budget - budget_change)
            if res:
                final_vc = vc | vc_new | (branch or {cur_node})
                return True, utils.unfold(final_vc, folded_verts)
        return False, set()


def get_degree_heap(graph: nx.Graph):
    degrees = [(-d, v) for (v, d) in graph.degree_iter()]
    heapq.heapify(degrees)
    return degrees


from itertools import combinations

if __name__ == '__main__':
    start = time()
    #g = nx.random_geometric_graph(20, 0.2)
    g = nx.Graph()
    g.add_edges_from([(0,1),(1,2),(0,3),(2,3),(3,4),(2,5),(4,5)])
    print("m", g.number_of_edges())
    print(vertex_branching(g, 3))
    '''
    vc = set()
    for budget in range(1, 20):
        print("trying budget", budget)
        res, vc = vertex_branching(g, budget, -1)
        if res: break
    print(time()-start, "seconds")
    print(vc, len(vc))
    nodes = set(range(20))
    for c in combinations(nodes, len(vc)-1):
        if g.subgraph(nodes-set(c)).number_of_edges() == 0:
            print("found", c)
            break
    else:
        print("none found")
    '''