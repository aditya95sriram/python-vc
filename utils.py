#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from time import time
import itertools
from operator import itemgetter
import heapq
# import cvxopt


def read_file(file, debug=False):
    n, m = map(int, file.readline().split()[2:])
    graph = nx.Graph()
    edges = np.loadtxt(file, dtype=int, comments='c')
    graph.add_edges_from(edges)
    graph.graph['max_id'] = n+1
    graph.true_edges = m
    return graph


def read_instance(n: int, debug=False):
    fname = r"C:\Users\aditya\Desktop\academics\mtech\pace\src\instances\vc-exact_{:03d}.gr".format(n)
    with open(fname, 'r', encoding='utf8') as file:
        return read_file(file, debug=debug)


'''
def lpvc(graph: nx.Graph, solver: str = 'cvxopt'):
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    #edge_constraints = np.zeros((n, m))
    row_idx = [i for i in range(n)]
    col_idx = [i for i in range(n)] + [n+i//2 for i in range(2*m)]
    for i,(u,v) in enumerate(graph.edges):
        #edge_constraints[u-1,i] = -1
        #edge_constraints[v-1,i] = -1
        row_idx.append(int(u-1))
        row_idx.append(int(v-1))
        #col_idx.extend((i,i))
    #A = cvxopt.matrix(np.hstack((edge_constraints, -np.eye(n))).T)
    A = cvxopt.spmatrix(-1.0, col_idx, row_idx)
    #b = cvxopt.matrix(np.hstack((-np.ones(m), np.zeros(n))))
    b = cvxopt.matrix(np.hstack((np.zeros(n), -np.ones(m))))
    c = cvxopt.matrix(np.ones(n))
    #return A, b, c
    cvxopt.solvers.options['show_progress'] = False
    #cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    return cvxopt.solvers.lp(c, A, b, solver=solver)
'''


def make_double(graph: nx.Graph):
    newgraph = nx.Graph()
    offset = graph.graph['max_id']
    for u, v in graph.edges():
        newgraph.add_edge(u, v+offset)
        newgraph.add_edge(v, u+offset)
    return newgraph


def halfint_lpvc2(graph: nx.Graph):
    """
    Deprecated half integral LP solver
    """
    n = graph.number_of_nodes()
    doubled = make_double(graph)
    print("double graph created")
    mates = bipartite.maximum_matching(doubled)
    print("matching found")
    vc = bipartite.to_vertex_cover(doubled, mates)
    print("vc found")
    lpval = {v: 0.5*((v in vc) + (v+n in vc)) for v in graph.nodes_iter()}
    return lpval


def halfint_lpvc(graph: nx.Graph, debug=False):
    def log(*args):
        if debug: print(*args)
    offset = graph.graph['max_id']
    log("highest node", max(graph.nodes()), "offset", offset)
    doubled = make_double(graph)
    log("highest node", max(doubled.nodes()), "offset", offset)
    log("doubled graph created")
    mates = nx.algorithms.bipartite.maximum_matching(doubled)
    log("matching found")
    unsat = set(doubled.nodes_iter()) - mates.keys()
    vc = {i: 0 for i in graph.nodes_iter()}
    if unsat: log('phase1')
    while unsat:
        v = unsat.pop()
        # neigh = list(doubled.neighbors(v))
        neigh = doubled.neighbors(v)  # intentionally kept as list not iterator
        # vc.update(neigh)
        # unsat.update(mates[nv] for nv in neigh)  # mark mates of neighbors as unsat
        for nv in neigh:
            vc[(nv-1) % offset + 1] += 0.5
            unsat.add(mates[nv])  # mark mates of neighbors as unsat
        doubled.remove_nodes_from(neigh)
        doubled.remove_node(v)
    # print("stopped at vc:", vc, "nodes:", doubled.nodes())
    # now no unsaturated vertices left
    # graph not empty => remaining perfect matching
    if doubled.number_of_nodes() > 0:
        # if left half chosen
        log('phase2')
        for v in doubled.nodes_iter():
            if v <= offset:
                vc[(v-1) % offset + 1] += 0.5
    log("final vc:", set(vc.values()))
    log("sum:", sum(vc.values()))
    return vc


def check_lpvc(graph: nx.Graph, vc: dict):
    for u, v in graph.edges():
        if vc[u] + vc[v] < 1:
            print("failed at", u, v, vc[u], vc[v])
    return True


def collect_lpsoln(sol):
    r = {val: [] for val in [0.0, 0.5, 1.0]}
    for v, val in sol.items():
        r[val].append(v)
    return r


def count_lpsoln(sol):
    r = {val: 0 for val in [0.0,0.5,1.0]}
    for v, val in sol.items():
        r[val] += 1
    return r

def powerset(iterable, min_size=0):
    """
    Return generator of all possible subsets of given iterable

    ``powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)``

    Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    ""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r)
                                         for r in range(min_size, len(s)+1))


def first(iterable):
    return next(iter(iterable))


def greedy_vc(graph: nx.Graph):
    di = MaxDegreeIter(graph)
    vc = set()
    while di.graph.number_of_edges() > 0:
        cur_node = di.pop()
        vc.add(cur_node)
    return vc


class MaxDegreeIter(object):

    REMOVED = -1e5

    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.degrees = []
        self.entry_finder = dict()
        for v,d in graph.degree_iter():
            entry = [-d, v]
            self.degrees.append(entry)
            self.entry_finder[v] = entry
        heapq.heapify(self.degrees)

    def decrement(self, v):
        entry = self.entry_finder.pop(v)
        entry[-1] = MaxDegreeIter.REMOVED
        old_d, _ = entry
        new_entry = [old_d+1, v]
        self.entry_finder[v] = new_entry
        heapq.heappush(self.degrees, new_entry)

    def peak(self):
        d, v = self.degrees[0]
        return v

    def pop(self):
        while self.degrees:
            d, v = heapq.heappop(self.degrees)
            if v is not MaxDegreeIter.REMOVED:
                del self.entry_finder[v]
                for n in self.graph.neighbors_iter(v):
                    self.decrement(n)
                self.graph.remove_node(v)
                return v
        raise KeyError("no vertices left to pop")


def degree_hist(graph: nx.Graph):
    hist = {}
    for v,d in graph.degree_iter():
        hist[d] = hist.get(d, 0) + 1
    return hist


def unfold(vc: set, folded_verts: list):
    """
    Compute final vc given the partial vc and list of folded vertices (unreversed)
    :param vc: partial vc as a set
    :param folded_verts: list of folded vertices as 3-tuples
    :return: final vertex cover as a set
    """
    final_vc = set(vc)
    for u, v, w in folded_verts[::-1]:
        if u in final_vc:
            final_vc.remove(u)
            final_vc.add(v)
            final_vc.add(w)
        else:
            final_vc.add(u)
    return final_vc


if __name__ == '__main__':
    t = time()
    graph = read_instance(1)
    print("t = {:.2f}".format(time() - t))
    print("n:{}\tm:{}".format(graph.number_of_nodes(), graph.number_of_edges()))
    selfsol = halfint_lpvc2(graph)
    print("valid lpvc:", check_lpvc(graph, selfsol))
    print("t = {:.2f}".format(time() - t))
    solver = 'cvxopt'
    print("solver:", solver)
    # sol = lpvc(graph, solver=solver)
    print("t = {:.2f}".format(time() - t))
    # print("status:{}\tlpopt:{}\tvals:{}".format(sol['status'], np.sum(sol['x']), set(np.round(sol['x'],5)[:,0])))
    # read_file(r".\instances\vc-exact_001.hgr")
