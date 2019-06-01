#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils, branching, reduction, compression
from time import time
import networkx as nx
import sys
import os


def eprint(*args, **kwargs):
    #return
    print(*args, file=sys.stderr, **kwargs)


start = time()
def timestamp(s):
    eprint("time({}): {}".format(s, time()-start))


def min_vertex_cover(graph: nx.Graph, **kwargs):
    # apply budget-insensitive reduction rules
    # todo: maybe reduction not needed
    flag, graph, vc, folded_verts = reduction.rrhandler(graph, budget=-1)
    timestamp("reduction")
    eprint("after reduction n:", graph.number_of_nodes(), "m:", graph.number_of_edges())
    #eprint("degree hist:", utils.degree_hist(graph))

    if graph.number_of_edges() == 0:
        eprint("#"*80)
        eprint("ending after reduction")
        eprint("#"*80)
        return vc

    # use half int lp to get lpopt
    lp_soln = utils.halfint_lpvc(graph)
    lpopt = sum(lp_soln.values())
    timestamp("lpvc")
    lo, hi = int(lpopt), 2*int(lpopt)
    eprint("lo: {}, hi: {}".format(lo, hi))
    # eprint("lpvals", utils.collect_lpsoln(lp_soln))

    # nx.draw(graph, with_labels=True)
    # plt.show()

    # try greedy vc
    if 'greedy_vc' in kwargs and kwargs['greedy_vc']:
        gvc = utils.greedy_vc(graph)
        graph_ = graph.copy()
        graph_.remove_nodes_from(gvc)
        # print("gvc verts", gvc)
        eprint("greedy vc:", len(gvc))
        timestamp("greedy vc")
        hi = min(hi, len(gvc))
        if lo == hi:
            return gvc
    eprint("final range: {}-{}".format(lo, hi))

    #branching.vertex_branching(graph, 280)
    #return
    # try some initial values near lpopt
    if 'initial' in kwargs:
        for i in range(kwargs['initial']):
            eprint("trying for vc of", lo)
            res, vc = branching.vertex_branching(graph, lo)
            if res:
                return vc
            else:
                lo = lo + 1

    # binary search on final range
    vc = binsearch(graph, lo, hi)
    timestamp("binsearch end")
    return vc


def binsearch(graph: nx.Graph, lo: int, hi: int):
    vc, cache_vc = set(), set()
    res = True
    while lo <= hi:
        mid = (lo+hi)//2
        # print("###lo-mid-hi", lo, mid, hi)
        '''
        new_graph, vc, new_k, flag = reduction.rrhandler(graph, mid)
        print("after reduction, k:", new_k, "flag:", flag, "len", len(vc))
        if not flag:  # NO-instance
            lo = mid+1
            print("not flag incrementing k")
            continue
        '''
        eprint("binsearch: ({},{},{})".format(lo, mid, hi))
        timestamp("part binsearch")
        if vc:
            cache_vc = vc
            eprint("caching vc")
        res, vc = branching.vertex_branching(graph, mid)
        # print("after branching, res:", res)
        if not res:
            lo = mid+1
            eprint("not res, incrementing k")
        else:
            hi = mid-1
            eprint("res decrementing k", sorted(vc))
    eprint("final res:", res)
    if not res:  # if final run was a NO-instance then return latest YES-instance
        eprint("### using cached vc")
        return cache_vc
    return vc
# print("final vc", sorted(vc), "size", len(vc))
# status, new_vc = compression.compress(graph, vc, smart_subsets=True)
# if status:
#     print("compression succeeded", file=sys.stderr)
#     vc = new_vc
#     # print(sorted(new_vc), "(", len(new_vc), ")")
# else:
#     print("could not compress", file=sys.stderr)


def main(graph: nx.Graph, **kwargs):
    all_vc = set()
    flag, graph, new_vc, folded_verts = reduction.rrhandler(graph, -1)
    all_vc.update(new_vc)
    eprint("after first reduction n:", graph.number_of_nodes(), "m:", graph.number_of_edges())
    eprint("vc:", len(new_vc), "folded verts:", len(folded_verts))
    for component in nx.algorithms.components.connected_component_subgraphs(graph):
        cn, cm = component.number_of_nodes(), component.number_of_edges()
        eprint("working on component of size", len(component))
        if cn == cm*(cm-1)/2:  # this component is a clique
            eprint("#"*80)
            eprint("#####found clique component#######")
            eprint("#"*80)
            vc = set(component.nodes_iter())
            vc.pop()  # remove one node
        else:
            vc = min_vertex_cover(component, **kwargs)
        all_vc.update(vc)
        eprint("compressing", len(vc))
        #res, new_vc = compression.compress(component, vc)
        #if res:
        #    print("compression succeeded on component of size", len(component))
        #    print("old vc", len(vc), "new vc", len(new_vc))
        #    sys.exit()
        eprint("found component vc: {}(total:{})".format(len(vc), len(all_vc)))
    return utils.unfold(all_vc, folded_verts)


graph = utils.read_file(sys.stdin)
n = graph.number_of_nodes()
m = graph.number_of_edges()
eprint("n:", n, "m:", m)
#eprint(utils.degree_hist(graph))
vc = main(graph, greedy_vc=True)
eprint("final vc size: {}/{}".format(len(vc), n))
print("s vc {0} {1}".format(n, len(vc)))
print(*vc, sep='\n')
eprint("time:", time()-start)


if __name__ == '__main__':
    eprint("compresing...")
    res, svc = compression.compress(graph, vc)
    timestamp("compression")
    if res:
        eprint("smaller vc found, size:", len(svc))
    else:
        eprint("vc could not be compressed")
    import psutil
    import matplotlib.pyplot as plt
    # nx.draw_random(graph, with_labels=True)
    # plt.show()
    process = psutil.Process(os.getpid())
    # graph = utils.read_instance(1)
    # graph = nx.random_regular_graph(3,6000,seed=1)
    # graph = nx.Graph([(u+1,v+1) for u,v in graph.edges()])
    eprint("Memory usage: {} MB".format(process.memory_info().rss / 1e6))
