#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import utils


def print(*args):  # mute printing in this module
    pass


def degreedel(graph: nx.Graph, k: int):
    """
    Degree 0 and degree greater than k vertices (RR1)

    :param graph: graph to reduce
    :param k: parameter k
    :return: 4-tuple (flag, changed, new_graph, vc)
    where flag=false denotes that this is a NO-instance and vice-versa,
    changed indicates if the reduction rule caused any change,
    new_graph is the reduced subgraph,
    vc is the partial vc constructed while applying reduction rules
    """
    n = graph.number_of_nodes()
    # print('n:', n)
    # print('V',v)
    degree_dir = graph.degree_iter()
    # print('degree_dir', degree_dir)

    to_delete = set()  # delete degree 0
    vc = set()  # add degree>k to vc

    for v, deg in degree_dir:
        if deg == 0:
            to_delete.add(v)
        if deg > k:
            vc.add(v)

    # handling high degree vertices
    len_vc = len(vc)
    #graph.remove_nodes_from(deg_large)
    if len_vc > k:  # more than k high-degree => NO-instance
        return False, False, graph, set()

    # print('VC-del:', vc)

    # repeatedly applying reduction rules
    if len_vc + len(to_delete) == 0:  # no improvement in this case
        return True, False, graph, vc
    else:
        nodes = set(graph.nodes_iter())
        new_graph = graph.subgraph(nodes - vc - to_delete)
        #graph_new, vc_small, k, flag = degreedel(graph, k)
        # print('vc_small, vc_new', vc_small,  vc_new)
        #c = vc.union(vc_small)

        # print('c', c)
        #graph = graph_new
        return True, True, new_graph, vc

'''
def degreeone(graph: nx.Graph, k: int):
    """
    Degree 1 vertices (RR2)

    :param graph: graph to reduce
    :param k: parameter k
    :return: 4-tuple (flag, changed, new_graph, vc)
    where flag=false denotes that this is a NO-instance and vice-versa,
    changed indicates if the reduction rule caused any change,
    new_graph is the reduced subgraph,
    vc is the partial vc constructed while applying reduction rules
    """
    n = graph.number_of_nodes()
    print('n:', n)
    nodes = graph.nodes()
    nodes = set(nodes)
    degree_dir = graph.degree
    print('degree_dir', degree_dir)

    deg_1 = set()
    vc = set()
    to_delete = set()
    len_vc = 0
    flag = True

    for v, deg in degree_dir:
        if deg == 1:
            deg_1.add(v)

    #v = set(v)
    # handling degree1 vertices
    for i in deg_1:
        if i not in nodes:
            continue
        n = list(graph.neighbors(i))
        if len(n) == 0:
            to_delete.add(i)
            #graph.remove_node(i)
            continue

        # if n[0] in vc_new:  #this thing and if condition just above is same
        # hence no need of this, this causes a trouble for empty n
        #  graph.remove_node(i)
        #  v.remove(i)
        else:
            vc.add(n[0])
            len_vc += 1

            if len_vc > k:
                return False, False, graph, set()

            to_delete =
            graph.remove_node(n[0])
            graph.remove_node(i)
            nodes.remove(i)
            nodes.remove(n[0])

    k -= len(vc_new)
    print('VC-one:', vc_new)

    if len(deg_1) == 0:  # no improvement this time
        return graph, vc_new, k, flag
    else:
        graph_new, vc_small, k, flag = degreeone(graph, k)
        print('vc_small, vc_new', vc_small, vc_new)
        vc = vc_new.union(vc_small)
        print('vc-here:', vc)
        # print('c', c)
        graph = graph_new
        return graph, vc, k, flag

def rrhandler(graph: nx.Graph, k: int):
    """
    Handles application of reduction rules (both count and order).
    Doesn't take already constructed vc as part of input and
    thus the input graph must be cleaned with respect to the already
    known vc

    :param graph: graph to reduce
    :param k: parameter k
    :return: reduced 4-tuple (new_graph, vc, k, flag)
    where new_graph is the reduced graph
    vc is the partial vc constructed during reduction rules
    k is the new value of k after the reduction rules
    flag, if false denotes that this is a NO-instance
    """
    n = graph.number_of_nodes()

    graph, vc_1, k, flag = degreedel(graph, k)
    if not flag:
        return graph, vc_1, k, flag
    graph, vc_2, k, flag = degreeone(graph, k)
    if not flag:
        return graph, vc_2, k, flag

    vc = vc_1.union(vc_2)
    print('handler-union:', vc)
    k -= len(vc)

    n_red = graph.number_of_nodes()
    if n == n_red:
        return graph, vc, k, flag
    else:
        graph, vc_new, k, flag = rrhandler(graph, k)
        c = vc.union(vc_new)
        return graph, c, k, flag
'''


def degreeone(graph: nx.Graph, k: int = -1):
    """
    Degree 1 vertices (RR2)

    :param graph: graph to reduce
    :param k: parameter k
    :return: 4-tuple (flag, changed, new_graph, vc)
    where flag=false denotes that this is a NO-instance and vice-versa,
    changed indicates if the reduction rule caused any change,
    new_graph is the reduced subgraph,
    vc is the partial vc constructed while applying reduction rules
    """
    deg_one = set()
    for v, deg in graph.degree_iter():
        if deg == 1:
            deg_one.add(v)

    if len(deg_one) == 0:  # no degree 1 vertices found, so no change
        return True, False, graph, set()

    vc = set()
    to_delete = set()
    while deg_one:
        v = deg_one.pop()
        if v in vc:
            continue
        neigh = utils.first(graph.neighbors_iter(v))
        to_delete.add(v)
        vc.add(neigh)
        if len(vc) > k:  # NO-instance
            return False, False, graph, set()
    nodes = set(graph.nodes_iter())
    new_graph = graph.subgraph(nodes - vc - to_delete)
    return True, True, new_graph, vc


def degree_k(graph: nx.Graph, budget: int):
    """
    Delete high-degree (degre > budget) vertices

    :param graph: copy of graph to reduce
    :param budget: parameter k
    :return: (new_graph, vc)
    """
    vc = set()
    for v,d in graph.degree_iter():
        if d > budget:
            vc.add(v)
    graph.remove_nodes_from(vc)
    return graph, vc


def sage_reduction(graph: nx.Graph):
    """

    :param graph: copy of graph to reduce
    :return: (new_graph, vc, folded_verts)
    """
    vc = []
    folded_vertices = []
    degree_at_most_three = {u for u in graph if graph.degree(u) <= 3}

    while degree_at_most_three:

        u = degree_at_most_three.pop()
        du = graph.degree(u)

        if not du:
            # RULE 1: isolated vertices are not part of the cover. We
            # simply remove them from the graph. The degree of such
            # vertices may have been reduced to 0 while applying other
            # reduction rules
            graph.remove_node(u)

        elif du == 1:
            # RULE 2: If a vertex u has degree 1, we select its neighbor
            # v and remove both u and v from g.
            v = next(graph.neighbors_iter(u))
            vc.append(v)
            graph.remove_node(u)

            for w in graph.neighbors_iter(v):
                if graph.degree(w) <= 4:
                    # The degree of w will be at most three after the
                    # deletion of v
                    degree_at_most_three.add(w)

            graph.remove_node(v)
            degree_at_most_three.discard(v)

        elif du == 2:
            v, w = graph.neighbors(u)

            if graph.has_edge(v, w):
                # RULE 3: If the neighbors v and w of a degree 2 vertex
                # u are incident, then we select both v and w and remove
                # u, v, and w from g.
                vc.append(v)
                vc.append(w)
                graph.remove_node(u)
                neigh = set(graph.neighbors(v) + graph.neighbors(w)).difference([v, w])
                graph.remove_node(v)
                graph.remove_node(w)

                for z in neigh:
                    if graph.degree(z) <= 3:
                        degree_at_most_three.add(z)

            else:
                # RULE 4, folded vertices: If the neighbors v and w of a
                # degree 2 vertex u are not incident, then we contract
                # edges (u, v), (u, w). Then, if the solution contains u,
                # we replace it with v and w. Otherwise, we let u in the
                # solution.
                neigh = set(graph.neighbors(v) + graph.neighbors(w)).difference([u, v, w])
                graph.remove_node(v)
                graph.remove_node(w)
                for z in neigh:
                    graph.add_edge(u, z)

                folded_vertices.append((u, v, w))

                if graph.degree(u) <= 3:
                    degree_at_most_three.add(u)

            degree_at_most_three.discard(v)
            degree_at_most_three.discard(w)

        elif du == 3:
            v, w, x = graph.neighbors(u)

            if graph.has_edge(v,w) and graph.has_edge(w,x) and graph.has_edge(v,x):
                # RULE 5, similar to RULE 3 except it looks at 4-cliques
                # instead of 3-cliques
                #print("degree 3 RR applied")
                vc.append(v)
                vc.append(w)
                vc.append(x)
                graph.remove_node(u)
                neigh = set(graph.neighbors(v) + graph.neighbors(w) + graph.neighbors(x)).difference([v, w, x])
                graph.remove_node(v)
                graph.remove_node(w)
                graph.remove_node(x)

                for z in neigh:
                    if graph.degree(z) <= 3:
                        degree_at_most_three.add(z)

                degree_at_most_three.discard(v)
                degree_at_most_three.discard(w)
                degree_at_most_three.discard(x)

    return graph, set(vc), folded_vertices


def lp_reduction(graph: nx.Graph):
    """
    Delete high-degree (degre > budget) vertices

    :param graph: copy of graph to reduce
    :param budget: parameter k
    :return: (new_graph, vc)
    """
    lpsoln = utils.halfint_lpvc(graph)
    num_half = 0
    lpvals = set(lpsoln.values())
    if 1 in lpvals or 0 in lpvals:
        print("#"*80)
        print(set(lpsoln.values()))
        print("#"*80)
    vc = set()
    for u, lpval in lpsoln.items():
        if lpval <= 0.25:
            graph.remove_node(u)
        elif lpval >= 0.75:
            vc.add(u)
            graph.remove_node(u)
        else:
            num_half += 1
    #if num_half == len(lpsoln):
    #    print("all half", check_all_half_lp(graph, lpsoln))
    #    sys.exit()
    return graph, vc, sum(lpsoln.values())


def check_all_half_lp(graph: nx.Graph, old_lpopt):
    nodes = set(graph.nodes_iter())
    for u in graph.nodes_iter():
        lpsoln = utils.halfint_lpvc(graph.subgraph(nodes - {u}))
        lpopt = sum(lpsoln.values())
        if lpopt <= old_lpopt - 1:
            return False
    return True


def rrhandler(graph: nx.Graph, budget: int):
    """
    Handles application of reduction rules (both count and order).
    Doesn't take already constructed vc as part of input and
    thus the input graph must be cleaned with respect to the already
    known vc

    :param graph: graph to reduce
    :param budget: parameter k
    :return: 4-tuple (flag, new_graph, vc, folded_verts)
    where flag=false denotes that this is a NO-instance and vice-versa,
    new_graph is the reduced subgraph,
    vc is the partial vc constructed while applying reduction rules
    folded_verts is list of folded_verts
    """
    vc = set()
    folded_verts = []
    budget_active = (budget >= 0)
    changed = True
    while changed:
        if graph.number_of_edges() == 0:
            return True, graph, vc, folded_verts

        changed = False

        # apply sage reduction (RR1-4)
        old_verts = graph.number_of_nodes()
        graph, new_vc, new_folded_verts = sage_reduction(graph.copy())
        # noinspection PyChainedComparisons
        if budget_active and len(new_vc)+len(new_folded_verts) > budget:
            return False, graph, set(), []
        if graph.number_of_nodes() < old_verts:  # some change
            folded_verts.extend(new_folded_verts)
            vc.update(new_vc)
            if budget_active:
                budget -= (len(new_vc)+len(new_folded_verts))
            changed = True
            continue

        # apply lp based crown reduction rule
        graph, new_vc, lpopt = lp_reduction(graph.copy())
        if budget_active and len(new_vc) > budget:
            return False, graph, set(), []
        if graph.number_of_nodes() < old_verts:  # some change
            print("#"*80)
            print("lp crown found")
            print("#"*80)
            vc.update(new_vc)
            if budget_active:
                budget -= len(new_vc)
            changed = True
            continue
        else:
            if budget_active:
                #print("lpopt", lpopt, "budget", budget)
                if lpopt > budget:
                    return False, graph, set(), []
            #print("all half", check_all_half_lp(graph, lpopt))

        # apply degree k reduction rule only if positive budget
        if budget_active:
            graph, new_vc = degree_k(graph.copy(), budget)
            if len(new_vc) > budget:
                return False, graph, set(), []
            if len(new_vc) > 0:
                vc.update(new_vc)
                budget -= len(new_vc)
                changed = True
                continue

    return True, graph, vc, folded_verts
