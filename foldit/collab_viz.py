import colorsys
import subprocess
import argparse
import os
import csv
import json
import logging
from itertools import groupby

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from concurrent.futures import ProcessPoolExecutor

from foldit.foldit_data import get_relevant_sids
from util import PDB_Info, get_data_value
from typing import NamedTuple, Tuple, List, Dict
import matplotlib
matplotlib.use("Agg")


def get_source(e):
    cands = e.pdl[-2::-1]
    i = 0
    while cands[i]['header']['score'] == 9999.99 or cands[i]['actions'] == {} or cands[i]['header']['uid'] == e.uid:
        i += 1
    return cands[i]


def col_to_str(col):
    return '#' + ("%0.2X%0.2X%0.2X" % (int(col[0] * 0xff), int(col[1] * 0xff), int(col[2] * 0xff)))


def is_corrupted(pdl, uid):
    return all(p['actions'] == {} or p['header']['uid'] == uid or p['header']['uid'] == '0' or p['header']['score'] == 9999.99 for p in pdl)


def remove_corrupted(pdb_infos):
    return [x for x in pdb_infos if not is_corrupted(x.pdl, x.uid)]


def render_collab(data):
    user_colors = {}
    for pid in data.pid.unique():
        print(pid)
        df = data[data.pid == pid]
        start = df.timestamps.apply(min).min()
        end = df.timestamps.apply(max).max()
        pdb_infos = df.apply(lambda r: sorted(([p for l in r.lines for p in l.pdb_infos] if r.lines else []) +
                                              ([p for l in r.evol_target_lines for p in l.pdb_infos] if r.evol_lines else []),
                                              key=lambda p: p.timestamp), axis=1)
        shared = {uid: list(pdbs) for uid, pdbs in groupby(sorted(sum(pdb_infos.map(
            lambda d: [x for x in d if int(x.sharing_gid) > 1]), []), key=lambda p: p.uid), lambda p: p.uid)}
        active_evolves = remove_corrupted(sum(pdb_infos.map(
            lambda d: [x for x in d if int(x.sharing_gid) == 0 and x.scoretype == '2' and x.pdl[-1]['actions']]), []))
        passive_evolves = remove_corrupted(sum(pdb_infos.map(
            lambda d: [x for x in d if int(x.sharing_gid) == 0 and x.scoretype == '2' and x.pdl[-1]['actions'] == {}]), []))

        # map gids to colors
        cdict = {}
        # gids = set([xs[0].gid for xs in shared.values()] + [get_source(e)['header']['gid'] for e in active_evolves + passive_evolves])
        # cm = plt.get_cmap('tab10' if len(gids) <= 10 else 'tab20')
        # for gid, c in zip(gids, cm.colors):
        #     cdict[gid] = col_to_str(c)

        groups_uids = {gid: [uid for _, uid in g] for gid, g in groupby(
            sorted(set(sum(pdb_infos.map(lambda d: [(x.gid, x.uid) for x in d if int(x.sharing_gid) > 1 or
                                                         (int(x.sharing_gid) == 0 and x.scoretype == '2' and x.pdl[-1][
                                                             'actions'])]), []))), lambda p: p[0])}
        for gid, uids in groups_uids.items():
            cdict[gid] = {}
            group_colors = user_colors.setdefault(gid, {})
            colors = [col_to_str(c) for c in list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)]
            new_uids = uids[:]
            for prev_uid, color in group_colors.items():
                colors.remove(color)
                if prev_uid in new_uids:
                    new_uids.remove(prev_uid)
                cdict[gid][prev_uid] = color
            assert len(colors) >= len(new_uids)
            for uid, c in zip(new_uids, colors):
                cdict[gid][uid] = c
                group_colors[uid] = c


        dot = Digraph(name="parent", graph_attr={'forecelabels': 'true', 'K': '0.6', 'repulsiveforce': '2'},
                      node_attr={'style': 'filled'}, edge_attr={'color': '#00000055'})
        group_clusters = {gid: Digraph(name="cluster_{}".format(gid),
                                       graph_attr={'label': "group_{}".format(gid)}) for gid in groups_uids}

        evolved = {}
        # create evolver nodes & edges
        uid_grouped = groupby(sorted(active_evolves, key=lambda e: e.uid), lambda e: e.uid) # group by evolver
        uid_source_grouped = {
            uid: {k: list(g) for k, g in
                  groupby(sorted(g, key=lambda e: (get_source(e)['header']['uid'], get_source(e)['header']['score'])),
                          lambda e: (get_source(e)['header']['uid'], get_source(e)['header']['score']))} for uid, g in
            uid_grouped} # further group by source
        active_uids = list(uid_source_grouped.keys())

        # evolver_clusters = {gid: Digraph(name="cluster_active_evolve_{}".format(gid),
        #                                  node_attr={'shape': 'oval'},
        #                                  graph_attr={'label': '{}_active_evolvers'.format(gid)})
        #                     for gid in gids}
        for uid, evolved_targets in uid_source_grouped.items():
            gid = list(evolved_targets.values())[0][0].gid
            # evolver_clusters[gid].node(uid, fillcolor=cdict[gid], label=uid)
            for target, evolves in evolved_targets.items():
                group_clusters[gid].node("{} on {}@{:.2f}".format(uid, *target), fillcolor=cdict[gid][uid], shape="oval",
                                         label="{:.2f}".format(min(e.energy for e in evolves)))
                # evoling_start = min(e.timestamp for e in evolves)
                # edge_color = colorsys.hsv_to_rgb(0.28, 0.1 + 0.9 * (evoling_start - start) / (end - start), 0.7)
                # evolving_time = sum(get_sessions([e.timestamp for e in evolves])
                group_clusters[gid].edge("{} on {}@{:.2f}".format(uid, *target), "{}@{:.2f}".format(*target),
                         penwidth=str(0.2 + np.log10(len(evolves))),
                         style="dashed" if min(e.energy for e in evolves) >= target[1] else "solid")
                evolved["{}@{:.2f}".format(*target)] = True
        # for sg in evolver_clusters.values():
        #     dot.subgraph(sg)

        # do it again, this time for people who just loaded in a shared solution but didn't do anything
        # uid_grouped = groupby(sorted(passive_evolves, key=lambda e: e.uid), lambda e: e.uid)
        # uid_source_grouped = {
        #     uid: {k: min(g, key=lambda p: p.energy) for k, g in
        #           groupby(sorted(g, key=lambda e: (get_source(e)['header']['uid'], get_source(e)['header']['score'])),
        #                   lambda e: (get_source(e)['header']['uid'], get_source(e)['header']['score']))} for uid, g in
        #     uid_grouped if uid not in active_uids} # screen out anyone who later actively evolved
        #
        # evolver_clusters = {gid: Digraph(name="cluster_passive_ evolve_{}".format(gid),
        #                                  node_attr={'shape': 'oval'},
        #                                  graph_attr={'label': '{}_passive_evolvers'.format(gid)})
        #                     for gid in gids}
        # for uid, evolved_targets in uid_source_grouped.items():
        #     gid = list(evolved_targets.values())[0].gid
        #     evolver_clusters[gid].node(uid, fillcolor=cdict[gid], label=uid)
        #     for target, evolve in evolved_targets.items():
        #         dot.edge(uid, "{}@{:.2f}".format(*target), penwidth='3', style='dashed')
        #         evolved["{}@{:.2f}".format(*target)] = True
        # for sg in evolver_clusters.values():
        #     dot.subgraph(sg)

        # nodes and edges for shared solutions
        for uid, pdbs in shared.items():
            gid = pdbs[0].gid
            for p in pdbs:
                if p.scoretype == '2' and not is_corrupted(p.pdl, p.uid):
                    source = get_source(p)
                    # edge_color = colorsys.hsv_to_rgb(0.28, 0.1 + 0.9 * (p.timestamp - start) / (end - start), 0.7)
                    group_clusters[gid].edge("{}@{:.2f}".format(uid, p.energy),
                             "{}@{:.2f}".format(source['header']['uid'], source['header']['score']),
                             penwidth='3')
                    evolved.setdefault("{}@{:.2f}".format(uid, p.energy), False)
                    evolved["{}@{:.2f}".format(source['header']['uid'], source['header']['score'])] = True
        for uid, pdbs in shared.items():
            gid = pdbs[0].gid
            num_ignored = len([p for p in pdbs if "{}@{:.2f}".format(uid, p.energy) not in evolved])
            # with dot.subgraph(name="cluster_{}".format(uid),
            #                   graph_attr={'label': "{}_shared ({} ignored)".format(uid, num_ignored), 'forcelabels': 'true',
            #                               'style': 'filled', 'fillcolor': cdict[pdbs[0].gid]},
            #                   node_attr={'style': 'filled'}) as c:
            for p in pdbs:
                if "{}@{:.2f}".format(uid, p.energy) in evolved:
                    shape = "box" if p.scoretype == '1' or is_corrupted(p.pdl, p.uid) else "diamond"
                    # c.node("{}@{:.2f}".format(uid, p.energy), label="{:.2f}".format(p.energy), shape=shape,
                    group_clusters[gid].node("{}@{:.2f}".format(uid, p.energy), label="{:.2f}".format(p.energy), shape=shape,
                                             style='filled,solid' if evolved["{}@{:.2f}".format(uid, p.energy)] else 'filled,dashed',
                                             fillcolor=cdict[gid][uid])
                             # color="#ffffff")
        for cluster in group_clusters.values():
            dot.subgraph(cluster)

        # output raw source, then use command line graphviz tools to fix cluster layout
        outname = "collab_viz/collab_{}".format(pid)
        with open(outname, 'w') as out:
            out.write(dot.source)
        subprocess.run(
            "ccomps -xC {} | dot | gvpack -array_c{} | neato -Tpng -n2 -o {}.png".format(outname, len(groups_uids)+1, outname),
            shell=True, check=True)


class ShareTag(NamedTuple):
    uid: str
    energy: float


class Collaborator(NamedTuple):
    uid: str
    gid: str
    pdbs: List[PDB_Info]
    energy_comps: Dict[str, float]
    tag: ShareTag
    parent: "Collaborator"
    source: ShareTag
    children: List["Collaborator"]


def get_tag(s: PDB_Info) -> ShareTag:
    return ShareTag(s.uid, round(s.energy, 4))


def get_source_tag(s: PDB_Info) -> ShareTag:
    source = get_source(s)
    return ShareTag(source['header']['uid'], round(source['header']['score'], 4))


def get_evolver_uid(s: PDB_Info, lines: list) -> str:
    if s.scoretype == "1":
        return s.uid
    for i, line in enumerate(lines):
        sids = [p.sid for p in line.pdb_infos]
        if s.sid in sids:
            return s.uid + "evol" + str(i)
    raise ValueError("evolver pdb {} not found in any evolver lines for {}".format(s.sid, (s.uid, s.pid)))


def get_collab_children(root_tag: ShareTag, collab: Collaborator, evolves_by_source: dict) -> List[Collaborator]:
    children = []
    for uid, pdbs in evolves_by_source[root_tag].items():
        best = min(pdbs, key=lambda p: p.energy)
        child = Collaborator(uid, collab.gid, pdbs, {c.name: c.energy * c.weight for c in best.energy_components},
                             ShareTag(uid, round(best.energy, 4)), collab, root_tag, [])
        children.append(child)
        for pdb in pdbs:
            if get_tag(pdb) in evolves_by_source and int(pdb.sharing_gid) > 1:
                child.children.extend(get_collab_children(get_tag(pdb), child, evolves_by_source))
    return children


def get_team_structures(data, soln_lookup, child_lookup):
    collabs = {}
    for pid in data.pid.unique():
        logging.debug("getting team structures for {}".format(pid))
        df = data[data.pid == pid]
        pdb_infos = df.apply(lambda r: sorted(([p for l in r.lines for p in l.pdb_infos] if r.lines else []) +
                                              ([p for l in r.evol_target_lines for p in l.pdb_infos] if r.evol_lines else []),
                                              key=lambda p: p.timestamp), axis=1)

        evol_lines_lookup = {uid: get_data_value(uid, pid, "evol_target_lines", df) for uid in df.uid}
        euid_lookup = {pdb.sid: get_evolver_uid(pdb, evol_lines_lookup[pdb.uid]) for pdb in sum(pdb_infos.values, [])}

        shared = {uid: list(pdbs) for uid, pdbs in groupby(sorted(sum(pdb_infos.map(
            lambda d: [x for x in d if int(x.sharing_gid) > 1]), []), key=lambda p: euid_lookup[p.sid]),
                                                           lambda p: euid_lookup[p.sid])}
        active_evolves = remove_corrupted(sum(pdb_infos.map(
            lambda d: [x for x in d if x.scoretype == '2' and x.pdl[-1]['actions']]), []))

        uid_grouped = groupby(sorted(active_evolves, key=lambda e: euid_lookup[e.sid]),
                              lambda e: euid_lookup[e.sid]) # group by evolver
        uid_source_grouped = {
            uid: {k: list(g) for k, g in
                  groupby(sorted(g, key=lambda e: get_source_tag(e)), lambda e: get_source_tag(e))} for uid, g in
            uid_grouped} # further group by source
        evolved_targets = {target for targets in uid_source_grouped.values() for target in targets}

        roots = sum(([p for p in pdbs if p.scoretype == '1' and get_tag(p) in evolved_targets] for pdbs in shared.values()), [])

        evolves_by_source = {tag: list(pdbs) for tag, pdbs in groupby(sorted(active_evolves, key=lambda s: get_source_tag(s)),
                                                                      lambda s: get_source_tag(s))}
        evolves_by_source = {tag: {uid: list(pdbs) for uid, pdbs in groupby(sorted(xs, key=lambda x: euid_lookup[x.sid]),
                                                                            lambda x: euid_lookup[x.sid])}
                             for tag, xs in evolves_by_source.items()}

        collabs[pid] = []
        for root in roots:
            tag = get_tag(root)
            sids = get_relevant_sids(root, soln_lookup, child_lookup)
            collab = Collaborator(tag.uid, root.gid, [soln_lookup[sid] for sid in sids] if sids else [],
                                  {c.name: c.energy * c.weight for c in root.energy_components}, tag, None, None, [])
            collab.children.extend(get_collab_children(tag, collab, evolves_by_source))
            collabs[pid].append(collab)

    return collabs

# for collab in sorted(collabs['2003642'], key=lambda c: c.pdbs[0].gid):
#     print(collab.pdbs[0].gid)
#     print(collab.tag)
#     front = [(1, c) for c in collab.children]
#     while len(front) > 0:
#         ntab, cur = front.pop()
#         print("    "*ntab, cur.tag)
#         front.extend([(ntab + 1, c) for c in cur.children])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='collab_viz.py')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('pids', nargs='+')
    args = parser.parse_args()

    if args.debug:
        for pid in args.pids:
            render_collab(pid)
    else:
        with ProcessPoolExecutor(30) as pool:
            pool.map(render_collab, args.pids, chunksize=1)
