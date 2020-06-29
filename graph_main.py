from __future__ import print_function, division

import sys, csv, logging, argparse, subprocess, os.path, pickle, json, colorsys
from typing import Tuple, Dict

import numpy as np
from operator import itemgetter
from functools import partial

from graphviz import Digraph
import Bio.PDB
import matplotlib.pyplot as plt
import editdistance

from Node import Node
from graph_analysis import *
from RawNode import RawNode, initialize
from util import get_ranks, weighted_rmsd, get_atoms

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
sys.setrecursionlimit(20000)


def datapath(postfix, pid):
    return "data/puzzle_solutions/solution_{}/{}_{}".format(pid, pid, postfix)


def col_to_str(col):
    return '#' + ("%0.2X%0.2X%0.2X" % (int(col[0] * 0xff), int(col[1] * 0xff), int(col[2] * 0xff)))


def interpolate_color(scale, colr1, colr2):
    def interp_val(v, v1, v2):
        return v1 + (v2 - v1) * v

    r = interp_val(scale, colr1[0], colr2[0])
    g = interp_val(scale, colr1[1], colr2[1])
    b = interp_val(scale, colr1[2], colr2[2])

    return (r, g, b)


def interpolate_values_color(val, mini, maxi, colr1, colr2):
    clamped = max(min(val, maxi), mini)
    scale = (clamped - mini) / max((maxi - mini), 1)
    return interpolate_color(scale, colr1, colr2)


ROOT_COLOR = (0, 0, 0)
BAD_COLOR = (1, 0, 0)
GOOD_COLOR = (0, 0, 1)
GOOD_COLOR_LIGHT = (0.6, 0.6, 1)

GDT_MAX_COLOR = (1, 1, 1)
GDT_MIN_COLOR = (0, 0, 0)


def recurse_print_tree(node, depth):
    indent = ''.join([' ' for i in range(depth)])
    print(indent + node.nid[0][0:8] + " " + (str(node.data['uid']) if node.data != None else ''))
    children = node.get_relevant_children()
    for child in children:
        recurse_print_tree(child, depth + 1)


def format_mrid(mrid):
    return mrid_to_mid[mrid] if mrid in mrid_to_mid else "*{}*".format(mrid)


def output_tree_helper(node, dot, uid_to_cluster, depth, nodes, minE, maxE, minT, maxT, node_to_color=None):
    indent = ''.join([' ' for _ in range(depth)])
    logging.debug(indent + node.nid[0][0:8] + " " + (str(node.data['energy']) if node.data != None else ''))
    fill_color = GDT_MIN_COLOR  # interpolate_values_color(0, 0, 13000, BAD_COLOR, GOOD_COLOR)
    tt = 'No solution data available'
    label = ""
    if node.data != None:
        if node_to_color:
            fill_color = node_to_color(node)
        else:
            fill_color = interpolate_values_color(-node.data['energy'], minE, maxE, BAD_COLOR, GOOD_COLOR)
            label = "\n".join(["{} {}".format(format_mrid(k), v) for k, v in node.get_new_macros().items()])
            if node.has_manual_action():
                if len(label) > 0:
                    label += "\n"
                label += "MANUAL"
        tt = str(node.data['energy'])
        nodes.append(node)
    if hasattr(node, 'cascade'):
        if len(label) > 0:
            label += "\n"
        mean_factor = np.mean([len(x) for x in node.cascade])
        label += "CASCADE {} by {:.2f}".format(len(node.cascade), mean_factor)
        uid_to_cluster[node.uid].node(node.get_name(), fillcolor=col_to_str(fill_color), tooltip=tt, label=label,
                                     forcelabels="true", shape='box', fontsize='32')
    else:
        uid_to_cluster[node.uid].node(node.get_name(), fillcolor=col_to_str(fill_color), tooltip=tt, label="",
                                     forcelabels="true")

    for child in node.children:
        output_tree_helper(child, dot, uid_to_cluster, depth + 1, nodes, minE, maxE, minT, maxT, node_to_color)

        delta = 0
        # eng_dist = None
        wrmsd = None
        if node.data is not None and child.data is not None:
            delta = node.data['energy'] - child.data['energy']
            eng_dist = sum((y[1] * y[2] - x[1] * x[2]) ** 2 for x, y in zip(json.loads(node.data['energies']),
                                                                            json.loads(child.data['energies']))) ** 0.5
            # wrmsd = weighted_rmsd(get_atoms(node.data), get_atoms(child.data))

        edge_color = (0, 0, 0)
        edge_width = 6
        if child.data != None:
            edge_color = colorsys.hsv_to_rgb(0.28, 0.1 + 0.9 * (child.data['timestamp'] - minT) / (maxT - minT), 0.7)
            edge_width = editdistance.eval(node.data['seq'], child.data['seq']) if node.data else 6
        dot.edge(node.get_name(), child.get_name(), color=col_to_str(edge_color) if edge_width > 0 else "#ff0000",
                 penwidth=str(edge_width) if edge_width > 0 else "6", arrowhead='none', fontcolor="#000000",
                 tooltip=str(delta))


def output_tree(node, rank, uid, pid, color_by_seq=False):
    print("building graph for {} (rank {})".format(uid, rank))
    dot = Digraph(comment='Player Solution Graph', format='pdf',
                  node_attr={'fontname': 'Helvetica',
                             'shape': 'hexagon',
                             'fontcolor': '#ffffff',
                             'style': 'filled'},
                  edge_attr={'color': 'black',
                             'arrowhead': 'open',
                             'fontname': 'Courier',
                             'fontsize': '12',
                             'fontcolor': 'white'})

    nodes = []
    ds = [x for x in node.get_descendants() if x.data]
    minE = max(0, np.percentile([-x.data['energy'] for x in ds], 40)) if len(ds) > 0 else 0
    maxE = max(-x.data['energy'] for x in ds) if len(ds) > 0 else 30
    minT = min(x.data['timestamp'] for x in ds) if len(ds) > 0 else 0
    maxT = max(x.data['timestamp'] for x in ds) if len(ds) > 0 else 1
    uids = {x.uid for x in ds}
    print("tree has {} solvers".format(len(uids)))
    uid_to_cluster = {node.get_soloist_uid(): dot}
    if len(uids) > 1:
        cm = plt.get_cmap('tab10' if len(uids) <= 10 else 'tab20')
        # uid_to_cluster = {uid: Digraph(name="cluster_{}".format(uid),
        #                                graph_attr={'style': 'filled', 'color': col_to_str(c)}) for uid, c in zip(uids, cm.colors)}
        uid_to_cluster = {uid: dot for uid in uids}
        uid_to_color = {uid: c for uid, c in zip(uids, cm.colors)}
        def get_uid_color(node):
            return uid_to_color[node.uid]

    if color_by_seq:
        colored_seqs = [s for s,_ in Counter([d.data['seq'] for d in ds]).most_common(10)]
        cm = plt.get_cmap('tab10')
        cdict = {s: col for s, col in zip(colored_seqs, cm.colors)}
        def get_seq_color(node):
            if node.data['seq'] not in cdict:
                return (1, 1, 0.53)
            return cdict[node.data['seq']]
        output_tree_helper(node, dot, uid_to_cluster, 1, nodes, minE, maxE, minT, maxT, get_seq_color)
    else:
        output_tree_helper(node, dot, uid_to_cluster, 1, nodes, minE, maxE, minT, maxT)

    # if len(uids) > 1:
    #     for cluster in uid_to_cluster.values():
    #         dot.subgraph(cluster)

    dot.node(min([n for n in nodes if n is not None], key=lambda n: n.data['energy']).get_name(),
             fillcolor="#F2EC29", shape='star', width='2', height='2')

    print("rendering {} nodes...".format(len(nodes)))
    dot.render('results/{}/{}_rank{}_{}.gv'.format(pid, pid, rank, uid))
    return nodes


def get_rank(node, pid):
    return [x['uid'] for x in puzzles[pid]['soloists']].index(node.get_soloist_uid())


def get_node(uid, include_evolvers=False):
    dummy_root = RawNode(('00000000-0000-0000-0000-000000000000', 0), None)
    dummy_root.uid = uid
    node = Node(dummy_root, include_evolvers)
    # prune useless children that no data and no children
    node.children = [c for c in node.children if c.get_nearest_data(provide_dummy=False)]
    if len(node.children) == 1:
        node = node.children[0]
    return node


def get_top_soloists(ranks, n, pid):
    print("gathering ranks {} soloist nodes".format(ranks))
    top_uids = [x['uid'] for x in puzzles[pid]['soloists'][ranks]]
    # top_nodes_raw = [x for x in n.get_merged_branching_children() if x.get_soloist_uid() in top_uids]
    # top_nodes = [min(g, key=lambda n: n.get_min_energy(True)) for _, g in
    #              groupby(sorted(top_nodes_raw, key=lambda n: n.get_soloist_uid()), lambda n: n.get_soloist_uid())]
    # return [Node(t) for t in top_nodes]
    top_nodes = []
    for uid in top_uids:
        top_nodes.append(get_node(uid))
    return top_nodes


def load_raw_data(pid: str) -> Tuple[Dict, Dict]:
    if os.path.isfile(datapath("hist.pickle", pid)):
        print("loading history")
        with open(datapath("hist.pickle", pid), 'rb') as fp:
            history = pickle.load(fp)
    else:
        with open(datapath("hist.csv", pid)) as fp:
            n = int(subprocess.check_output(["wc", "-l"], stdin=fp))
        with open(datapath("hist.csv", pid)) as fp:
            c = csv.DictReader(fp, fieldnames=["pid", "uuid", "count", "parent_uuid", "parent_count"])
            history = {}
            i = 0
            print("loading history")
            for r in c:
                print(i, "out of", n, '\r', end="")
                key = (r['parent_uuid'], int(r['parent_count']))
                if key not in history:
                    history[key] = []
                r['count'] = int(r['count'])
                r['parent_count'] = int(r['parent_count'])
                history[key].append(r)
                i += 1
        with open(datapath("hist.pickle", pid), 'wb') as fp:
            print("pickling history")
            pickle.dump(history, fp)
        print()

    if os.path.isfile(datapath("soln.pickle", pid)):
        print("loading solution data")
        with open(datapath("soln.pickle", pid), 'rb') as fp:
            solutions = pickle.load(fp)
    else:
        with open(datapath("soln.csv", pid)) as fp:
            n = int(subprocess.check_output(["wc", "-l"], stdin=fp))
        with open(datapath("soln.csv", pid)) as fp:
            print("loading solution data")
            c = csv.DictReader(fp,
                               fieldnames=["pid", "sid", "uuid", "count", "uid", "gid", "timestamp", "energy", "rmsd",
                                           "gdt", "pdl", "ca", "residues", "energies", "ss", "seq", "sharing_gid", "type",
                                           "scoretype", "filters"])
            solutions = {}
            i = 0
            for r in c:
                print(i, "out of", n, '\r', end="")
                r['pdl'] = json.loads(r['pdl'])
                r['energy'] = float(r['energy'])
                r['timestamp'] = int(r['timestamp'])
                r['ss'] = r['ss'].strip('"')
                r['seq'] = r['seq'].strip('"')
                for p in r['pdl']:  # convert dicts to Counters to make comparisons much more convenient
                    p['actions'] = Counter(p['actions'])
                    p['macros'] = Counter(p['macros'])
                    p['macros'].pop('0', None)  # clear out the dummy macros
                solutions[(r['uuid'], int(r['count']))] = r
                i += 1
        with open(datapath("soln.pickle", pid), 'wb') as fp:
            print("pickling solution data")
            pickle.dump(solutions, fp)
        print()

    return history, solutions


puzzles = get_ranks("data/rprp_puzzle_ranks_v3")

print("loading macro data")
with open("data/macro_families.json") as fp:
    macro_families = [frozenset(f) for f in json.load(fp)]
with open("data/rprp_macros.csv") as fp:
    mrid_to_mid_raw = []
    mrid_to_shared_raw = []
    for row in csv.DictReader(fp):
        mrid_to_mid_raw.append((row['mrid'], row['mid']))
        mrid_to_shared_raw.append((row['mrid'], row['shared']))
with open('data/rprp_macro_revisions.csv') as fp:
    c = csv.DictReader(fp)
    revisions = {row['mrid']: row for row in c}
prelim = dict(mrid_to_mid_raw)
mrid_to_mid_raw.extend((k, v['mid']) for k, v in revisions.items() if k not in prelim)
prelim = dict(mrid_to_shared_raw)
mrid_to_shared_raw.extend((k, v['shared']) for k, v in revisions.items() if k not in prelim)
mrid_to_mid = dict(mrid_to_mid_raw)
mrid_to_shared = dict(mrid_to_shared_raw)
mid_to_mrid = {k: list(map(itemgetter(0), g)) for k, g in
               groupby(sorted(mrid_to_mid.items(), key=itemgetter(1)), itemgetter(1))}

parser = argparse.ArgumentParser(prog='solution_graph_aaron.py')
parser.add_argument('--viz', action='store_true')
parser.add_argument('-P', '--produce-data', action='store_true')
parser.add_argument('-L', '--load-raw', action='store_true')
parser.add_argument('-E', '--evolver', action='store_true')
parser.add_argument('--uid')
parser.add_argument('pids', nargs='+')
args = parser.parse_args()
print("processing the following pids: {}".format(args.pids))
if args.viz:
    print("visualizations will be generated")

for pid in args.pids:
    print("puzzle {}".format(pid))

    if os.path.isfile(datapath("nodes.pickle", pid)) and \
            os.path.getmtime(datapath("nodes.pickle", pid)) > os.path.getmtime("Node.py") and \
            not args.uid:
        with open(datapath("nodes.pickle", pid), 'rb') as fp:
            print("loading nodes")
            nodes = pickle.load(fp)
            top_nodes = sorted(nodes[:15], key=partial(get_rank, pid=pid))
            bottom_nodes = sorted(nodes[15:], key=partial(get_rank, pid=pid))
            if args.viz:
                for t in top_nodes + bottom_nodes:
                    output_tree(t, get_rank(t, pid), t.get_soloist_uid(), pid)
            if args.load_raw:
                print("force loading raw")
                history, solutions = load_raw_data(pid)
                initialize(history, solutions)
                root = RawNode(('00000000-0000-0000-0000-000000000000', 0), None)
    else:
        history, solutions = load_raw_data(pid)
        print("generating nodes")
        initialize(history, solutions)
        root = RawNode(('00000000-0000-0000-0000-000000000000', 0), None)
        # some puzzles somehow have solution data for the global root, so we'll clear it as a hack to make things work
        root.uid = None
        root.data = None
        if args.uid:
            target_nodes = [get_node(args.uid, args.evolver)]
        else:
            top_nodes = sorted(get_top_soloists(slice(0, 15), root, pid), key=partial(get_rank, pid=pid))
            bottom_nodes = sorted(get_top_soloists(slice(35, 50), root, pid), key=partial(get_rank, pid=pid))
            target_nodes = top_nodes + bottom_nodes

        print("generating graphs")
        print("cleaning graphs")
        # with open(datapath("summarization.csv", pid), 'w') as fp:
        #     summ = csv.DictWriter(fp, fieldnames=['pid', 'uid', 'raw', 'initial', 'final'])
        for t in target_nodes:
            # output_tree(t, get_rank(t, pid), t.get_soloist_uid(), pid)
            # s = {'pid': pid, 'uid': t.get_soloist_uid()}
            print("{}, rank {}".format(t.get_soloist_uid(), get_rank(t, pid)))
            # s['raw'] = get_raw_count(t.nid, t.get_soloist_uid())
            # z = len(t.get_descendants())
            # s['initial'] = z
            # Node.prune(t)
            # Node.collapse(t)
            # Node.condense(t)
            # while z > len(t.get_descendants()):
                # z = len(t.get_descendants())
                # Node.prune(t)
                # Node.collapse(t)
                # Node.condense(t)
            Node.clean_empty(t)
            Node.seq_change_only(t)
            Node.remove_false_evolvers(t)
            # s['final'] = z
            # summ.writerow(s)
            t.order_children(lambda n: n.get_nearest_data()['timestamp'])
            if args.viz:
                output_tree(t, get_rank(t, pid), t.get_soloist_uid(), pid)
        if not args.uid:
            with open(datapath("nodes.pickle", pid), 'wb') as fp:
                print("pickling nodes")
                pickle.dump(top_nodes + bottom_nodes, fp)

    fields = ['pid', 'uid', 'rank', 'num_lines', 'num_early_lines', 'num_late_lines',
              'num_manual_sections', 'median_manual_size', 'num_optima_escapes',
              'inquis_frac', 'median_inquis_size', 'focus_best_frac', 'num_target_macro',
              'median_macro_freq', 'macro_freq_std']


    def produce_data(t):
        print("producing data for {}, rank {}".format(t.get_soloist_uid(), get_rank(t, pid)))
        lines = detect_lines(t)
        extent = t.get_extent()
        manual = detect_extended_manual(t)
        manual_roots, manual_sizes = list(zip(*manual)) if len(manual) > 0 else ([], [])
        inquis_temp, num_branches = inquisitive_metric(t)
        inquis_branches, inquis_counts = list(zip(*inquis_temp)) if len(inquis_temp) > 0 else ([], [])
        best_only_branches, num_branches_explored = focus_metric(t)
        nodes_by_time = sorted(t.get_descendants(), key=lambda n: n.data['timestamp'])
        focus_time = [nodes_by_time.index(b) / len(nodes_by_time) for b in best_only_branches]
        macro_freq = macro_freq_per_path(t)
        return {
            'pid': pid,
            'uid': t.get_soloist_uid(),
            'rank': get_rank(t, pid),
            'num_lines': len(lines),
            'num_early_lines': len([l for l in lines if l.get_depth() < 0.1 * extent]), # early -> in first 10% by depth
            'num_late_lines': len([l for l in lines if l.get_depth() > 0.9 * extent]),  # late -> in last 10% by depth
            'num_manual_sections': len(manual_roots),
            'median_manual_size': np.median(manual_sizes) if len(manual_sizes) > 0 else np.nan,
            'num_optima_escapes': len(detect_optima_escape(t, 0.1, 0.8)),
            'inquis_frac': len(inquis_branches) / num_branches if num_branches > 0 else np.nan,
            'median_inquis_size': np.median(inquis_counts) if len(inquis_counts) > 0 else np.nan,
            'focus_best_frac': len(best_only_branches) / num_branches_explored if num_branches_explored > 0 else np.nan,
            'num_target_macro': len(detect_targeted_macro(t)),
            'median_macro_freq': np.median(macro_freq) if len(macro_freq) > 0 else np.nan,
            'macro_freq_std': np.std(macro_freq) if len(macro_freq) > 0 else np.nan
        }


    if args.produce_data:
        with open('data/top_node_data_v4.csv', 'a') as fp:
            print('writing top data')
            out = csv.DictWriter(fp, fieldnames=fields)
            for t in top_nodes:
                out.writerow(produce_data(t))
        with open('data/bottom_node_data_v4.csv', 'a') as fp:
            print('writing bottom data')
            out = csv.DictWriter(fp, fieldnames=fields)
            for t in bottom_nodes:
                out.writerow(produce_data(t))

# print("median sessions for top players: {}".format(np.median([len(b) for b in list(zip(*[get_playtimes(t) for t in top_nodes]))[1]])))
# print("median sessions for bottom players: {}".format(np.median([len(b) for b in list(zip(*[get_playtimes(t) for t in bottom_nodes]))[1]])))
# print("median session length for top players: {}s").format(np.median([t for p in list(zip(*[get_playtimes(t) for t in top_nodes]))[1] for t in p]))
