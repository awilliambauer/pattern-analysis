import numpy as np
from collections import Counter
from itertools import groupby

# modify to look for sequential children separated by some threshold
def get_playtimes(node, threshold=28800):  # look for > 8-hour gaps by default
    times = [x.data['timestamp'] for x in sorted(node.get_descendants(), key=lambda n: n.data['timestamp'])]
    diff = np.diff(times)
    if not any([b > threshold for b in diff]):
        return [], [times[-1] - times[0]]
    indexes, breaks = zip(*[b for b in enumerate(diff) if b[1] > threshold])
    play = [times[e] - times[s] for s, e in zip([0] + list(indexes), list(indexes) + [len(times) - 1])]
    return list(breaks), play


def detect_lines(node, local_extent_factor=0.25, global_extent_factor=0.25, abs_threshold=10):
    lines = []
    total_extent = node.get_extent()
    q = [node.children[:]]
    while len(q) > 0:
        ch = q.pop()
        if len(ch) > 0:
            max_extent = max([c.get_extent() for c in ch])
            if max_extent > global_extent_factor * total_extent and max_extent > abs_threshold:
                potential = [c for c in ch if c.get_extent() > local_extent_factor * max_extent and c.get_extent() > abs_threshold]
                if len(potential) > 1:
                    lines.extend(potential)
                q.extend([c.children for c in ch if len(c.children) > 1])
    return lines


def is_global_root(node):
    return node.nid == ('00000000-0000-0000-0000-000000000000', 0)


def node_predecessors_manual(node):
    for an in node.get_ancestors():
        if is_global_root(an) or (an.has_manual_action() and len(an.get_new_macros()) == 0):
            return True
        elif len(an.get_new_macros()) > 0:
            return False
    return True


def detect_extended_manual(node, min_count=5):
    subtrees = []
    q = [node]
    while len(q) > 0:
        root = q.pop()
        # make sure root is clean (i.e., must encounter MANUAL in ancestors before macro)
        if (root.has_manual_action() and len(root.get_new_macros()) == 0) or node_predecessors_manual(root):
            count = 0
            candidates = [root]
            while len(candidates) > 0:
                candidate = candidates.pop()
                if len(candidate.get_new_macros()) == 0: # we know everything up to the root is clean, so just check child
                    count += sum(len(cas) for cas in candidate.cascade) if hasattr(candidate, 'cascade') else 1
                    candidates.extend(candidate.children)
                else:
                    q.extend(candidate.children)
            if count > min_count:
                subtrees.append((root, count))
        else:
            q.extend(root.children)
    return subtrees


def detect_optima_escape(node, low_score_p=0.2, high_score_p=0.9, subtree_threshold=2):
    ds = node.get_descendants()
    if len(ds) > 0:
        energies = [x.data['energy'] for x in ds]
        best = min(energies)
        return [x for x in ds if x.parent.data and x.data['energy'] > x.parent.data['energy'] - low_score_p * best and
                x.parent.data['energy'] < high_score_p * best and len(x.get_descendants()) > subtree_threshold]
    return []


def explored_children(node, explore_threshold):
    return [x for x in node.children if x.get_extent() > explore_threshold]


def inquisitive_metric(node, explore_threshold=2, branch_factor=1, min_num_explored=2):
    # number of branches with multiple children explored, mean number explored per branch
    # relationship with depth?
    bs,_ = node.get_branches(branch_factor)
    targets = zip(bs,[len(explored_children(b, explore_threshold)) for b in bs])
    return [(b,count) for b,count in targets if count >= min_num_explored], len(bs)


def focus_metric(node, explore_threshold=2, branch_factor=1):
    # best-scoring, most-recent
    bs,_ = node.get_branches(branch_factor)
    explored = [b for b in bs if len(explored_children(b, explore_threshold)) > 0]
    singles = [b for b in bs if len(explored_children(b, explore_threshold)) == 1]
    best = [x for x in singles if min(x.children, key=lambda n: n.data['energy']) == explored_children(x, explore_threshold)[0]]
    recent = [x for x in singles if max(x.children, key=lambda n: n.data['timestamp']) == explored_children(x, explore_threshold)[0]]
    # return best, recent, len(explored), len(bs)
    return best, len(explored)


def detect_targeted_macro(node, impact_p=10, abs_rank=5, best_p=0.8):
    # select the biggest impact child for each parent since otherwise a bad node with a bunch of children will dominate
    # screen out positive energy parents, as they can spuriously dominate the energy gains
    nodes = [min(g, key=lambda n: n.data['energy'] - n.parent.data['energy']) for _, g in groupby([n for n in node.get_descendants() if n.parent.data and n.parent.data['energy'] < 0], lambda n: n.parent.nid)]
    if len(nodes) > 0:
        gains = [n.data['energy'] - n.parent.data['energy'] for n in nodes]
        threshold = max(np.percentile(gains, impact_p), sorted(gains)[min(abs_rank, len(gains)-1)])  # percentile can be too harsh for small graphs
        cutoff = min(best_p * node.get_min_energy(), np.percentile([n.data['energy'] for n in node.get_descendants()], 50))
        candidates = [n for n in nodes if n.data['energy'] - n.parent.data['energy'] < threshold
                      and n.data['energy'] - n.parent.data['energy'] < 0.01 * node.get_min_energy() and n.data['energy'] < cutoff]
        candidates = [n for n in candidates if all(an.data is None or n.data['energy'] < an.data['energy'] for an in n.get_ancestors())]
        final = []
        nodes = ([node] if node.data else []) + sorted(node.get_descendants(), key=lambda n: n.data['timestamp'])
        for candidate in candidates:
            siblings = candidate.parent.children[:]
            siblings.remove(candidate)
            i = nodes.index(candidate)
            predecessors = nodes[:i]
            cmacros = set((candidate.get_new_macros() + candidate.parent.get_new_macros()).keys())
            smacros = set(sum([s.get_new_macros() for s in siblings], Counter({})).keys())
            pmacros = set(sum([p.get_new_macros() for p in predecessors], Counter({})).keys())
            if smacros.union(pmacros).isdisjoint(cmacros) and len(cmacros) > 0:
                final.append(candidate)
        return final
    return []


def get_all_paths(node):
    terminals = [n for n in node.get_descendants() if len(n.children) == 0]
    return [[t] + [a for a in t.get_ancestors() if not is_global_root(a)] for t in terminals]


def macro_freq_per_path(node):
    return [np.mean([1 if len(x.get_new_macros()) > 0 else 0 for x in p]) for p in get_all_paths(node)]


# def detect_long_backtrack(node):
