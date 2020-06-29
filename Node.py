import sys
from collections import Counter
from itertools import groupby

sys.setrecursionlimit(20000)

MANUAL_ACTIONS = {"ActionBandAddDrag", "ActionBandDrag", "ActionBandLengthMenu", "ActionBandMenu",
                  "ActionBandStrengthMenu", "ActionButtonBridge", "ActionDisulfideAddDrag", "ActionInsDelResidueMenu",
                  "ActionInsertResiduesMenu", "ActionJumpWidget", "ActionPull", "ActionPullSidechain",
                  "ActionReplaceResidueMenu", "ActionSecStructDrag", "ActionSecStructFloatMenu", "ActionSecStructMenu",
                  "ActionSetPhiPsi", "ActionTweak", "ActionAlignPoseToDensity"}


class Node:
    def __init__(self, rn, include_evolvers=False):
        self.nid = rn.get_nid()
        self.parent = rn.get_parent()
        self.data = rn.data  # rn.get_nearest_data()
        self.uid = rn.uid
        # if we're the global root, including evolvers in our children will just get every solver
        if self.parent is None:
            self.children = [Node(x, include_evolvers) for x in rn.get_merged_branching_children()]
        else:
            self.children = [Node(x, include_evolvers) for x in rn.get_merged_branching_children(include_evolvers)]
        for c in self.children:
            c.parent = self

    def get_name(self):
        name = str(self.nid[0]) + '-' + str(self.nid[1])
        return name

    def get_ancestors(self):
        p = self.parent
        while p:
            yield p
            p = p.parent

    def get_descendants(self):
        a = []
        b = [self.children]
        while any([len(x) for x in b]):
            for x in b:
                a.extend(x)
            b = [n.children for n in [x for l in b for x in l]]
        return a

    def get_min_energy(self):
        nodes = [x for x in self.get_descendants() if x.data]
        return min(nodes, key=lambda n: n.data['energy']).data['energy'] if len(nodes) > 0 else 9999

    def traverse_descendants(self):
        nodes = [self.children]
        while any([len(x) for x in nodes]):
            for cs in nodes:
                for x in cs:
                    yield x
            nodes = [n.children for n in [x for l in nodes for x in l]]

    def get_nearest_data(self, provide_dummy=True):
        if self.data:
            return self.data
        for d in self.traverse_descendants():
            if d.data:
                return d.data
        # return dummy data if this subtree is actually just empty
        return {'uid': '000000', 'energy': 9999} if provide_dummy else None

    def get_soloist_uid(self):
        return self.get_nearest_data()['pdl'][0]['header']['uid']

    def get_new_counts(self, kind):
        if self.data and 'pdl' in self.data:
            for a in self.get_ancestors():
                if a.data and 'pdl' in a.data:
                    if len(self.data['pdl']) == len(a.data['pdl']):
                        return self.data['pdl'][-1][kind] - a.data['pdl'][-1][kind]
                    else:  # arbitrary number of new entries in pdl
                        n = len(self.data['pdl']) - len(a.data['pdl'])
                        diff = self.data['pdl'][-(n + 1)][kind] - a.data['pdl'][-1][kind]
                        return diff + sum([p[kind] for p in self.data['pdl'][-n:]], Counter({}))
            # first node with data, provide empty Counter as start value
            return sum([p[kind] for p in self.data['pdl']], Counter({}))
        # raise ValueError("can't get new macros on RawNode with no data")
        return Counter({})

    def get_new_macros(self):
        return self.get_new_counts('macros')

    def get_new_actions(self):
        return self.get_new_counts('actions')

    def has_manual_action(self):
        return any(a in MANUAL_ACTIONS for a in self.get_new_actions().keys())

    def instinct(self):
        """
        returns a Counter showing the number of times the children of a branch that were used (i.e., that have children)
        did not include the child with the best energy (False -> best child not included)
        only branches where we have the data for all children and where at least one child was used are considered
        """
        b, _ = self.get_branches()
        c = Counter(
            [min(x.children, key=lambda n: n.data['energy']) in [c for c in x.children if len(c.children) > 0] for x in
             b if all(c.data for c in x.children) and any(len(c.children) > 0 for c in x.children)])
        return c

    def get_quantify_diff(self, kind):
        if self.data:
            for a in self.get_ancestors():
                if a.data:
                    return self.data[kind] - a.data[kind]
        return None

    def get_energy_diff(self):
        return self.get_quantify_diff('energy')

    def get_time_diff(self):
        return self.get_quantify_diff('timestamp')

    def get_branches(self, k=2):
        branches = [n for n in self.get_descendants() if len(n.children) > k]
        factor = [len(b.children) for b in branches]
        return branches, factor

    def get_extent(self):
        """
        get the depth of deepest child
        """
        c = self.children
        if len(c) > 0:
            return (len(self.cascade) if hasattr(self, 'cascade') else 1) + max([n.get_extent() for n in c])
        return 1

    def extent_at_least(self, target):
        for d in self.traverse_descendants():
            if d.get_depth(self) >= target:
                return True
        return False

    def get_depth(self, relative_to=None):
        """
        get depth of this node, not counting previous players or the global root node
        """
        d = 0
        p = self.parent
        while p and (p.uid is None or p.uid == self.uid) and p.nid != ('00000000-0000-0000-0000-000000000000', 0) \
                and (relative_to is None or p != relative_to):
            d += len(p.cascade) if hasattr(p, 'cascade') else 1
            p = p.parent
        return d

    def order_children(self, key):
        for c in self.children:
            c.order_children(key)
        self.children.sort(key=key)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.nid == other.nid and (repr(sorted(self.data.items())) == repr(
                sorted(other.data.items())) if self.data and other.data else self.data == other.data)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash((self.nid, repr(sorted(self.data.items())) if self.data else (('energy', 9999), ('uid', '000000'))))

    @staticmethod
    def prune(node):
        # for nodes with 2 or 3 children, prune children with no children of their own
        if 1 < len(node.children) < 4 and len([c for c in node.children if len(c.children) > 0]) == 1:
            node.children = [c for c in node.children if len(c.children) > 0]
        # recursively prune children
        for c in node.children:
            Node.prune(c)
        # prune children that don't go anywhere (extent < 10) or that don't change the energy (i.e., nothing happened)
        # exception: don't prune if all children would be pruned -- this is to leave the bottom bits of the tree intact
        if any(c.extent_at_least(10) for c in node.children):
            node.children = [c for c in node.children if
                             c.extent_at_least(10) or (c.get_energy_diff() is None or c.get_energy_diff() != 0)]

        for c in node.children:
            c.parent = node

    @staticmethod
    def collapse(node):
        while len(node.children) == 1:
            c = node.children[0]
            if c.data:
                node.data = c.data
            node.children = c.children
        for c in node.children:
            c.parent = node
            Node.collapse(c)

    @staticmethod
    def condense(node):
        cascade = []
        h = []
        cur = node
        while len(cur.children) > 1 and len([c for c in cur.children if len(c.children) > 0]) == 1 and \
                all(len(c.get_new_macros()) == 0 for c in cur.children):
            cascade.append(cur.children)
            if cur.data:
                h.append(cur)
            cur = [c for c in cur.children if len(c.children) > 0][0]
        if len(cascade) > 2:
            if cur.data:
                node.data = cur.data
            elif len(h) > 0:
                node.data = h[-1].data
            node.children = cur.children
            node.cascade = cascade
        for c in node.children:
            c.parent = node
            Node.condense(c)

    @staticmethod
    def clean_empty(node):
        # print(node.nid, node.parent.nid, [x.nid for x in node.children])
        if node.data is None and node.nid != ('00000000-0000-0000-0000-000000000000', 0) \
                and node.parent.nid != ('00000000-0000-0000-0000-000000000000', 0):
            # print("moving {} from {} to {}".format([x.nid for x in node.children], node.nid, node.parent.nid))
            node.parent.children.extend(node.children)
            node.parent.children.remove(node)
            for c in node.children:
                c.parent = node.parent
        for c in node.children[:]:
            Node.clean_empty(c)

    @staticmethod
    def seq_change_only(node):
        while node.data and any(c.data and node.data['seq'] == c.data['seq'] and node.uid == c.uid for c in node.children):
            to_merge = [c for c in node.children if c.data and node.data['seq'] == c.data['seq'] and node.uid == c.uid]
            node.children = [c for c in node.children if c.data is None or node.data['seq'] != c.data['seq'] or node.uid != c.uid]
            for c in to_merge:
                node.children.extend(c.children)
        for c in node.children:
            c.parent = node
            Node.seq_change_only(c)
        if all(c.data for c in node.children):
            has_children = [c for c in node.children if len(c.children) > 0]
            included_seqs = {c.data['seq'] for c in has_children}
            to_merge = [c for c in node.children if len(c.children) == 0 and c.data['seq'] not in included_seqs]
            node.children = has_children
            node.children.extend([min(g, key=lambda x: x.data['energy']) for _, g in
                                  groupby(sorted(to_merge, key=lambda c: c.data['seq']), lambda c: c.data['seq'])])


    @staticmethod
    def remove_false_evolvers(node):
        # because of missing data, subtrees that actually never include the intended solver
        # get attached, we can remove them once the tree is cleaned up
        node.children = [c for c in node.children if c.get_nearest_data()['uid'] == node.uid]
