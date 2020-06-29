import logging
from functools import partial


def _get_raw_children(nid, history):
    uuid, count = nid
    main = list(filter(lambda x: x['uuid'] == uuid, history[nid])) if nid in history else []
    if len(main) == 0 and count != 0:
        main = list(filter(lambda x: x['uuid'] == uuid and x['count'] > count, history[(uuid, 0)]))
    other = list(
        filter(lambda x: x['uuid'] != uuid and x['parent_count'] == count, history[nid])) if nid in history else []

    children = []
    if len(main) > 0:
        c = min(main, key=lambda x: x['count'])
        children.append((c['uuid'], c['count']))
    for r in other:
        children.append((r['uuid'], r['count']))
    return children


def _get_solution_data(nid, solutions):
    return solutions[nid] if nid in solutions else None


def get_solution_data(nid):
    raise NotImplementedError


def get_raw_children(nid):
    raise NotImplementedError


def initialize(history, solutions):
    global get_solution_data, get_raw_children
    get_solution_data = partial(_get_solution_data, solutions=solutions)
    get_raw_children = partial(_get_raw_children, history=history)


class RawNode:
    def __init__(self, nid, parent):
        self.nid = nid
        self.data = get_solution_data(self.nid)
        self.parent = parent
        self.descendants_cache = None
        self.uid = None
        if self.data:
            self.uid = self.data['uid']
        elif self.get_parent():
            self.uid = self.get_parent().uid

    def get_nid(self):
        return self.nid

    def get_name(self):
        name = str(self.nid[0]) + '-' + str(self.nid[1])
        return name

    def get_parent(self):
        return self.parent

    def get_ancestors(self):
        p = self.get_parent()
        while p:
            yield p
            p = p.get_parent()

    def get_children(self, include_evolvers=False):
        raw_children = get_raw_children(self.nid)
        children = [RawNode(nid, self) for nid in raw_children]
        if not include_evolvers and self.uid:
            return [c for c in children if c.uid == self.uid]
        return children


    def get_merged_branching_children(self, include_evolvers=False):
        children = self.get_children(include_evolvers)
        relevant_children = []
        for child in children:
            depth = 1
            logging.debug("%s %s %s", child.get_nid(), child.uid, depth)
            child_children = child.get_children(include_evolvers)
            relevant_child = None
            best_data = None

            while True:
                logging.debug("%s %s %s", child.get_nid(), child.uid, depth)
                sol_data = child.get_solution_data()

                if sol_data != None:
                    logging.debug("Solution child found (Depth %s)", depth)
                    if best_data == None or sol_data['energy'] < best_data['energy']:
                        best_data = sol_data
                    # if len(child_children) == 0:
                    #     relevant_child = child
                    #     break
                    relevant_child = child
                    break

                if len(child_children) == 0:
                    logging.debug("No suitable children found (Depth %s)", depth)
                    break
                if len(child_children) > 1:
                    logging.debug("Branching child found (Depth %s)", depth)
                    relevant_child = child
                    break

                child = child_children[0]
                depth += 1
                child_children = child.get_children(include_evolvers)

            if relevant_child != None:
                relevant_child.data = best_data
                relevant_child.parent = self
                relevant_children.append(relevant_child)
        return relevant_children

    def get_descendants(self, include_evolvers=False):
        if self.descendants_cache is None:
            a = []
            b = [self.get_merged_branching_children(include_evolvers)]
            while any([len(x) for x in b]):
                for x in b:
                    a.extend(x)
                b = [n.get_merged_branching_children(include_evolvers) for n in [x for l in b for x in l]]
            self.descendants_cache = a
            return a
        else:
            return self.descendants_cache

    def get_min_energy(self, include_evolvers=False):
        nodes = [n for n in self.get_descendants(include_evolvers) if n.data]
        return min(nodes, key=lambda n: n.data['energy']).data['energy'] if len(nodes) > 0 else 9999

    def traverse_descendants(self, include_evolvers=False):
        nodes = [self.get_merged_branching_children(include_evolvers)]
        while any([len(x) for x in nodes]):
            for children in nodes:
                for x in children:
                    yield x
            nodes = [n.get_merged_branching_children(include_evolvers) for n in [x for l in nodes for x in l]]

    def get_nearest_data(self):
        if self.data:
            return self.data
        for d in self.traverse_descendants():
            if d.data:
                return d.data
        return {'uid': '000000', 'energy':9999} # return dummy data if this subtree is actually just empty

    def get_soloist_uid(self):
        return self.get_nearest_data()['pdl'][0]['header']['uid']

    def get_solution_data(self):
        return self.data

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_nid() == other.get_nid() and self.data == other.data
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash((self.get_nid(), tuple(sorted(self.data.items()))))
