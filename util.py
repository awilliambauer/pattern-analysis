# coding: utf-8

import os, pickle, csv
import subprocess
from typing import NamedTuple, List, TextIO, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product, groupby, takewhile
from collections import namedtuple, Counter
import multiprocessing
import logging
import string
import matplotlib

matplotlib.use("Agg")

# pids with missing data (i.e., pdbs missing for either sid, eid, and/or gid)
pids_missing_data = {'2000524',
                     '2001234',
                     '2001249',
                     '2001255',
                     '2001287',
                     '2001291',
                     '2001306',
                     '2001308',
                     '2001311',
                     '2002239',
                     '2002243',
                     '2002247',
                     '2002255',
                     '2002713',
                     '2002963',
                     '2002990',
                     '2002992',
                     '2003008',
                     '2003011',
                     '2003015',
                     '997529',
                     '996023'}

unfetched_pids = {'2000659',
                  '2001302',
                  '2002102',
                  '2002465',
                  '2002809',
                  '2002833',
                  '2002850',
                  '2003001',
                  '2003047',
                  '2003059',
                  '2003078',
                  '2003126',
                  '2003183',
                  '996313',
                  '996492',
                  '996508',
                  '997542',
                  '997940',
                  '998465',
                  '998529',
                  '998574'}

# fetched, but corrupt
bad_pids = {'1998935',
            '2000659',
            '2001302',
            '2002102',
            '2002465',
            '2002809',
            '2002833',
            '2002850',
            '2003078',
            '2003126',
            '2003183',
            '2003763',
            '2003832',
            '997766'}

# stopped early due to crashes or errors
stopped_pids = {'2003699',
                '2003183',
                '2002494',
                '2002247',
                '2002912',
                '2003801'}

# restarted version of stopped puzzle
restarted_pids = {'2003704',
                  '2002499',
                  '2002255',
                  '2002914',
                  '2003806'}

pids_missing_energies = {'996547'}
pids_missing_pdl_actions = {'998071',
                            '1998729',
                            '998219'}
skip_pids = pids_missing_energies.union(pids_missing_pdl_actions).union(bad_pids)


class EnergyComponent(NamedTuple):
    name: str
    weight: float
    energy: float


class PDB_Info(NamedTuple):
    sid: str
    pid: str
    uid: str
    gid: str
    sharing_gid: str
    scoretype: str
    pdl: Dict
    energy: float
    energy_components: List[EnergyComponent]
    timestamp: int
    parent_sid: Optional[str]
    tmscore: float
    deviations: np.ndarray


class SnapshotDelta(NamedTuple):
    sid: str
    parent_sid: Optional[str]
    timestamp: int
    action_diff: Counter
    macro_diff: Counter
    action_count: int
    energy_diff: float


class SolvingLineVariant(NamedTuple):
    action_count: int
    time: int
    indices: List[int]


class SolvingLine(NamedTuple):
    action_count: int
    time: int
    pdb_infos: List[PDB_Info]
    variants: List[SolvingLineVariant]

    @property
    def energies(self):
        return [x.energy for x in self.pdb_infos]


class EvolvingLine(NamedTuple):
    source: Dict
    pdb_infos: List[PDB_Info]

    @property
    def energies(self):
        return [x.energy for x in self.pdb_infos]


class PuzzleMeta(NamedTuple):
    pid: str
    best_tmscores: Dict
    pfront: np.ndarray
    upload_baseline: float
    energy_baseline: float
    structure: Dict


class PatternInstance(NamedTuple):
    cid: int
    uid: str
    pid: str
    start_idx: int
    end_idx: int


class PatternInstanceExt(NamedTuple):
    cid: int
    uid: str
    pid: str
    start_idx: int
    end_idx: int
    start_pdb: PDB_Info
    end_pdb: PDB_Info
    pre_best: PDB_Info
    post_best: PDB_Info


class SubPatternInstance(NamedTuple):
    p: PatternInstance
    label: str
    start_idx: int
    end_idx: int


@pd.api.extensions.register_series_accessor("foldit")
class FolditSeriesAccessor:
    def __init__(self, pandas_obj: pd.Series):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.Series):
        # verify there is a column latitude and a column longitude
        if ('lines' not in obj.index or 'evol_lines' not in obj.index) and (obj.name != "lines" and obj.name != "evol_lines"):
            raise AttributeError("Must have 'lines' and 'evol_lines'.")

    @property
    def solo_pdbs(self):
        return [p for l in self._obj.lines for p in l.pdb_infos] if self._obj.lines else []

    @property
    def evol_pdbs(self):
        return [p for l in self._obj.evol_lines for p in l.pdb_infos] if self._obj.evol_lines else []

    @property
    def solo_energies(self):
        return [p.energy for p in self._obj.foldit.solo_pdbs]

    @property
    def evol_energies(self):
        return [p.energy for p in self._obj.foldit.evol_pdbs]


@pd.api.extensions.register_dataframe_accessor("foldit")
class FolditAccessor:
    def __init__(self, pandas_obj: pd.Series):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.Series):
        # verify there is a column latitude and a column longitude
        if 'lines' not in obj.columns or 'evol_lines' not in obj.columns:
            raise AttributeError("Must have 'lines' and 'evol_lines'.")

    @property
    def solo_pdbs(self):
        return self._obj.apply(lambda r: r.foldit.solo_pdbs, axis=1)

    @property
    def evol_pdbs(self):
        return self._obj.apply(lambda r: r.foldit.evol_pdbs, axis=1)

    @property
    def solo_energies(self):
        return self._obj.apply(lambda r: r.foldit.solo_energies, axis=1)

    @property
    def evol_energies(self):
        return self._obj.apply(lambda r: r.foldit.evol_energies, axis=1)

    # @property
    # def pdbs(self):



ROOT_NID = ('00000000-0000-0000-0000-000000000000', 0)

category_lookup = {
    'overall': '992758',
    'beginner': '992759',
    'prediction': '992760',
    'design': '992761',
    'electron': '994237',
    'contacts': '997946',
    'symmetry': '992769',
    'casp10': '992762',
    'casp11': '997398',
    'casp_roll': '993715',
    'hand_folding': '994890',
    'small_molecule_design': '2002074',
    "pilot": "2004148",
    'all': 'all',  # dummy to allow select of all categorized puzzles
}

action_types = {
    'optimize': {'ActionGlobalMinimize', 'ActionGlobalMinimizeBackbone', 'ActionGlobalMinimizeSidechains',
                 'ActionLocalMinimize', 'ActionRepack'},
    'hybrid': {'ActionLocalMinimizePull', 'LoopHash', 'ActionBuild', 'ActionPullSidechain', 'ActionTweak',
               'ActionRebuild'},
    'manual': {'ActionSetPhiPsi', 'ActionJumpWidget', 'ActionRotamerCycle', 'ActionRotamerSelect'},
    'guiding': {'ActionInsertCut', 'ActionLockToggle', 'ActionCopyToggle', 'ActionSecStructAssignHelix',
                'ActionSecStructAssignLoop', 'ActionSecStructAssignSheet', 'ActionSecStructDSSP', 'ActionSecStructDrag',
                'ActionBandAddAtomAtom', 'ActionBandAddDrag', 'ActionBandAddResRes', 'ActionBandDrag',
                'ActionBandLength', 'ActionBandStrength'},
}
action_types['deliberate'] = action_types['hybrid'].union(action_types['manual']).union(action_types['guiding'])


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def iden(x):
    return x


def get_ranks(datafile):
    puzzles = {}
    with open("{}.csv".format(datafile)) as fp:
        ranks_in = csv.DictReader(fp)
        for row in ranks_in:
            row['energy'] = float(row['best_score'])
            row['best_score'] = max(float(row['best_score']) * -10 + 8000, 0)
            pid = row['pid']
            if pid not in puzzles:
                puzzles[pid] = {
                    'groups': {},
                    'soloists': [],
                    'evolvers': [],
                    'categories': []
                }
            if row['gid'] == '0':
                row['gid'] = 'NULL'  # no sense in having both 0 and NULL for no group
            gid = row['gid']
            if gid != 'NULL':
                gs = puzzles[pid]['groups']
                if gid not in gs:
                    gs[gid] = {
                        'score': row['best_score'],
                        'type': row['type'],
                        'gid': gid,
                        'uid': row['uid'],
                    }
                if gs[gid]['score'] < row['best_score']:
                    gs[gid]['score'] = row['best_score']
                    gs[gid]['type'] = row['type']
                    gs[gid]['uid'] = row['uid']
            if row['type'] == '1':
                puzzles[pid]['soloists'].append(row)
            if row['type'] == '2':
                puzzles[pid]['evolvers'].append(row)

    for pid in puzzles:
        p = puzzles[pid]
        p['groups'] = list(p['groups'].values())
        # reverse sorts to put them in descending order (top ranked should be first)
        p['groups'].sort(key=lambda x: x['score'], reverse=True)
        for i, g in enumerate(p['groups']):
            g['rank'] = i
            g['norm_rank'] = i / len(p['groups'])
        p['soloists'].sort(key=lambda x: x['best_score'], reverse=True)
        for i, s in enumerate(p['soloists']):
            s['rank'] = i
            s['norm_rank'] = i / len(p['soloists'])
        p['evolvers'].sort(key=lambda x: x['best_score'], reverse=True)
        for i, e in enumerate(p['evolvers']):
            e['rank'] = i
            e['norm_rank'] = i / len(p['evolvers'])

    return puzzles


def get_ranks_labeled():
    puzzles = get_ranks("data/rprp_puzzle_ranks_latest")
    with open("data/puzzle_categories_latest.csv") as fp:
        cat_in = csv.DictReader(fp)
        for r in cat_in:
            pid = r['nid']
            if pid in puzzles:
                puzzles[pid]['categories'] = r['categories'].split(',')
                puzzles[pid]['categories'].append('all')
    with open("data/puzzle_labels_latest.json") as fp:
        lab_in = json.load(fp)
        for r in lab_in:
            pid = r['pid']
            if pid in puzzles:
                assert r['title'] is not None
                puzzles[pid]['title'] = r['title']
                if r['desc'] is not None:
                    puzzles[pid]['desc'] = r['desc']
    return puzzles


def add_pdbs_to_ranks(puzzles):
    print("loading pdbs")
    with open("data/top_pdbs.pickle", 'rb') as pdb_fp:
        pdbs = pickle.load(pdb_fp)

    pdbs = [p for p in pdbs if 'PID' in p and len(p['PDL']) > 0]

    print("grouping pdbs")
    pdbs_by_pid = {pid: list(g) for pid, g in groupby(pdbs, lambda p: p['PID'])}

    for pid in pids_missing_data.union(unfetched_pids):
        pid in puzzles and puzzles.pop(pid)

    for pid in puzzles.copy():
        pid not in pdbs_by_pid and puzzles.pop(pid)

    for pid, ps in pdbs_by_pid.items():
        if pid in puzzles:
            puzzles[pid]['pdbs'] = ps


def sig_test(a, b, fstr="{} (n={}) {} (n={})", normal=False, thresholds=frozenset()):
    if normal:
        t, p = stats.ttest_ind(a, b, equal_var=False)
    else:
        U2, p = stats.mannwhitneyu(np.array(a), np.array(b), use_continuity=True, alternative='two-sided')
        U = min(U2, len(a) * len(b) - U2)
    N = len(a) * len(b)
    f = len(list(filter(lambda xy: xy[0] > xy[1], product(a, b)))) / N
    u = len(list(filter(lambda xy: xy[0] < xy[1], product(a, b)))) / N

    if ('p' not in thresholds or p < thresholds['p']) and ('r' not in thresholds or abs(f - u) > thresholds['r']):
        print(fstr.format("mean={:.6f}, median={:.6f}, std={:.6f}".format(np.mean(a), np.median(a), np.std(a)), len(a),
                          "mean={:.6f}, median={:.6f}, std={:.6f}".format(np.mean(b), np.median(b), np.std(b)), len(b)))
        if normal:
            print("test statistic t: {:.6f}".format(t))
        else:
            print("Mann Whitney U: {:.6f}".format(U))
        print("significance (two-tailed): {:.6f}".format(p))
        print("rank-biserial correlation: {:.3f}".format(f - u))

    return p, f - u


def get_atoms(pdb):
    raw = [[float(x) for x in s.strip(' "[]').split(" ")] for s in pdb['ca'].split(",")]
    if all(k == 0 for k in raw[-1]):
        return np.array(raw[:-1])
    # remove spurious atom at 0 0 0 that appears at the end of each of these
    return np.array(raw)


def rmsd(X, Y):
    # center of mass
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # covariance matrix
    R = np.dot(X.T, Y)
    V, S, Wt = np.linalg.svd(R)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, Wt)
    Xp = np.dot(X, U)
    deviations = np.linalg.norm(Xp - Y, axis=1)
    return (deviations ** 2).sum() ** 0.5, deviations


# https://github.com/charnley/rmsd
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1471868/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321859/
def weighted_rmsd(X, Y, p=50):
    weights = np.array([[1]] * len(Y))
    wrmsd = 0
    wrmsd_old = float('inf')
    i = 0
    # there may be rare cases where this doesn't converge, so limit to 1000 iterations just in case
    while abs(wrmsd - wrmsd_old) > 1e-6 and i < 1000:
        i += 1
        wrmsd_old = wrmsd
        # weighted center of mass
        X = X - (weights * X).mean(axis=0)
        Y = Y - (weights * Y).mean(axis=0)
        # weighted covariance matrix
        R = np.dot(X.T, weights * Y)
        V, S, Wt = np.linalg.svd(R)
        d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
        if d:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        U = np.dot(V, Wt)
        Xp = np.dot(X, U)
        deviations = np.linalg.norm(Xp - Y, axis=1)
        wrmsd = ((weights.flatten() * deviations ** 2).sum() / weights.sum()) ** 0.5
        dp = np.percentile(deviations, p)
        weights = np.exp(-deviations ** 2 / dp ** 2).reshape((len(deviations), 1))
    return wrmsd, weights, deviations


# take in index i and series of Unix-style timestamps
# return indices start, end representing the period containing i with no gaps of size break_threshold or larger
# break_threshold given in seconds
def expand_seed(i, timestamps, break_threshold=900):
    start = end = i
    i = end + 1
    while i < len(timestamps) and timestamps[i] - timestamps[end] < break_threshold:
        end = i
        i += 1
    return start, end


# takes in list of Unix-style timestamps
def get_sessions(timestamps):
    sessions = []
    i = 0
    while i < len(timestamps):
        sessions.append(expand_seed(i, timestamps))
        i = sessions[-1][1] + 1
    return sessions


def time_splits_helper(timestamps, chunk, splits):
    sessions = get_sessions(timestamps)
    start_idx = end_idx = 0
    time_left = chunk
    session_idx = 0
    ret = []
    times = []
    for i in range(splits):
        logging.debug('split {}'.format(i))
        while time_left > 0 and session_idx < len(sessions):
            logging.debug('time left {}'.format(time_left))
            ses = sessions[session_idx]
            session_start, session_end = ses
            if session_duration(ses, timestamps) <= time_left:
                logging.debug('session {} {} fits'.format(session_idx, sessions[session_idx]))
                end_idx = session_end
                time_left -= session_duration(ses, timestamps)
                session_idx += 1
                if session_idx == len(sessions):
                    logging.debug('adding {} to the end'.format(start_idx))
                    ret.append((start_idx, len(timestamps)))
                    times.append(sum(session_duration((s, e), timestamps) for s, e in sessions if s >= start_idx))
                    logging.debug('time: {}'.format(times[-1]))
                else:
                    ns, ne = sessions[session_idx]
                    minimal_addition = session_duration((ns, ne), timestamps) if ns == ne else timestamps[ns + 1] - \
                                                                                               timestamps[ns]
                    # the minimum we could add to the current split would put us further away than we currently are
                    if abs(time_left - minimal_addition) > abs(time_left):
                        times.append(session_duration((session_start, end_idx), timestamps) + sum(
                            session_duration((s, e), timestamps) for s, e in sessions if
                            s >= start_idx and e < session_start))
                        if start_idx == end_idx:
                            end_idx += 1
                        logging.debug("close as we can get, adding {} up to {}".format(start_idx, end_idx))
                        ret.append((start_idx, end_idx))
                        logging.debug('time: {}'.format(times[-1]))
                        start_idx = end_idx
                        time_left = 0
            else:
                if session_start == session_end:
                    end_idx = session_end
                else:
                    end_idx = session_start + 1
                    while session_duration((session_start, end_idx), timestamps) < time_left:
                        end_idx += 1
                    if abs(time_left - (timestamps[end_idx] - timestamps[session_start])) > abs(
                            time_left - (
                                    timestamps[end_idx - 1] - timestamps[session_start])) and end_idx > start_idx + 1:
                        end_idx -= 1
                    logging.debug('splitting session at {}'.format(end_idx))
                    sessions[session_idx] = (end_idx, session_end)
                times.append(session_duration((session_start, end_idx), timestamps) + sum(
                    session_duration((s, e), timestamps) for s, e in sessions if s >= start_idx and e < session_start))
                if start_idx == end_idx:
                    end_idx += 1
                logging.debug('adding {} up to {}'.format(start_idx, end_idx))
                ret.append((start_idx, end_idx))
                logging.debug('time: {}'.format(times[-1]))
                start_idx = end_idx
                time_left = 0
        time_left = chunk
    return ret, times


def get_time_splits(time, timestamps, splits):
    chunk = time / splits
    ret, times = time_splits_helper(timestamps, chunk, splits)
    while len(ret) < splits - 1 and len(timestamps) >= 2 * splits:
        chunk *= 0.9
        logging.debug("bad split possibly due to degenerate chunk size, trying with {}".format(chunk))
        ret, times = time_splits_helper(timestamps, chunk, splits)

    if len(ret) == splits - 1 and any(e - s > 0 for s, e in ret):
        idx = np.argmax([t if s != e else 0 for (s, e), t in zip(ret, times)])
        shifted = ret[:idx] + [(ret[idx][0], ret[idx][1] - 1)] + [(s - 1, e - 1) for s, e in ret[idx + 1:]] + [
            (ret[-1][1] - 1, ret[-1][1])]
        logging.debug("short one slice, shifting everything to make another ({} to {})".format(ret, shifted))
        ret = shifted
    if ret[-1][1] < len(timestamps):
        logging.debug("extending final slice to the end ({} to {})".format(ret[-1][1], len(timestamps)))
        ret = ret[:-1] + [(ret[-1][0], len(timestamps))]
    assert len(ret) == splits or len(timestamps) < 2 * splits, "{} -- wanted {} splits, got {}".format(ret, splits,
                                                                                                       len(ret))
    # assert len(ret) == splits or splits > 12, "{} -- wanted {} splits, got {}".format(ret, splits, len(ret))
    assert all(e1 == s2 for (s1, e1), (s2, e2) in zip(ret, ret[1:]))
    covered_timestamps = np.concatenate([np.arange(s, e) for s, e in ret])
    assert all(x in covered_timestamps for x in range(len(timestamps))), \
        "{} -- not all timestamp indices accounted for: {}".format(ret, [x for x in range(len(timestamps)) if
                                                                         x not in covered_timestamps])
    # allowed_deviation = max([max(np.diff(timestamps[s:e]), default=0) for s, e in get_sessions(timestamps) if s != e], default=300) / 2
    # assert all(abs(t - chunk) < max(allowed_deviation, chunk * 0.1) for t in times[:-1]) or len(timestamps) <= 2 * splits, \
    #     "{} -- splits deviate too far from target size ({} ± {}): {}".format(ret, chunk, max(allowed_deviation, chunk * 0.1), times)
    return ret


def session_duration(session, timestamps):
    st, en = session
    if st == en:
        # assume 5 minutes of work where we have just a single snapshot for the session, based on standard upload rate
        return 300
    return timestamps[en] - timestamps[st]


def time_played(timestamps):
    return sum(session_duration(ses, timestamps) for ses in get_sessions(timestamps))


def align_timestamps(timestamps):
    sessions = [slice(s, e + 1) for s, e in get_sessions(timestamps)]
    ts_aligned = timestamps
    for ses1, ses2 in zip(sessions, sessions[1:]):
        adjustment = ts_aligned[ses2.start] - ts_aligned[ses1.stop - 1]
        ts_aligned = np.concatenate((ts_aligned[:ses2.start], ts_aligned[ses2.start:] - adjustment + 900))
    assert (np.diff(ts_aligned) <= 900).all()
    return ts_aligned


def get_children(nid, history):
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


def get_nid(s):
    return (s['uuid'], int(s['count']))


def output_atoms(atoms: np.ndarray, fp: TextIO) -> None:
    i = 1
    for ca in atoms:
        fp.write("ATOM {:6d}  CA  XXX A {:3d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(i, i, *ca))
        i += 1


def tmscore(pairs: List[Tuple[str, str]], tmp_input_name: str, atoms_lookup: Dict) -> Dict:
    if len(pairs) == 0:
        return {}
    logging.debug("{}: batch computing {} tmscores".format(tmp_input_name, len(pairs)))
    if not os.path.exists(tmp_input_name):
        os.makedirs(tmp_input_name)
    # write the necessary atom files
    sids = {s for ss in pairs for s in ss}
    for sid in sids:
        with open("{}/{}.atoms".format(tmp_input_name, sid), 'w') as fp:
            output_atoms(atoms_lookup[sid], fp)

    # empirically derived formula for chunksize to equalize batch time and spawning time
    # based on estimates that batches run 100 scores in ~1.5s, and Python starts ~6 batches per second
    chunksize = max(100, (len(pairs) / 0.09) ** 0.5)
    if len(pairs) // chunksize > (multiprocessing.cpu_count() / 4):
        chunksize = len(pairs) / (
                    multiprocessing.cpu_count() / 4)  # avoid spawning huge numbers of batches as this kills the performance
    splits = np.array_split(pairs, len(pairs) // chunksize if len(pairs) > chunksize else 1)
    ps = []
    for i, split in enumerate(splits):
        input_name = "{}/{}.tmscore_input".format(tmp_input_name, i)
        with open(input_name, 'w') as fp:
            for a, b in split:
                fp.write("{} {}\n".format(a, b))
        ps.append((subprocess.Popen(['./tmscore_batch.zsh', input_name], stdout=subprocess.PIPE, encoding='utf-8'),
                   input_name))
    scores = []
    for p, fname in ps:
        scores.extend([s.split() for s in p.communicate()[0].splitlines()])
        subprocess.run(['rm', fname])
    subprocess.run(["rsync", "-a", "--delete", "tmp_data/empty_dir/", "{}/".format(tmp_input_name)])
    return {(a, b): float(s) for a, b, s in scores}


def get_overlap(segment, target):
    seg_sessions = get_sessions(segment)
    tar_sessions = get_sessions(target)
    tar_adj = []
    for s, e in tar_sessions:
        cands = [ses for ses in seg_sessions if target[s] < segment[ses[1]] and target[e] > segment[ses[0]]]
        if len(cands) > 0:
            start = s
            while all(target[start] < segment[cs] for cs, ce in cands):
                start += 1
            # assert start < e
            end = e
            while all(target[end] > segment[ce] for cs, ce in cands):
                end -= 1
            # assert end >= start
            if start <= end:
                tar_adj.append((start, end))
    return tar_adj


def load_frame(datafile):
    df = pd.read_hdf(datafile, 'df')
    bts = pd.read_hdf(datafile, 'bts')
    puz = pd.read_hdf(datafile, 'puz').iloc[0]  # tuple gets wrapped in a pandas data structure, so unwrap it here
    logging.debug(datafile)
    return df, bts, puz


def collect_pdl_entries(soln):
    entries = list(takewhile(lambda x: x['header']['uid'] == soln.uid, soln.pdl[::-1]))
    actions = {}
    macros = {}
    for e in entries:
        for a, c in e['actions'].items():
            actions[a] = actions.get(a, 0) + c
        for m, c in e['macros'].items():
            macros[m] = macros.get(m, 0) + c
    return actions, macros


def get_data_value(uid, pid, key, data):
    r = data[(data.uid == uid) & (data.pid == pid)]
    return r[key].iloc[0]


def get_action_labels():
    return ['band', 'build', 'cut', 'global_min', 'idealize', 'local_min', 'lock',
            'rebuild', 'repack', 'assign_loop', 'save', 'reset', 'ss_load', 'ss_save']


def get_action_keys():
    """
    index: action type
    0: banding
    1: build
    2: cuts
    3: global minimize
    4: idealize
    5: local minimize
    6: locking
    7: rebuild
    8: repack
    9: assign secondary structure loop
    10: quicksave
    11: reset recent best
    12: load secondary structure
    12: save secondary structure
    """
    actionset_band = {'ActionBandAddAtomAtom',
                      'ActionBandAddDrag',
                      'ActionBandAddResRes',
                      'ActionBandDrag',
                      'ActionBandLength',
                      'ActionBandStrength',
                      'ActionBandDelete',
                      'ActionBandDisableToggle'}
    actionset_cut = {'ActionDeleteCut',
                     'ActionInsertCut'}
    actionset_global = {'ActionGlobalMinimize',
                        'ActionGlobalMinimizeBackbone',
                        'ActionGlobalMinimizeSidechains'}
    actionset_save = {'ActionStandaloneQuicksave', 'ActionNoviceQuicksave'}
    actionset_load = {'ActionStandaloneResetRecentBest', 'ActionNoviceResetRecentBest'}
    actionset_ss_save = {'ActionStandaloneSecstructSave', 'ActionNoviceSecstructSave'}
    actionset_ss_load = {'ActionStandaloneSecstructLoad', 'ActionNoviceSecstructLoad'}

    return [actionset_band, {'ActionBuild'}, actionset_cut, actionset_global, {'ActionIdealize'},
            {'ActionLocalMinimize'},
            {'ActionLockToggle'}, {'ActionRebuild'}, {'ActionRepack'}, {'ActionSecStructAssignLoop'}, actionset_save,
            actionset_load, actionset_ss_load, actionset_ss_save]


def get_action_stream(action_diff: Counter):
    keys = get_action_keys()
    return [sum(action_diff.get(a, 0) for a in k) for k in keys]


def get_pattern_label(p, cid, sub_k):
    if sub_k == 0:
        assert p.cid == cid
        return str(cid)
    return str(cid) + string.ascii_uppercase[p.cid]
