import csv
import numpy as np
import math
from typing import NamedTuple
from itertools import groupby
import logging
import matplotlib
matplotlib.use("Agg")
from pattern_extraction import *
from datetime import datetime
from check_models import load_sub_lookup
from pattern_model_evaluation import *
from dump_predicted_patterns import get_pattern_count

events = []
print("loading event dumps")
for event_file in os.listdir("mozak/user-event-dumps"):
    with open("mozak/user-event-dumps/" + event_file) as fp:
        eventlines = fp.readlines()
        for event in eventlines:
            e = json.loads(event)["_source"]
            # dataset includes at least one corrupted event without a type and numerous with None as the type and some without a dataset_id
            if "type" in e and e["type"] and "dataset_id" in e and e["dataset_id"]:
                # some timestamps include nanoseconds (which datetime can't parse) and some don't
                # since all are in UTC, we can truncate timestamps at seconds and be consistent across all events
                e["timestamp"] = int(datetime.fromisoformat(e["@timestamp"][:19]).timestamp())
                events.append(e)

user_events = {uid: {did: sorted(xs, key=lambda e: e["timestamp"]) for did, xs in groupby(sorted(es, key=lambda e: e["dataset_id"]), lambda e: e["dataset_id"])}
               for uid, es in groupby(sorted(events, key=lambda e: e["user_id"]), lambda e: e["user_id"])}


def get_action_keys():
    return [{"CONNECT_THE_DOTS"}, {"VIRTUAL_FINGER"}, {"UNDO"}, {"DELETE_NODE", "DELETE_EDGE", "DELETE_BRANCH"},
            {"SET_HOTSPOTS_VISIBILITY", "SET_COMMENTS_VISIBILITY", "TOGGLE_HOT_SPOTS_VISIBILITY", "TOGGLE_CONSENSUS_VISIBILITY", "TOGGLE_COMMENTS_VISIBILITY",
             "TOGGLE_TRACES_COMMENTS_VISIBILITY", "CHANGE_VIEW_MODE"}, {"OSD_VIEWPORT_CHANGE", "ENABLE_TRACE_MODE"}]


streams = ['CONNECT_THE_DOTS',
           'VIRTUAL_FINGER',
           'UNDO',
           'DELETE',
           'OVERLAYS',
           'PRETRACE']

series_lookup = {}
puz_idx_lookup = {}
binwidth = 60 * 10
noise = [200] * len(streams)
print("constructing series")
for i, (uid, d) in enumerate(user_events.items()):
    print("{} of {}\r".format(i, len(user_events)), end="")
    s = []
    i = 1
    for pid, es in d.items():
        if len(es) == 1:
            continue
        num_bins = max(1, (es[-1]["timestamp"] - es[0]["timestamp"]) // binwidth)
        bins = np.linspace(es[0]["timestamp"], es[-1]["timestamp"], num_bins + 1)
        for start, end in zip(bins, bins[1:]):
            in_bin = [e for e in es if start <= e["timestamp"] < end + 1]
            s.append([len([e for e in in_bin if e["type"] in fs]) for fs in get_action_keys()])
        puz_idx_lookup[(uid, pid)] = (len(s) - num_bins, len(s))
        if i < len(d):
            s.extend([noise] * 100)
        i += 1
    if len(s) > 2:
        logging.debug("{} on {} datasets".format(uid, len(d)))
        logging.debug("{} of {} bins are empty".format(len([r for r in s if sum(r) == 0]), len(s)))
        logging.debug("mean action count in active sections: {}".format(np.mean([sum(r) for r in s if r != noise and sum(r) > 0])))
        logging.debug("")
        series_lookup[uid] = np.array(s)

print("included {} of {} users".format(len(series_lookup), len(user_events)))
idx_lookup, all_series = combine_user_series(series_lookup, noise, 200)
puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0]) for (uid, pid), (s, e) in puz_idx_lookup.items() if uid in idx_lookup}
# run_TICC({"all": all_series}, "mozak", [5, 6, 7, 8, 9, 10, 12, 15, 20])

print("loading TICC output")
krange = [5, 6, 7, 8, 9, 10, 12, 15, 20]
cluster_lookup, mrf_lookup, model_lookup, _ = load_TICC_output("mozak", ["all"], krange)
with open("mozak/all/subpatterns/subseries_lookup.pickle", 'rb') as fp:
    subseries_lookup = pickle.load(fp)
sub_lookup = load_sub_lookup("mozak/all", subseries_lookup)
sub_clusters = sub_lookup["clusters"]

print("finding patterns")
pattern_lookup = get_patterns_lookup(krange, sub_clusters, sub_lookup["mrfs"], subseries_lookup, cluster_lookup["all"], mrf_lookup["all"], puz_idx_lookup)
best_k, best_subs = find_best_dispersion_model(all_series, pattern_lookup, subseries_lookup, sub_clusters, sub_lookup["mrfs"])

ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in pattern_lookup[best_k][cid][sub_k]] for cid, sub_k in best_subs], [])
ps_uid_pid = {tag: sorted(xs) for tag, xs in groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
pattern_use_lookup = {tag: {pt for pt, _ in xs} for tag, xs in ps_uid_pid.items()}
pts = {pt for pt, p in ps}

with open("mozak/consensus.csv") as fp:
    rows = [r for r in csv.DictReader(fp)]
    raw = {tag: max(xs, key=lambda x: x["time"]) for tag, xs in groupby(sorted(rows, key=lambda r: (r['user_id'], r['dataset_id'])), lambda r: (r['user_id'], r['dataset_id']))}
    consensus = {tag: {"nodes": int(x["nodes"]), "contributed": int(x["nodeContribution"]), "timestamp": int(x["time"])} for tag, x in raw.items()}

acc = []
for uid, pid in consensus:
    c = consensus[(uid, pid)]
    if (uid, pid) not in ps_uid_pid or c["nodes"] == 0:
        print("skipping", uid, pid, c)
        continue
    r = {"uid": uid, "pid": pid, "nodes": c["nodes"], "contributed": c["contributed"], "efficiency": c["contributed"] / c["nodes"]}
    prior = [x for (u, p), x in consensus.items() if u == uid and x["timestamp"] < c["timestamp"]]
    r["experience"] = len(prior)
    r["contributed_prior"] = np.median([x["contributed"] for x in prior])
    r["efficiency_prior"] = np.median([(c["contributed"] / c["nodes"]) if c["nodes"] > 0 else 0  for c in prior])
    r["action_count"] = 0
    for target_pt in pts:
        ps = [p for pt, p in ps_uid_pid[(uid, pid)] if pt == target_pt]
        count = sum(get_pattern_count(p, target_pt, all_series, subseries_lookup[best_k]) for p in ps)
        r["action_count"] += count
        r["pattern_{}_action".format(target_pt)] = count
        r["pattern_{}_use".format(target_pt)] = 1 if count > 0 else 0
    acc.append(r)
features = pd.DataFrame(data=acc)
features.contributed_prior.fillna(features.contributed_prior.median(), inplace=True)
features.efficiency_prior.fillna(features.efficiency_prior.median(), inplace=True)

seed = 13*17*31
models = {#"ridge": Ridge,
          "ensemble": GradientBoostingRegressor}
model_params = {"ridge": {"random_state": [seed], "alpha": [0.1, 0.5, 1, 5, 10], "normalize": [True, False]},
                "ensemble": {"random_state": [seed], "learning_rate": [0.01, 0.02, 0.05, 0.1], "subsample": [0.3, 0.5, 0.7],
                             "n_estimators": [1000], "n_iter_no_change": [200]}}
model_params["ensemble"]["loss"] = ["huber"]
model_params["ensemble"]["alpha"] = [0.85, 0.9, 0.95, 0.99]
baseline_features = ["action_count", "experience", "contributed_prior"]
outcomes = ["contributed", "efficiency"]

experience_cohorts = {"exp_min{}".format(t): features.experience > t for t in [1, 5, 10]}
experience_cohorts["exp_max5"] = features.experience <= 5
action_cohorts = {"action_min{}".format(t): features.action_count > t for t in [50, 100, 500]}
action_cohorts["action_max50"] = features.action_count <= 50

with Pool(50, maxtasksperchild=10) as pool:
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=seed)
    model_scores = {}
    print("fitting models")
    scores = {}
    X = features[baseline_features].values
    y = features.contributed.values.ravel()
    for lab, model in models.items():
        evals = []
        for param in ParameterGrid(model_params[lab]):
            evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
        print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
        scores[lab] = [(x.get(), param) for x, param in evals]
    baseline_scores = scores
    print("baseline")
    print(max(scores["ensemble"], key=lambda x: x[0][0]))

    for ftype in ["action", "use"]:
        scores = {}
        X = features[["experience", "contributed_prior"] + ["pattern_{}_{}".format(pt, ftype) for pt in pts]].values
        y = features.contributed.values.ravel()
        for lab, model in models.items():
            evals = []
            for param in ParameterGrid(model_params[lab]):
                evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
            print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
            scores[lab] = [(x.get(), param) for x, param in evals]
        model_scores[ftype] = scores
        print("best pattern model")
        print(max(scores["ensemble"], key=lambda x: x[0][0]))

    cohort_scores = {}
    for cohort, sel in chain(experience_cohorts.items(), action_cohorts.items()):
        print("fitting {} cohort models".format(cohort))
        cohort_scores[cohort] = {}
        print("fitting baseline")
        scores = {}
        X = features[sel][baseline_features].values
        y = features[sel]["contributed"].values.ravel()
        for lab, model in models.items():
            evals = []
            for param in ParameterGrid(model_params[lab]):
                evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
            print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
            scores[lab] = [(x.get(), param) for x, param in evals]
        cohort_scores[cohort]["baseline"] = scores
        print(cohort, "baseline")
        print(max(scores["ensemble"], key=lambda x: x[0][0]))

        print("fitting pattern")
        candidate_features = ["experience", "contributed_prior"] + ["pattern_{}_action".format(pt) for pt in pts]
        scores = {}
        X = features[sel][candidate_features].values
        y = features[sel]["contributed"].values.ravel()
        for lab, model in models.items():
            evals = []
            for param in ParameterGrid(model_params[lab]):
                evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
            print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
            scores[lab] = [(x.get(), param) for x, param in evals]
        cohort_scores[cohort]["pattern"] = scores
        print("best {} pattern model".format(cohort))
        print(max(scores["ensemble"], key=lambda x: x[0][0]))


# print("plotting")
# plot_model("mozak", best_k, best_subs, all_series, pattern_lookup, sub_clusters, subseries_lookup)
# plot_user_series("mozak", best_k, best_subs, puz_idx_lookup, all_series, pattern_lookup, pts, subseries_lookup)

fig, ax = plt.subplots(figsize=(18, 10))
elite_uids = features[(features.experience > 10) & (features.contributed_prior > np.percentile(features.contributed_prior, 85))].uid
use_data = features[features.uid.isin(elite_uids)].groupby("uid")[["pattern_{}_use".format(f) for f in sorted(pts)]].mean()
im = ax.imshow(use_data.values, cmap="Greens")
cbar = ax.figure.colorbar(im, cax=fig.add_axes([0.9, 0.12, 0.015, 0.75]))
cbar.ax.set_ylabel("Frequency of Use", rotation=-90, va="bottom")
ax.set_xticks(np.arange(use_data.shape[1]))
ax.set_yticks(np.arange(use_data.shape[0]))
ax.set_xticklabels([c.strip("pattern_use") for c in use_data.columns])
ax.set_yticklabels(use_data.index)
ax.set_xlabel("Pattern")
ax.xaxis.set_label_position('top')
ax.set_ylabel("User ID")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_xticks(np.arange(use_data.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(use_data.shape[0]+1)-.5, minor=True)
ax.tick_params(which="minor", bottom=False, left=False)
fig.savefig("mozak/pattern_use_heatmap.png", bbox_inches="tight")
plt.close()
