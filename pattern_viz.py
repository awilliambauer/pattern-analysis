"""
useful plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Iterable
import os
import string
from pattern_extraction import get_pattern_masks
from util import PatternInstance, PatternLookup, SubSeriesLookup


def plot_labeled_series(x: np.ndarray, y: np.ndarray, pattern_masks: Dict[str, np.ndarray], action_labels: List[str],
                        filename: str, scale="log", figsize=(100, 6)):
    """

    :param x:
    :param y:
    :param pattern_masks:
    :param action_labels:
    :param filename:
    :param scale:
    :param figsize:
    """
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(y.shape[1]):
        ax.plot(x, y[:, j], color=plt.cm.get_cmap("tab20").colors[j], alpha=0.7)
    line_legend = ax.legend(action_labels, loc="upper center", ncol=7, fancybox=True, shadow=True,
                            bbox_to_anchor=(0.5, 1.05))
    colors = plt.cm.get_cmap("viridis", len(pattern_masks)).colors
    fills = []
    pattern_labels = []
    for i, (pt, mask) in enumerate(pattern_masks.items()):
        if mask.any():
            fills.append(ax.fill_between(x, 0, y.max(), mask, color=colors[i], alpha=0.4))
            pattern_labels.append(pt)
            if mask[0]:
                ax.text(-2, y.max(), pt)
            for i, (a, b) in enumerate(zip(mask, mask[1:])):
                if b and not a:
                    ax.text(i, y.max(), pt)
    ax.set_xlim(-10, len(x) + 10)
    if len(pattern_masks) > 0:
        ax.legend(fills, pattern_labels, bbox_to_anchor=(1.01, 1), fancybox=True, shadow=True)
    ax.add_artist(line_legend)
    ax.set_yscale(scale, nonposy='clip')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_model(model_dir: str, k: int, subs: Iterable[Tuple[int, int]], all_series: np.ndarray,
               pattern_lookups: Dict[int, PatternLookup], subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]],
               action_labels: List[str]):
    """

    :param model_dir:
    :param k:
    :param subs:
    :param all_series:
    :param pattern_lookups:
    :param subseries_lookups:
    :param action_labels:
    """
    for cid, sub_k in subs:
        print("plotting", cid, sub_k)
        os.makedirs("{}/eval/viz/".format(model_dir), exist_ok=True)
        if sub_k == 0:
            ps = [p for p in pattern_lookups[k][cid][0] if p.start_idx < p.end_idx]
            ser = np.concatenate([np.concatenate((all_series[p.start_idx:p.end_idx],
                                                  np.tile(np.array([0] * all_series.shape[1]), (60, 1)))) for p in ps])
            plot_labeled_series(np.arange(len(ser)), ser, {}, action_labels,
                                "{}/eval/viz/{}_pattern.png".format(model_dir, cid), figsize=(200, 6))
        else:
            subpatterns = pattern_lookups[k][cid][sub_k]
            subcids = {p.cid for p in subpatterns}
            for sub_cid in subcids:
                label = str(cid) + string.ascii_uppercase[sub_cid]
                sub_ps = [p for p in subpatterns if p.cid == sub_cid and p.start_idx < p.end_idx]
                ser = np.concatenate([np.concatenate((subseries_lookups[k][cid]["series"][p.start_idx:p.end_idx],
                                                      np.tile(np.array([0] * all_series.shape[1]), (30, 1)))) for p in
                                      sub_ps])
                plot_labeled_series(np.arange(len(ser)), ser, {}, action_labels,
                                    "{}/eval/viz/{}_pattern.png".format(model_dir, label))


def plot_user_series(model_dir: str, k: int, subs: Iterable[Tuple[int, int]], puz_idx_lookup: dict,
                     all_series: np.ndarray, pattern_lookups: Dict[int, PatternLookup], pts: Iterable[str],
                     subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]], action_labels: List[str]):
    """

    :param model_dir:
    :param k:
    :param subs:
    :param puz_idx_lookup:
    :param all_series:
    :param pattern_lookups:
    :param pts:
    :param subseries_lookups:
    :param action_labels:
    """
    os.makedirs("{}/eval/user_series/".format(model_dir), exist_ok=True)
    for i, ((uid, pid), idx) in enumerate(puz_idx_lookup.items()):
        print("plotting user {} of {}\r".format(i, len(puz_idx_lookup)), end="")
        masks = get_pattern_masks(uid, pid, idx, pts, dict(subs), pattern_lookups[k], subseries_lookups[k])
        y = all_series[slice(*idx)]
        if len(y) > 1:
            plot_labeled_series(np.arange(len(y)), y, masks, action_labels,
                                "{}/eval/user_series/{}_{}.png".format(model_dir, uid, pid))
    print()