import numpy as np
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
# import seaborn
import csv
from textwrap import wrap
from util import *

fields = ['pid', 'uid', 'rank', 'num_lines', 'num_early_lines', 'num_late_lines',
          'num_manual_sections', 'median_manual_size', 'num_optima_escapes',
          'inquis_frac', 'median_inquis_size', 'focus_best_frac', 'num_target_macro',
          'median_macro_freq', 'macro_freq_std']

pids = {'2002824', '2002872', '2002906', '2000696', '2000813', '2001066', '2001433', '2001803', '2002267', '2002549',
        '2002963'}  # data for 2003011 appears to currently be incomplete

plot_problems = [('2001433', '215720')]  # massive outlier badly distorts manual plot


def read_data(fp):
    data = []
    for r in csv.DictReader(fp, fieldnames=fields):
        for f in r:
            if f not in {'pid', 'uid'}:
                r[f] = float(r[f])
            if f == 'num_lines':
                r[f] = max(1, r[f])
        if r['pid'] in pids and (r['pid'], r['uid']) not in plot_problems:
            data.append(r)
    return data


with open("data/top_node_data_v3.csv") as fp:
    top_data = read_data(fp)

with open("data/bottom_node_data_v3.csv") as fp:
    bottom_data = read_data(fp)

# lines
output_freq_results(top_data, bottom_data, "num_lines", ("high performing", "lower performing"),
                    "number of explored hypotheses", "lines.png")

# manual
output_freq_results(top_data, bottom_data, "num_manual_sections", ("high performing", "lower performing"),
                    "number of fully manual sections", "manual.png")

# optima
output_freq_results(top_data, bottom_data, 'num_optima_escapes', ("high performing", "lower performing"),
                    "instances of optima escape behavior", "optima.png")

# inquisitive
output_freq_results(top_data, bottom_data, 'inquis_frac', ("high performing", "lower performing"),
                    "proportion of inquisitive exploration", "inquis.png")

# focus
metric = "focus_best_frac"
print(metric)
top = [x[metric] for x in top_data if not np.isnan(x[metric])]
bottom = [x[metric] for x in bottom_data if not np.isnan(x[metric])]
make_boxplot(top, bottom, ("high performing", "lower performing"), "proportion of greedy exploration", "focus.png")
# make_plot([np.mean(top)], [np.mean(bottom)], np.std(top), np.std(bottom), ("",), "mean proportion of greedy exploration per player", "focus.png")
sig_test(np.array([x[metric] for x in top_data if not np.isnan(x[metric])]),
         np.array([x[metric] for x in bottom_data if not np.isnan(x[metric])]))
print()

# targeted macro
# output_freq_results(top_data, bottom_data, 'num_target_macro', ("high performing", "lower performing"),
#                     "number of players", "target.png")

# macro freq
metric = "median_macro_freq"
print(metric)
top = [x[metric] for x in top_data if not np.isnan(x[metric])]
bottom = [x[metric] for x in bottom_data if not np.isnan(x[metric])]
make_boxplot(top, bottom, ("high performing", "lower performing"),
             "median frequency of recipe use along all solution paths", "macro_freq.png")
# make_plot([np.mean(top)], [np.mean(bottom)], np.std(top), np.std(bottom), ("",), "mean recipes used along all solution paths per player", ("high performing", "low performing"), "macro_freq.png")
sig_test(np.array([x[metric] for x in top_data if not np.isnan(x[metric])]),
         np.array([x[metric] for x in bottom_data if not np.isnan(x[metric])]))
print()

# macro consistency
# metric = "macro_freq_std"
# print(metric)
# top = [x[metric] for x in top_data if not np.isnan(x[metric])]
# bottom = [x[metric] for x in bottom_data if not np.isnan(x[metric])]
# make_plot([np.median(top)], [np.median(bottom)], ("consistency of macro use",), "median standard deviation in frequency of macro use", ("high performing", "low performing"), "macro_std.png")
# sig_test(np.array([x[metric] for x in top_data if not np.isnan(x[metric])]),
#          np.array([x[metric] for x in bottom_data if not np.isnan(x[metric])]))
# print()
