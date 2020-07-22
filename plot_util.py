import csv
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from util import sig_test


def make_plot(y1, y2, y1std, y2std, xlabels, ylabel, legend, filename):
    with open("data/Color_Palette_Tableau10.csv") as fp:
        raw = [row['RGB'].split('.') for row in csv.DictReader(fp)]
        colors = [(int(r) / 255, int(g) / 255, int(b) / 255) for r, g, b in raw]
    plt.clf()
    ind = np.arange(len(y1))
    width = 0.35
    b1 = plt.bar(ind + 0.05, y1, width, color=colors[0], yerr=y1std)
    b2 = plt.bar(ind + 0.05 + width, y2, width, color=colors[1], yerr=y2std)
    plt.legend((b1[0], b2[0]), legend)
    plt.xticks(ind + 0.05 + width, ["\n".join(wrap(l, 30)) for l in xlabels])
    plt.ylabel("\n".join(wrap(ylabel, 60)))
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def make_boxplot(series, categories, ylabel, filename, ylims=None, yscale="linear", markers=None):
    assert len(series) == len(categories), \
        "Each series must have a category label (provided {} series and {} categories)".format(len(series), len(categories))
    plt.clf()
    fig, ax = plt.subplots(figsize=(2 * len(series), 4))

    for i, s in enumerate(series, 1):
        x = np.random.normal(i, 0.04, size=len(s))
        if markers:
            for mark, mask in markers.items():
                plt.plot(np.array(x)[mask], np.array(s)[mask], mark, alpha=0.2)
        else:
            plt.plot(x, s, 'r.', alpha=0.2)
    
    bp = plt.boxplot(series, sym='b+')
    # bp['boxes'][0].set_facecolor(colors[0])
    # bp['boxes'][1].set_facecolor(colors[1])
    plt.setp(bp['medians'], color='black', linewidth='2.5')

    ax.set_yscale(yscale)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # plt.legend((bp['boxes'][0], bp['boxes'][1]), categories)

    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom='off',  # ticks along the bottom edge are off
    #     top='off',  # ticks along the top edge are off
    #     labelbottom='off')  # labels along the bottom edge are off
    plt.setp(ax, xticklabels=categories)
    if ylims:
        plt.ylim(*ylims)
    plt.ylabel("\n".join(wrap(ylabel, 60)))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def output_freq_results(top_data, bottom_data, metric, xlab, ylab, filename):
    print(metric)
    top = [x[metric] for x in top_data if not np.isnan(x[metric])]
    bottom = [x[metric] for x in bottom_data if not np.isnan(x[metric])]
    # make_plot([np.mean(top)], [np.mean(bottom)], np.std(top), np.std(bottom), (xlab,), ylab, filename)
    make_boxplot(top, bottom, xlab, ylab, filename)
    sig_test(np.array([x[metric] for x in top_data if not np.isnan(x[metric])]),
             np.array([x[metric] for x in bottom_data if not np.isnan(x[metric])]))
    # print(stats.chi2_contingency(np.array([[len(top), len(top_data) - len(top)],
    #                                        [len(bottom), len(bottom_data) - len(bottom)]])))
    print()
