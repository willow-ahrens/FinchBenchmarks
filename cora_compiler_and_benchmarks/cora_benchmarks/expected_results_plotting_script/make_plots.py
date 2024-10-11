import matplotlib.colors as colors
from scipy import stats
from matplotlib import cm
import pandas as pd
import matplotlib as mpl
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import common
from common import geomean
import argparse
import math

palettes = {}

suh_height = 2

import numpy as np

pd.set_option("display.max_rows", None, "display.max_columns", None)
def make_mmha_plot(csvdir, figdir, args):
    datasets = ['race', 'mnli']
    b_sizes = [32, 64, 128]
    figpath = figdir + '/mmha_eval.pdf'

    csvpath = csvdir + '/bert_layer_mmha_results_cuda.csv'
    df = pd.read_csv(csvpath)

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    pytorch_idx = 3
    cora_dense_idx = 4
    cora_masked_idx = 5
    cols = df.columns

    pytorch_rel_time = 'PyTorch'
    cora_dense_rel_time = 'CoRa-Pad'
    cora_masked_rel_time = 'CoRa-NoPad'
    framework = 'Framework'
    exe_time = 'Relative Exe. Time'

    race_128_df = df.loc[df[cols[dataset_idx]].isin(['race']) & df[cols[b_size_idx]].isin([128])]
    mnli_128_df = df.loc[df[cols[dataset_idx]].isin(['mnli']) & df[cols[b_size_idx]].isin([128])]
    print('RACE,CD,128', (race_128_df[cols[cora_dense_idx]] / race_128_df[cols[cora_masked_idx]]).tolist()[0])
    print('MNLI,CD,128', (mnli_128_df[cols[cora_dense_idx]] / mnli_128_df[cols[cora_masked_idx]]).tolist()[0])

    print('PT', geomean((df[cols[pytorch_idx]] / df[cols[cora_masked_idx]]).tolist()))
    print('CD', geomean((df[cols[cora_dense_idx]] / df[cols[cora_masked_idx]]).tolist()))

    df = df.loc[(df[cols[dataset_idx]].isin(datasets))]
    df = df.loc[(df[cols[b_size_idx]].isin(b_sizes))]
    df[pytorch_rel_time] = df[cols[pytorch_idx]] / df[cols[pytorch_idx]]
    df[cora_dense_rel_time] = df[cols[cora_dense_idx]] / df[cols[pytorch_idx]]
    df[cora_masked_rel_time] = df[cols[cora_masked_idx]] / df[cols[pytorch_idx]]
    df = df.drop(columns = [cols[target_idx], cols[pytorch_idx], cols[cora_dense_idx], cols[cora_masked_idx]])

    df = df.melt(id_vars=[cols[dataset_idx], cols[b_size_idx]], var_name = framework, value_name = exe_time)
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    height = 1.5
    g = sns.catplot(
        data=df, kind="bar",
        x=cols[b_size_idx], y=exe_time, hue=framework, col=cols[dataset_idx],
        ci=None, height=height, aspect = 5/(2*height), legend = True, palette = palettes[3],
        sharey = False, sharex = True, legend_out=False)

    axes = g.axes.flatten()
    axes[0].set_title(axes[0].get_title().replace('Dataset = ', ''))
    axes[1].set_title(axes[1].get_title().replace('Dataset = ', ''))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=
                   labels, loc='upper center',
                   bbox_to_anchor = (1, 1.65), ncol=3,
                   labelspacing=0.3, borderpad=0.4)

    if args.debug: plt.show()

    g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_memory_plot(csvdir, figdir, args):
    figpath = figdir + '/mem_eval.pdf'
    csvpath = csvdir + '/bert_layer_mem_results_cuda.csv'
    df = pd.read_csv(csvpath)

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    dense_idx = 3
    ragged_idx = 4
    cols = df.columns

    framework = 'Framework'
    mem_used = 'Relative''\n''Mem. Usage'
    batch_size = 64
    df = df.drop(columns = [cols[target_idx]])
    df = df.loc[df[cols[b_size_idx]] == batch_size]
    df[cols[ragged_idx]] = df[cols[ragged_idx]] / df[cols[dense_idx]]
    df[cols[dense_idx]] = df[cols[dense_idx]] / df[cols[dense_idx]]
    print('Improvement: ', geomean([1/i for i in df[cols[ragged_idx]].tolist()]))


    df = df.melt(id_vars=[cols[dataset_idx], cols[b_size_idx]], var_name = framework, value_name = mem_used)
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)
    df[framework] = df[framework].apply(lambda s: s.replace(' (ms)', ''))
    df[framework] = df[framework].apply(common.get_framework_repl)

    plt.figure(figsize = (5, 1.2))
    g = sns.barplot(data=df, x=cols[dataset_idx], y=mem_used, hue=framework, ci=None,
        palette = palettes[2])

    plt.xticks(rotation=5)
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, ncol=2, loc='lower left', bbox_to_anchor=(0.04, 0.02))
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)

    if args.debug: plt.show()

    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_pad_fusion_plot(csvdir, figdir, args):
    figpath = figdir + '/pad_fusion_eval.pdf'
    csvpath = csvdir + '/pad_fusion_results_cuda.csv'
    df = pd.read_csv(csvpath)

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    unfused_idx = 3
    fused_idx = 4
    cols = df.columns
    b_sizes = [32, 64, 128]
    datasets = ['mnli']

    framework = 'Variant'
    exe_time = 'Execution Time (ms)'
    df = df.loc[(df[cols[b_size_idx]].isin(b_sizes)) & (df[cols[dataset_idx]].isin(datasets))]
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    exe_time = 'Rel. Exe. Time'
    unfused_rel_time = "Unfused Rel. Time"
    fused_rel_time = "Fused Rel. Time"

    df[unfused_rel_time] = df[cols[unfused_idx]] / df[cols[unfused_idx]]
    df[fused_rel_time] = df[cols[fused_idx]] / df[cols[unfused_idx]]
    df = df.drop(columns = [cols[target_idx], cols[fused_idx], cols[unfused_idx]])

    df = df.rename(columns = {
        unfused_rel_time: 'Unfused',
        fused_rel_time: 'Fused',
    })

    df = df.melt(id_vars=[cols[dataset_idx], cols[b_size_idx]], var_name = framework, value_name = exe_time)
    df[framework] = df[framework].apply(lambda s: s.replace(' (ms)', ''))

    print(df)
    plt.figure(figsize = (2, 0.9))
    g = sns.barplot(data=df, x=cols[b_size_idx], y=exe_time,
                    hue=framework, ci=None, palette=palettes[3])

    g.set_title(g.get_title().replace('Dataset = ', ''))
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.02), loc='lower center')
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.set_ylabel(exe_time, fontsize=9)

    if args.debug: plt.show()

    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_trmm_plot(csvdir, figdir, args):
    m_sizes = [1 << i for i in range(7, 14)]
    figpath = figdir + '/trmm_eval.pdf'

    m_sizes = [512, 1024, 2048, 4096, 8192]

    sq_op = 'Sq'
    op_idx = 0
    target_idx = 1
    m_idx = 2
    n_idx = 3
    cublas_trmm_idx = 4
    cublas_dense_idx = 5
    cora_unsplit_idx = 6
    cora_unbalanced_idx = 7
    cora_balanced_idx = 8
    taco_idx = 9

    relative_cublas_trmm_col = 'CuBLAS trmm'
    relative_cublas_dense_col = 'CuBLAS sgemm'
    relative_cora_trmm_unsplit_col = 'CoRa-UnSplit-Unbalanced'
    relative_cora_trmm_unbalanced_col = 'CoRa-Split-Unbalanced'
    relative_cora_trmm_balanced_col = 'CoRa-Split-Balanced'

    csvpath = csvdir + '/trmm_results_cuda.csv'
    df = pd.read_csv(csvpath)
    cols = df.columns

    df = df.loc[df[cols[op_idx]] == sq_op]
    df = df.loc[df[cols[m_idx]].isin(m_sizes)]

    df[relative_cublas_dense_col] = df[cols[cublas_trmm_idx]] / df[cols[cublas_dense_idx]]
    df[relative_cora_trmm_unsplit_col] = df[cols[cublas_trmm_idx]] / df[cols[cora_unsplit_idx]]
    df[relative_cora_trmm_unbalanced_col] = df[cols[cublas_trmm_idx]] / df[cols[cora_unbalanced_idx]]
    df[relative_cora_trmm_balanced_col] = df[cols[cublas_trmm_idx]] / df[cols[cora_balanced_idx]]
    df[relative_cublas_trmm_col] = df[cols[cublas_trmm_idx]] / df[cols[cublas_trmm_idx]]

    print(min(df[relative_cora_trmm_balanced_col].tolist()))

    df = df.drop(columns = [cols[op_idx], cols[n_idx], cols[target_idx], cols[cublas_trmm_idx], cols[cublas_dense_idx],
                            cols[cora_unsplit_idx], cols[cora_unbalanced_idx], cols[cora_balanced_idx]])

    variant_col = "Variant"
    speedup_col = "Speedup"

    print(df)

    df = df.melt(id_vars=[cols[m_idx]], var_name = variant_col, value_name = speedup_col)

    x_col_name = "GEMM Size"
    df = df.rename(columns={cols[m_idx]: x_col_name})
    plt.figure(figsize = (5, 1.2))
    g = sns.barplot(data=df, x=x_col_name, y=speedup_col, hue=variant_col, ci=None,
        palette = palettes[5])

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, fontsize='x-small', ncol=2, borderpad=0.2, columnspacing=0.8, labelspacing=0.4)
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)

    if args.debug: plt.show()
    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)


def make_partial_pad_overheads_plot(csvdir, figdir, args):
    m_sizes = [1 << i for i in range(7, 14)]
    figpath = figdir + '/partial_pad_overheads_eval.pdf'

    dataset_idx = 0
    batch_size_idx = 1
    dense_flops_idx = 2
    real_flops_idx = 3
    ragged_flops_idx = 4
    flops_ratio = 'FLOPS Ratio'

    b_sizes = [32, 128]
    csvpath = csvdir + '/intro_flops.csv'
    df = pd.read_csv(csvpath)
    cols = df.columns
    df[cols[batch_size_idx]] = df[cols[batch_size_idx]].apply(lambda a: pow(2, a))
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    relative_dense_col = 'Dense Computation'
    relative_real_col = 'Actual Computation'
    relative_ragged_col = 'Ideal Computation'

    df[relative_dense_col] = df[cols[dense_flops_idx]] / df[cols[ragged_flops_idx]]
    df[relative_real_col] = df[cols[real_flops_idx]] / df[cols[ragged_flops_idx]]
    df[relative_ragged_col] = df[cols[ragged_flops_idx]] / df[cols[ragged_flops_idx]]
    df = df.loc[df[cols[batch_size_idx]].isin(b_sizes)]
    print("Partial Padding Overhead 32:", geomean(df.loc[df[cols[batch_size_idx]].isin([32])][relative_real_col].tolist()))
    print("Partial Padding Overhead 128:", geomean(df.loc[df[cols[batch_size_idx]].isin([128])][relative_real_col].tolist()))
    df = df.drop(columns = [cols[dense_flops_idx], cols[real_flops_idx], cols[ragged_flops_idx]])

    variant_col = "Variant"
    speedup_col = "Relative""\n""Overheads"

    df = df.melt(id_vars=[cols[dataset_idx], cols[batch_size_idx]], var_name = variant_col, value_name = speedup_col)

    x_col_name = "GEMM Size"
    height = 1.4
    g = sns.catplot(
        data=df, kind="bar",
        x=cols[dataset_idx], y=speedup_col, hue=variant_col, row=cols[batch_size_idx],
        ci=None, height=height, aspect=5/(height), legend=False, sharey=True, sharex=True,
        palette = palettes[3], margin_titles=False)

    axes = g.axes.flatten()
    ax = axes[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, ncol=3, loc='upper center', bbox_to_anchor=(0.50, 1.6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=5)

    if args.debug: plt.show()
    g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)


def make_bert_layer_summary_plot(csvdir, figdir, args):
    figpath = figdir + '/layer_summary_eval.pdf'

    b_sizes = [32, 64, 128]
    num_datasets = 8

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    pytorch_idx = 3
    ftrans_pad_idx = 4
    ftrans_nopad_idx = 5
    cora_idx = 6

    num_layers = 6

    pytorch_relative_col = "PyTorch"
    ftrans_pad_relative_col = "FT"
    ftrans_nopad_relative_col = "FT-Eff"
    cora_relative_col = "CoRa"

    csvpath = csvdir + '/bert_layer_results_cuda.csv'
    pcsvpath = csvdir + '/bert_layer_prelude_results_cuda.csv'
    df = pd.read_csv(csvpath)
    cols = df.columns
    pdf = pd.read_csv(pcsvpath)
    df = df.loc[df[cols[b_size_idx]].isin(b_sizes)]
    df = df.reset_index()

    pcol = pdf[cols[cora_idx]] * ((num_layers - 1) / num_layers)
    updated_cora_times = df[cols[cora_idx]] - pcol
    df[cols[cora_idx]] = updated_cora_times

    df[pytorch_relative_col] = df[cols[pytorch_idx]] / df[cols[cora_idx]]
    df[ftrans_pad_relative_col] = df[cols[ftrans_pad_idx]] / df[cols[cora_idx]]
    df[ftrans_nopad_relative_col] = df[cols[ftrans_nopad_idx]] / df[cols[cora_idx]]
    df[cora_relative_col] = df[cols[cora_idx]] / df[cols[cora_idx]]

    py_speedup = df[pytorch_relative_col].tolist()
    ftp_speedup = df[ftrans_pad_relative_col].tolist()
    ftn_speedup = df[ftrans_nopad_relative_col].tolist()

    print("PyTorch Speedup:", geomean(py_speedup), max(py_speedup))
    print("FT Speedup:", geomean(ftp_speedup), max(ftp_speedup))
    print("FT-Eff Speedup:", geomean(ftn_speedup), max(ftn_speedup))

    df = df.drop(columns = [cols[target_idx], cols[pytorch_idx], cols[ftrans_pad_idx],
                            cols[ftrans_nopad_idx], cols[cora_idx]])

    df = df.groupby(cols[b_size_idx]).prod().reset_index()
    df[pytorch_relative_col] = df[pytorch_relative_col].apply(lambda a: pow(a, 1/num_datasets))
    df[ftrans_pad_relative_col] = df[ftrans_pad_relative_col].apply(lambda a: pow(a, 1/num_datasets))
    df[ftrans_nopad_relative_col] = df[ftrans_nopad_relative_col].apply(lambda a: pow(a, 1/num_datasets))
    df[cora_relative_col] = df[cora_relative_col].apply(lambda a: pow(a, 1/num_datasets))

    variant_col = "Framework"
    speedup_col = "Rel. Exe. Time"

    df = df[[cols[b_size_idx], pytorch_relative_col, ftrans_pad_relative_col, cora_relative_col, ftrans_nopad_relative_col]]
    df = df.melt(id_vars=[cols[b_size_idx]], var_name = variant_col, value_name = speedup_col)

    x_col_name = "GEMM Size"
    plt.figure(figsize = (2, 0.9))
    g = sns.barplot(data=df, x=cols[b_size_idx], y=speedup_col, hue=variant_col, ci=None,
        palette = palettes[4])

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, fontsize=8,
             ncol=2, columnspacing=0.6, loc='lower center',
             bbox_to_anchor=(0.50, -0.05), borderpad=0.2,
             handletextpad=0.3, labelspacing=0.2)
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.set_ylabel(speedup_col, fontsize=9)
    g.set_ylim(top=2)

    if args.debug: plt.show()
    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)


def make_vbatch_gemm_plot(csvdir, figdir, args):
    frameworks = ['Ragged-HandOptimized', 'Ragged-CoRa', 'FullyPadded-HandOptimized']
    backends = ['cuda', 'cpu']
    figpath = figdir + '/vbatch_gemm_eval.pdf'
    b_sizes = [1 << i for i in range(1, 10)]

    target_idx = 0
    b_size_idx = 1
    lb_idx = 2
    ub_idx = 3
    cora_idx = 4

    backend_col = 'Backend'
    relative_lb_col = 'Ragged-HandOptimized'
    relative_cora_col = 'Ragged-CoRa'
    relative_coramkl_col = 'Ragged-CoRa-MKLCore'
    relative_ub_col = 'FullyPadded-HandOptimized'
    variant_col = "Variant"
    speedup_col = "Speedup"
    all_dfs = {}
    for backend in backends:
        csvpath = csvdir + '/vbatch_gemm_results_' + backend + '.csv'
        df = pd.read_csv(csvpath)
        cols = df.columns
        df = df.rename(columns={cols[lb_idx]: 'lb', cols[ub_idx]: 'ub'})
        # if backend == "cpu":
            # df = df.drop(columns=cols[cora_idx])
            # df = df.rename(columns={cols[cora_mkl_idx]: cols[cora_idx]})
            # print(df)
        all_dfs[backend] = df

    df = pd.concat(all_dfs)
    cols = df.columns
    df = df.loc[df[cols[b_size_idx]].isin(b_sizes)]
    df[relative_lb_col] = df[cols[lb_idx]] / df[cols[lb_idx]]
    df[relative_cora_col] = df[cols[lb_idx]] / df[cols[cora_idx]]
    # df[relative_coramkl_col] = df[cols[lb_idx]] / df[cols[cora_mkl_idx]]
    df[relative_ub_col] = df[cols[lb_idx]] / df[cols[ub_idx]]

    print("CoRa Least Perf:", min(df[relative_cora_col].tolist()))

    df = df.drop(columns = [cols[lb_idx], cols[ub_idx], cols[cora_idx]])

    df[cols[b_size_idx]] = df[cols[b_size_idx]].astype(str)

    df = df.melt(id_vars=[cols[target_idx], cols[b_size_idx]],
                 var_name=variant_col,
                 value_name=speedup_col)
    # print(df)

    fig, ax = plt.subplots()
    height = 1.6
    plt.figure(figsize = (5, height))

    with sns.plotting_context(rc={"legend.loc": 'upper left', 'legend.fontsize': 6}):
        g=sns.relplot(data=df, x=cols[b_size_idx], y=speedup_col, hue=variant_col, col=cols[target_idx],
                        ci=None, marker='o', facet_kws={'sharey': False, 'sharex': True, 'legend_out': False},
                        kind="line", height=height, aspect=5.0/(2*height), legend='full',
                        palette=palettes[3])

        axes = g.axes.flatten()
        axes[0].set_title(axes[0].get_title().replace('Target = cuda', 'GPU'))
        axes[1].set_title(axes[1].get_title().replace('Target = cpu', 'Intel CPU'))
        axes[0].grid()
        axes[1].grid()

        handles, labels = axes[0].get_legend_handles_labels()
        # axes[0].legend(handles=handles[1:], labels=labels[1:])
        axes[0].legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(0.2, 0.38),
                       borderpad=0.2, labelspacing=0.4)
        # axes[1].legend(handles=handles, labels=labels)

        print(type(axes[0]))
        print(type(g))

        plt.xticks(ticks=list(range(8)), labels=[1 << i for i in range(8)])

        axes[0].set_xticks(range(len(b_sizes)))
        axes[0].set_xticklabels(b_sizes, rotation='vertical')

        axes[1].set_xticks(range(len(b_sizes)))
        axes[1].set_xticklabels(b_sizes, rotation='vertical')

        if args.debug: plt.show()
        g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_intro_flops_plot(csvdir, figdir, args):
    figpath = figdir + '/intro_flops.pdf'

    csvpath = csvdir + '/intro_flops.csv'
    df = pd.read_csv(csvpath)

    dataset_idx = 0
    batch_size_idx = 1
    dense_flops_idx = 2
    real_flops_idx = 3
    ragged_flops_idx = 4
    flops_ratio = 'Wasted' + '\n' + 'Computation'

    cols = df.columns
    df = df.drop(columns=[cols[real_flops_idx]])
    # df[flops_ratio] = df[cols[ragged_flops_idx]] / df[cols[dense_flops_idx]]
    df[flops_ratio] = df[cols[dense_flops_idx]] / df[cols[ragged_flops_idx]]
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    fig, ax = plt.subplots()
    height = 1.2
    width = 5.0
    plt.figure(figsize = (width, height))
    with sns.plotting_context(rc={"legend.loc": 'upper left', 'legend.fontsize': 7}):
        g = sns.lineplot(data=df, x=cols[batch_size_idx], y=flops_ratio, hue=cols[dataset_idx], estimator=None)
        handles, labels = g.get_legend_handles_labels()
        g.legend(handles=handles, labels=labels, ncol=4, columnspacing=0.5, handletextpad=0.5)
        g.set_ylim(bottom=0.8, top=3)

        plt.xticks(ticks=list(range(8)), labels=[1 << i for i in range(8)])
        if args.debug: plt.show()
        g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_binpack_plot_old(csvdir, figdir, args):
    csvpath = csvdir + '/bin_packed_results_cuda.csv'
    figpath = figdir + '/bin_packed_eval.pdf'
    ops = ['qkt', 'attn_v']
    datasets = ['mnli', 'random_80_96']

    op_idx = 0
    target_idx = 1
    dataset_idx = 2
    b_size_idx = 3
    vanilla_idx = 4
    op_split_idx = 5
    hfuse_idx = 6
    cbt_idx = 7

    inf_lat = "Execution Time (ms)"
    opts = "Optimizations"

    df = pd.read_csv(csvpath)
    cols = df.columns
    df = df.drop(columns = [cols[target_idx]])

    # def sorter(series):
    #     if series.name == cols[model_idx]:
    #         return series.map({'fc':1, 'dag_rnn':2, 'gru':3, 'lstm':4, 'mvrnn':5})
    #     else: return series
    # df = df.sort_values(by = [cols[model_idx], cols[b_size_idx]], key = sorter)
    df = df.loc[df[cols[op_idx]].isin(ops)]
    df = df.loc[df[cols[dataset_idx]].isin(datasets)]
    df = df[[cols[op_idx], cols[dataset_idx], cols[b_size_idx], cols[cbt_idx],
             cols[vanilla_idx], cols[op_split_idx], cols[hfuse_idx]]]

    df[cols[op_idx]] = df[cols[op_idx]].apply(common.get_op_repl)
    df = df.melt(id_vars=[cols[op_idx], cols[dataset_idx], cols[b_size_idx]],
                 var_name = opts, value_name = inf_lat)


    height = 1.5
    g = sns.catplot(
        data=df, kind="bar",
        x=cols[b_size_idx], y=inf_lat, hue=opts, col=cols[op_idx], # row=cols[op_idx],
        ci=None, height=height, aspect=5/(2*height), legend = False, sharey = False, sharex = True,
        palette = palettes[4], margin_titles=False)

    axes = g.axes.flatten()
    for axis in axes:
        axis.set_title(axis.get_title().replace('Op = ', ''))
        axis.set_title(axis.get_title().replace('Dataset = ', ''))
        axis.grid(axis = 'y')

    # g.set_xlabels('')
    # g.fig.text(0.47, 0.072, 'Batch Size')
    plt.legend(loc='upper right', bbox_to_anchor = (0.95, 1.65), ncol=4,
               fontsize = 'small', labels = ['CBT', 'CoRa-NoSplit', 'CoRa-Split', 'CoRa-Split-HFused'])
    # plt.subplots_adjust(wspace=0.6)


    if args.debug: plt.show()
    g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_binpack_plot(csvdir, figdir, args):
    targets = ['cuda', 'cpu']
    ops = ['qkt', 'attn_v']
    datasets = ['mnli']

    all_dfs = {}
    for target in targets:
        csvpath = csvdir + '/bin_packed_results_' + target + '.csv'
        this_df = pd.read_csv(csvpath)
        all_dfs[target] = this_df

    full_df = pd.concat(all_dfs)

    for op in ops:
        figpath = figdir + '/bin_packed_' + op + '_eval.pdf'

        op_idx = 0
        target_idx = 1
        dataset_idx = 2
        b_size_idx = 3
        vanilla_idx = 4
        op_split_idx = 5
        hfuse_idx = 6
        cbt_idx = 7
        split2_idx = 8

        inf_lat = 'Relative''\n''Exe. Time'
        opts = "Optimizations"

        cols = full_df.columns
        df = full_df.loc[full_df[cols[op_idx]].isin([op])]
        # df = full_df.drop(columns = [cols[op_idx]])
        df = df.loc[df[cols[dataset_idx]].isin(datasets)]

        split_rel_col = 'SplitRel'
        hfuse_rel_col = 'HFuseRel'
        cbt_rel_col = 'CBTRel'
        vanilla_rel_col = 'Vanilla'

        df[split_rel_col] = df[cols[op_split_idx]] / df[cols[vanilla_idx]]
        df[hfuse_rel_col] = df[cols[hfuse_idx]] / df[cols[vanilla_idx]]
        if target == 'cuda':
            df[cbt_rel_col] = df[cols[cbt_idx]] / df[cols[vanilla_idx]]
        df[vanilla_rel_col] = df[cols[vanilla_idx]] / df[cols[vanilla_idx]]

        df = df[[cols[target_idx], cols[dataset_idx], cols[b_size_idx],
                 vanilla_rel_col, split_rel_col, hfuse_rel_col]]

        print(df)
        df[cols[b_size_idx]] = df[cols[b_size_idx]].astype(str)

        df[cols[target_idx]] = df[cols[target_idx]].apply(common.get_backend_repl)
        df = df.melt(id_vars=[cols[target_idx], cols[dataset_idx], cols[b_size_idx]],
                     var_name = opts, value_name = inf_lat)


        height = 1.3
        g = sns.relplot(data=df, x=cols[b_size_idx], y=inf_lat, hue=opts,
                        col=cols[target_idx], ci=None, marker = 'o',
                        legend=False, facet_kws={'sharey': False,
                                                 'sharex': True, 'legend_out': False}, kind =
                        "line", height=height, aspect = 5.0/(2*height),
                        palette = palettes[3])
        axes = g.axes.flatten()
        for axis in axes:
            axis.set_title(axis.get_title().replace('Op = ', ''))
            axis.set_title(axis.get_title().replace('Target = ', ''))
            axis.set_title(axis.get_title().replace('Intel CPU', 'ARM CPU'))
            axis.grid(axis = 'y')
            axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
            # if target == 'cuda':
                # axis.yaxis.set_ticks([0.75, 1, 1.25, 1.5])

        plt.legend(loc='upper right', bbox_to_anchor = (0.4, 2.05),
                   ncol=3, fontsize = 'small', labels = ['NoSplit',
                                                         'Split', 'Split-HFused'], borderpad=0.2)

        if args.debug: plt.show()
        g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_qkt_split12_binpack_plot(csvdir, figdir, args):
    targets = ['cuda', 'cpu']
    ops = ['qkt']
    datasets = ['mnli']

    all_dfs = {}
    for target in targets:
        csvpath = csvdir + '/bin_packed_results_' + target + '.csv'
        this_df = pd.read_csv(csvpath)
        all_dfs[target] = this_df

    full_df = pd.concat(all_dfs)

    for op in ops:
        figpath = figdir + '/split12_bin_packed_' + op + '_eval.pdf'

        op_idx = 0
        target_idx = 1
        dataset_idx = 2
        b_size_idx = 3
        vanilla_idx = 4
        op_split_idx = 5
        hfuse_idx = 6
        cbt_idx = 7
        split2_idx = 8

        inf_lat = "Rel. Exe. Time"
        opts = "Optimizations"

        cols = full_df.columns
        df = full_df.loc[full_df[cols[op_idx]].isin([op])]
        df = df.loc[df[cols[dataset_idx]].isin(datasets)]
        # df = full_df.drop(columns = [cols[op_idx], cols[dataset_idx]])

        shf1_rel_col = 'SplitHFuseRel1'
        shf2_rel_col = 'SplitHFuseRel2'
        vanilla_rel_col = 'Vanilla'

        df[shf1_rel_col] = df[cols[hfuse_idx]] / df[cols[vanilla_idx]]
        df[shf2_rel_col] = df[cols[split2_idx]] / df[cols[vanilla_idx]]
        df[vanilla_rel_col] = df[cols[vanilla_idx]] / df[cols[vanilla_idx]]

        df = df[[cols[target_idx], cols[b_size_idx],
                 vanilla_rel_col, shf1_rel_col, shf2_rel_col]]
        df[cols[b_size_idx]] = df[cols[b_size_idx]].astype(str)

        df[cols[target_idx]] = df[cols[target_idx]].apply(common.get_backend_repl)
        print(df)
        df = df.melt(id_vars=[cols[target_idx], cols[b_size_idx]],
                     var_name = opts, value_name = inf_lat)

        height = 1.5
        g = sns.relplot(data=df, x=cols[b_size_idx], y=inf_lat, hue=opts,
                        col=cols[target_idx], ci=None, marker = 'o',
                        legend=False, facet_kws={'sharey': False,
                                                 'sharex': True, 'legend_out': False}, kind =
                        "line", height=height, aspect = 5.0/(2*height),
                        palette = palettes[3])
        axes = g.axes.flatten()
        for axis in axes:
            axis.set_title(axis.get_title().replace('Op = ', ''))
            axis.set_title(axis.get_title().replace('Target = ', ''))
            axis.grid(axis = 'y')
            axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
            # if target == 'cuda':
                # axis.yaxis.set_ticks([0.75, 1, 1.25, 1.5])

        plt.legend(loc='upper right', bbox_to_anchor = (0.6, 1.65), ncol=3,
                   fontsize = 'small', labels = ['NoSplit', 'Split1-HFused', 'Split2-HFused'])

        if args.debug: plt.show()
        g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_ragged_overheads_plot(csvdir, figdir, args):
    csvpath = csvdir + '/ragged_overheads_results_cuda.csv'
    figpath = figdir + '/ragged_overheads_eval.pdf'
    ops = ['pre_linear', 'qkt', 'softmax', 'attn_v', 'post_linear']

    op_idx = 0
    target_idx = 1
    b_size_idx = 2
    dense_idx = 3
    vloop_idx = 4
    vdim_idx = 5
    hoisted_idx = 6
    b_size = 64

    inf_lat = "Exe. Time (ms)"
    opts = "Optimizations"
    dense_relative = 'Dense Relative'
    vloop_relative = '+vloop Relative'
    vdim_relative = '+vdim Relative'
    hoisted_relative = '+LoadHoist Relative'

    df = pd.read_csv(csvpath)
    cols = df.columns
    df = df.drop(columns = [cols[target_idx]])

    df = df.loc[df[cols[b_size_idx]] == b_size]
    df = df.loc[[4,1,2,3,0]]
    df[cols[op_idx]] = df[cols[op_idx]].apply(common.get_op_repl)
    df[vloop_relative] = df[cols[vloop_idx]]
    df[vdim_relative] = df[cols[vdim_idx]]
    df[hoisted_relative] = df[cols[hoisted_idx]]
    df[dense_relative] = df[cols[dense_idx]]
    df = df[[cols[op_idx], dense_relative, vloop_relative, vdim_relative, hoisted_relative]]

    df = df.melt(id_vars=[cols[op_idx]],
                 var_name = opts, value_name = inf_lat)

    g = sns.catplot(
        data=df, kind="bar",
        x=cols[op_idx], y=inf_lat, hue=opts,
        ci=None, height=1.4, aspect=4/1.4, legend=False, sharey = False, sharex = False,
        palette = palettes[4], margin_titles=False)

    axes = g.axes.flatten()
    for axis in axes:
        axis.set_xlabel('Operator')

    plt.legend(loc='upper right', bbox_to_anchor = (0.9, 1.17), ncol=2,
               labels = ['Dense', '+vloops', '+vdims', '+LoadHoist'])

    # if args.debug: plt.show()
    g.fig.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_per_ops_plot(csvdir, figdir, args):
    dataset_idx = 0
    b_size_idx = 1
    framework_idx = 2
    op_idx = 3
    time_idx = 4
    datasets = ['race', 'cola']
    b_sizes = [32, 128]

    full_df = pd.read_csv(csvdir + '/per_op_times_results_cuda.csv')
    prelude_df = pd.read_csv(csvdir + '/per_op_times_prelude_results_cuda.csv')
    cols = full_df.columns
    num_layers = 6

    for dataset in datasets:
        figpath = figdir + '/per_ops_plot_' + dataset + '.pdf'
        df = full_df.loc[(full_df[cols[dataset_idx]] == dataset) & (full_df[cols[b_size_idx]].isin(b_sizes))]
        df = df.drop(columns = [cols[dataset_idx], cols[b_size_idx]])

        cr = 'cora'
        fp = 'ftrans_pad'
        fn = 'ftrans_nopad'

        def id(f, op):
            tmp = df.loc[(df[cols[framework_idx]] == f) & (df[cols[op_idx]] == op)]
            time = tmp[cols[time_idx]].tolist()[0]
            if f == cr:
                prelude_tmp = prelude_df.loc[(prelude_df[cols[framework_idx]] == f) &
                                             (prelude_df[cols[op_idx]] == op) &
                                             (prelude_df[cols[dataset_idx]] == dataset)]
                prelude_time = prelude_tmp[cols[time_idx]].tolist()[0]
                time -= ((num_layers - 1) / num_layers) *prelude_time
            return time

        ops_order = [
            'Proj1',
            'QKt',
            'Softmax',
            'AttnV',
            'Proj2',
            'FF1',
            'FF2',
        ]

        cr_list = {
            'Proj1': id(cr, 'pre_linear'),
            'QKt': id(cr, 'qkt'),
            'Softmax': id(cr, 'softmax'),
            'AttnV': id(cr, 'attn_v'),
            'Proj2': id(cr, 'post_linear') + id(cr, 'norm_add1'),
            'FF1': id(cr, 'ff1'),
            'FF2': id(cr, 'ff2') + id(cr, 'norm_add2'),
        }

        fp_list = {
            'Proj1': id(fp, 'AttnQKV') + id(fp, 'AttnQKVBiasAddPadding'),
            'QKt': id(fp, 'AttnQKt'),
            'Softmax': id(fp, 'AttnSoftmax'),
            'AttnV': id(fp, 'AttnAttnV'),
            'Proj2': id(fp, 'PostLinearMB') + id(fp, 'PostLinearBiasNormAdd') + id(fp, 'AttnTransposeRemPad'),
            'FF1': id(fp, 'FF1MM') + id(fp, 'FF1BiasAct'),
            'FF2': id(fp, 'FF2MM') + id(fp, 'FF2BiasNormAdd')
        }

        fn_list = {
            'Proj1': id(fn, 'AttnQKV') + id(fn, 'AttnQKVBiasAddPadding'),
            'QKt': id(fn, 'AttnQKt'),
            'Softmax': id(fn, 'AttnSoftmax'),
            'AttnV': id(fn, 'AttnAttnV'),
            'Proj2': id(fn, 'PostLinearMB') + id(fn, 'PostLinearBiasNormAdd') + id(fn, 'AttnTransposeRemPad'),
            'FF1': id(fn, 'FF1MM') + id(fn, 'FF1BiasAct'),
            'FF2': id(fn, 'FF2MM') + id(fn, 'FF2BiasNormAdd')
        }

        cr_list = [('CoRa', o, cr_list[o]) for o in ops_order]
        fp_list = [('FT', o, fp_list[o]) for o in ops_order]
        fn_list = [('FT-Eff', o, fn_list[o]) for o in ops_order]
        ops_list = cr_list + fp_list + fn_list
        df = pd.DataFrame(ops_list, columns=['Framework', 'Op', 'Score'])

        df = df.pivot(columns='Op',index='Framework').fillna(0)
        df = df.reindex(ops_order, axis=1, level='Op')
        df = df.loc[['CoRa', 'FT-Eff', 'FT']]
        print(df)

        def truncate_colormap(cmap, minval=0, maxval=0.85, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap

        cmap = truncate_colormap(cm.get_cmap('plasma', 20))
        g = df.plot.barh(stacked=True, legend=True, colormap=cmap, figsize=(5, 0.7), width=0.66)

        handles, labels = g.get_legend_handles_labels()
        labels = [l.split(', ')[1][0:-1] for l in labels]
        labels[1] = r'$\mathrm{QK}^{\mathrm{T}}$'
        # g.legend(handles=handles, labels=labels, fontsize='x-small', ncol=7, loc='upper center', bbox_to_anchor=(0.39, 1.3))
        g.legend(handles=handles, labels=labels, ncol=4, loc='upper center', bbox_to_anchor=(0.39, 1.72),
                 labelspacing=0.23, borderpad=0.05)
        g.set_xlabel('Execution Time (ms)')
        g.set_ylabel('')
        g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_per_ops_arm_plot(csvdir, figdir, args):
    dataset_idx = 0
    b_size_idx = 1
    framework_idx = 2
    op_idx = 3
    time_idx = 4
    datasets = ['mnli', 'wiki_128', 'race']
    b_sizes = [32, 128]

    full_df = pd.read_csv(csvdir + '/per_op_times_results_cpu.csv')
    cols = full_df.columns

    for dataset in datasets:
        figpath = figdir + '/per_ops_plot_cpu_' + dataset + '.pdf'
        df = full_df.loc[(full_df[cols[dataset_idx]] == dataset) & (full_df[cols[b_size_idx]].isin(b_sizes))]
        df = df.drop(columns = [cols[dataset_idx], cols[b_size_idx]])
        print(df)

        cr = 'cora'
        pt = 'pytorch'
        tf = 'tensorflow'

        def id(f, op):
            tmp = df.loc[(df[cols[framework_idx]] == f) & (df[cols[op_idx]] == op)]
            time = tmp[cols[time_idx]].tolist()[0]
            return time

        ops_order = [
            'Proj1',
            'QKt',
            'Softmax',
            'AttnV',
            'Proj2',
            'PadChange'
        ]

        cr_list = {
            'Proj1': id(cr, 'pre_linear'),
            'QKt': id(cr, 'qkt'),
            'Softmax': id(cr, 'softmax'),
            'AttnV': id(cr, 'attn_v'),
            'Proj2': id(cr, 'post_linear'),
            'PadChange': id(cr, 'add_pad') + id(cr, 'rem_pad')
        }

        pt_list = {
            'Proj1': id(pt, 'pre_linear'),
            'QKt': id(pt, 'qkt'),
            'Softmax': id(pt, 'softmax'),
            'AttnV': id(pt, 'attn_v'),
            'Proj2': id(pt, 'post_linear'),
            'PadChange': 0
        }

        tf_list = {
            'Proj1': id(tf, 'pre_linear'),
            'QKt': id(tf, 'qkt'),
            'Softmax': id(tf, 'softmax'),
            'AttnV': id(tf, 'attn_v'),
            'Proj2': id(tf, 'post_linear'),
            'PadChange': 0
        }

        cr_list = [('CoRa', o, cr_list[o]) for o in ops_order]
        pt_list = [('PyTorch', o, pt_list[o]) for o in ops_order]
        tf_list = [('TensorFlow', o, tf_list[o]) for o in ops_order]
        ops_list = cr_list + pt_list + tf_list
        df = pd.DataFrame(ops_list, columns=['Framework', 'Op', 'Score'])

        df = df.pivot(columns='Op',index='Framework').fillna(0)
        df = df.reindex(ops_order, axis=1, level='Op')
        print(df)

        def truncate_colormap(cmap, minval=0, maxval=0.85, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap

        cmap = truncate_colormap(cm.get_cmap('plasma', 20))
        add_legend = (dataset == datasets[0])
        g = df.plot.barh(stacked=True, legend=add_legend, colormap=cmap, figsize=(5, 0.7), width=0.7)

        handles, labels = g.get_legend_handles_labels()
        labels = [l.split(', ')[1][0:-1] for l in labels]
        labels[1] = r'$\mathrm{QK}^{\mathrm{T}}$'
        if add_legend:
            g.legend(handles=handles, labels=labels, ncol=3, loc='upper center', bbox_to_anchor=(0.4, 1.77),
                     labelspacing=0.3, borderpad=0.1)
        g.set_xlabel('Execution Time (ms)')
        g.set_ylabel('')
        g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

def make_nsf_cuda_plots(csvdir, figdir, args):
    figpath = figdir + '/nsf_cuda.pdf'
    res_df = pd.read_csv(csvdir + '/bert_layer_results_cuda.csv')
    prelude_df = pd.read_csv(csvdir + '/bert_layer_prelude_results_cuda.csv')
    batch_size = 64

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    pytorch_idx = 3
    ft_idx = 4
    ftn_idx = 5
    cora_idx = 6
    prelude_cora_idx = 3

    res_df = res_df.loc[res_df[res_df.columns[b_size_idx]] == batch_size]
    prelude_df = prelude_df.loc[prelude_df[prelude_df.columns[b_size_idx]] == batch_size]

    res_df[res_df.columns[cora_idx]] = res_df[res_df.columns[cora_idx]] - prelude_df[prelude_df.columns[prelude_cora_idx]]
    df = res_df
    cols = df.columns

    pytorch_relative = 'PyTorch'
    ftn_relative = 'FasterTransformer'
    cora_relative = 'CoRa'

    df[pytorch_relative] = df[cols[pytorch_idx]] / df[cols[cora_idx]]
    df[ftn_relative] = df[cols[ftn_idx]] / df[cols[cora_idx]]
    df[cora_relative] = df[cols[cora_idx]] / df[cols[cora_idx]]

    df = df.drop(columns = [cols[target_idx], cols[ft_idx], cols[pytorch_idx], cols[ftn_idx], cols[cora_idx], cols[b_size_idx]])

    framework = 'Framework'
    inf_lat = 'Relative''\n''Execution Time'

    df = df.melt(id_vars=[cols[dataset_idx]], var_name = framework, value_name = inf_lat)
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    plt.figure(figsize = (5, 1.2))
    g = sns.barplot(data=df, x=cols[dataset_idx], y=inf_lat, hue=framework, ci=None,
        palette = palettes[3])

    plt.xticks(rotation=5)
    plt.grid(axis='y')
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, ncol=3, loc='upper center', bbox_to_anchor=(0.43, 1.35))
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)

    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)


def make_nsf_arm_plots(csvdir, figdir, args):
    figpath = figdir + '/nsf_arm.pdf'
    df = pd.read_csv(csvdir + '/bert_layer_results_cpu.csv')
    batch_size = 64

    target_idx = 0
    dataset_idx = 1
    b_size_idx = 2
    pytorch_idx = 3
    tf_idx = 4
    cora_idx = 5

    cols = df.columns
    df = df.loc[df[cols[b_size_idx]] == batch_size]

    pytorch_relative = 'PyTorch'
    tf_relative = 'TensorFlow'
    cora_relative = 'CoRa'

    df[pytorch_relative] = df[cols[pytorch_idx]] / df[cols[cora_idx]]
    df[tf_relative] = df[cols[tf_idx]] / df[cols[cora_idx]]
    df[cora_relative] = df[cols[cora_idx]] / df[cols[cora_idx]]

    df = df.drop(columns = [cols[target_idx], cols[pytorch_idx], cols[tf_idx], cols[cora_idx], cols[b_size_idx]])

    framework = 'Framework'
    inf_lat = 'Relative''\n''Execution Time'

    df = df.melt(id_vars=[cols[dataset_idx]], var_name = framework, value_name = inf_lat)
    df[cols[dataset_idx]] = df[cols[dataset_idx]].apply(common.get_dataset_repl)

    plt.figure(figsize = (5, 1.2))
    g = sns.barplot(data=df, x=cols[dataset_idx], y=inf_lat, hue=framework, ci=None,
        palette = palettes[3])

    plt.xticks(rotation=5)
    plt.grid(axis='y')
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=labels, ncol=3, loc='upper center', bbox_to_anchor=(0.43, 1.35))
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)

    print(df)

    g.figure.savefig(figpath, bbox_inches='tight', pad_inches = 0)

sns.set_style("ticks", {'grid.linestyle': '-'})
sns.set_context("paper")

def main(args):
    for i in range(1, 11):
        palettes[i] = [cm.inferno(1 - x) for x in np.linspace(1.0, 1.0 - (i - 1) * (0.85/i), i)]

    csvdir = args.csv_dir
    figdir = args.fig_dir

    make_intro_flops_plot(csvdir, figdir, args)
    make_partial_pad_overheads_plot(csvdir, figdir, args)
    make_vbatch_gemm_plot(csvdir, figdir, args)
    make_trmm_plot(csvdir, figdir, args)
    make_pad_fusion_plot(csvdir, figdir, args)
    make_memory_plot(csvdir, figdir, args)
    make_mmha_plot(csvdir, figdir, args)
    make_binpack_plot(csvdir, figdir, args)
    make_qkt_split12_binpack_plot(csvdir, figdir, args)
    make_bert_layer_summary_plot(csvdir, figdir, args)
    make_per_ops_plot(csvdir, figdir, args)
    make_ragged_overheads_plot(csvdir, figdir, args)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--csv-dir', dest='csv_dir', default=False, action='store_true')
    parser.add_argument('--fig-dir', dest='fig_dir', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
