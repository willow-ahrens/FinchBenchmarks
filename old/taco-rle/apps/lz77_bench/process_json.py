#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


plt.rcParams['figure.dpi'] = 600

path_res = '/Users/danieldonenfeld/Developer/taco/apps/lz77_bench/out.json'
path_out = '/Users/danieldonenfeld/Developer/taco/apps/lz77_bench/out_plots/'

def split_name_str(s):
    res = s.split(':')
    return (res[0], res[1])

def parse_name(name):
    name = name.split('/')
    size = int(name[1])
    thresh = int(name[2])
    isDense = bool(int(name[3]))
    kind = name[-1]
    return (name, size, thresh, isDense, kind)

def dict_name(name):
    name_list = name.split('/')
    del name_list[1]
    return "/".join(name_list)


def process_benchmark(benchmark_json_dict, d):
    name = benchmark_json_dict['name']
    name, size, thresh, isDense, kind = parse_name(name)
    if kind == 'process_time_mean':
        xs, ys = d.setdefault("{}_{}".format(thresh,isDense), ([], []))
        xs.append(size)
        ys.append(benchmark_json_dict['cpu_time'])



def process_all():
    with open(path_res) as f:
        data = json.load(f)

    d = dict()

    for b in data['benchmarks']:
        process_benchmark(b, d)
    return d

def append_data(d, data, thresh, isDense):
    xs, ys = d["{}_{}".format(thresh, isDense)]
    if not 'x' in data.keys():
        data['x'] = xs
    data['{}_{}'.format(thresh, isDense)] = map(math.log, ys)


def create_bar_charts():
    d = process_all()
    data = {}
    append_data(d, data, 0, True)
    for thresh_v in range(0,11,5):
        append_data(d, data, thresh_v, False)

    df = pd.DataFrame(data)
                
    # plot grouped bar chart
    df.plot(x='x',
            kind='bar',
            stacked=False,
            title="Threshold {}".format(thresh_v))

    plt.tight_layout()
    plt.savefig("{}thresh_runs.png".format(path_out))
    plt.close()


if __name__ == "__main__":
    create_bar_charts()