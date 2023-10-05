#!/usr/bin/env python

import json
import argparse
import os
import csv

def split_name_str(s):
    res = s.split(':')
    return (res[0], res[1])

def parse_name(name):
    name = name.split('/')
    size = int(name[1])
    thresh = int(name[2])
    runl = int(name[3])
    isDense = bool(int(name[4]))
    return (size, thresh, runl, isDense)

def copy_to_csv(json_file, csv_file):
    with open(json_file) as f:
        data = json.load(f)

    with open(csv_file, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['size', 'threshold', 'run_length', 'is_dense', 'unit', 'Time', 'CPU', 'Iterations', 'd0_bytes', 'd0_raw_vals', 'd1_bytes', 'd1_raw_vals'])
        for b in data['benchmarks']:
            size, thresh, runl, isDense = parse_name(b['name'])
            writer.writerow([size, thresh, runl, isDense, b['time_unit'],\
                b['real_time'], b['cpu_time'], b['iterations'],  int(b['d0_bytes']), int(b['d0_raw_vals']), int(b['d1_bytes']), int(b['d1_raw_vals'])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert google benchmark json to CSV.')
    parser.add_argument('--json', metavar='j', dest='json_file', default="",
                        help='The location of the json file to process')
    parser.add_argument('--csv', metavar='o', dest='csv_file', default="",
                        help='The output location')
    args = parser.parse_args()

    if args.csv_file == "":
        args.csv_file = os.path.dirname(args.json_file) + "/" + os.path.splitext(os.path.basename(args.json_file))[0] + ".csv"

    copy_to_csv(args.json_file, args.csv_file)