#!/usr/bin/env python -O

import argparse
import numpy as np
import scipy.stats as st

parser = argparse.ArgumentParser()
parser.add_argument('episodes', type=int)
parser.add_argument('seeds', type=int)
args = vars(parser.parse_args())

all_data = np.zeros(
    (args['episodes'], args['seeds']), dtype=np.float64) * np.nan

for i in range(args['seeds']):
    with open('{}.csv'.format(i), 'r') as infile:
        raw_data = infile.read()
    if len(raw_data.strip()) == 0:
        continue
    data = np.array([float(entry) for entry in raw_data.split()])
    np.copyto(all_data[:, i], data)

mean = np.nanmean(all_data, axis=1)
sem = st.sem(all_data, axis=1, nan_policy='omit')

print('mean,sem')
for i in range(args['episodes']):
    print('{0:0.4f},{1:0.4f}'.format(mean[i], sem[i]))
