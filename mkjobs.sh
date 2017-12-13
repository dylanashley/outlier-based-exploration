#!/bin/sh

if [ -f 'jobs.sh' ]
    then
        echo 'ERROR: jobs.sh already exists'
        exit 1
    fi

NUM_SEEDS=300

for i in `seq 0 $((NUM_SEEDS - 1))`
    do
        echo "./src/random_exploration.py results/random_exploration/$i.npy --seed $i" >> jobs.sh
        echo "./src/policy_iteration.py results/policy_iteration/$i.npy --seed $i" >> jobs.sh
        echo "./src/k_distance_based_exploration.py results/k_distance_based_exploration/$i.npy --seed $i" >> jobs.sh
        echo "./src/transition_k_distance_based_exploration.py results/transition_k_distance_based_exploration/$i.npy --seed $i" >> jobs.sh
        echo "./src/lof_based_exploration.py results/lof_based_exploration/$i.npy --seed $i" >> jobs.sh
        echo "./src/transition_lof_based_exploration.py results/transition_lof_based_exploration/$i.npy --seed $i" >> jobs.sh
    done
