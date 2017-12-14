#!/bin/sh

FILE=$(mktemp)

function add() {
    for SEED in `seq 0 $(($2 - 1))`
        do
            OUTFILE="results/$1/$SEED.npy"
            echo "if [ ! -f $OUTFILE ]; then ./src/$1.py $OUTFILE --seed $SEED; fi" >> $FILE
        done
}

add "random_exploration" 1000
add "policy_iteration" 1000
add "policy_iteration_for_optimal_policy" 100
add "k_distance_based_exploration" 500
add "transition_k_distance_based_exploration" 500
add "lof_based_exploration" 300
add "transition_lof_based_exploration" 300

parallel :::: $FILE
rm $FILE
./src/plot.py
