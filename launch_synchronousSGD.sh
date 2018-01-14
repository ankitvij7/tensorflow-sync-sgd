#!/bin/bash

export TF_LOG_DIR="/home/ubuntu/tf/logs"
source /home/ubuntu/assignment3/uploaded-scripts/tfdefs.sh

# startserver.py has the specifications for the cluster.
terminate_cluster
start_cluster startserver.py

echo "Executing the distributed tensorflow job from exampleSynchronousUpdate.py"

for i in 100
do
START=$(date +%s)
python synchronoussgd.py

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Program run time : $DIFF"
done
# defined in tfdefs.sh to terminate the cluster
tensorboard --logdir=$TF_LOG_DIR
terminate_cluster