# tensorflow-sync-sgd
A tensorflow python program running Logistic Regression Stochastic Gradient Descent Algorithm on the input dataset that is spread across 5 VMs in a synchronous manner

## Dataset
[Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) sponsored by Criteo Labs. It is available [here](http://pages.cs.wisc.edu/~akella/CS838/F16/assignment3/criteo-tfr-kaggle.tar.gz).

## Execution
* A local_gradient is computed on every device and then aggregated on the master VM.
* The local_gradient computation happens in parallel in all 5 VMs on every iteration. The aggregation is done on the master VM at the end of each iteration.
* Each iteration synchronously waits for the local_gradients to be computed on all 5 VMs before proceeding further.
* Cross-validation runs on the master VM at the end of every 100 iterations.

## Environment
5 node cluster where each node has 20GB RAM and 4 cores 
