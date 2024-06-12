# Temporal Link Prediction in Social Networks Based on Agent Behavior Synchrony and a Cognitive Mechanism

This repository contains code and data samples accompanying the manuscript "Temporal Link Prediction in Social Networks Based on Agent Behavior Synchrony and a Cognitive Mechanism" [Duan et al](http://arxiv.org/abs/2406.06814).

## Repository content
This repository contains a [dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6Z3CGX) with metadata on email exchange between the employees of a mid-sized manufacturing company.

- **manufacturing.csv** - contains all email communication
- **manufacturing-train.csv** - training set
- **manufacturing-test.csv** - test set

Please refer to [1] if you use manufacturing company dataset.

## Code scripts
To get the results run `run.sh`.

`run.sh` uses following scripts:
1. `cogsnet-compute.c` - CogSNet model [2] implementation written in C
2. `link_prediction_models.py` - implementaion of link prediction methods written in Python

You might like to chnage the following parameters in `run.sh`:
- **FORGETTING**: linear OR exponential OR power
- **MU** $\in (0,1]$
- **THETA** $\in (0, \text{MU})$
- **LAMBDA** computed based on equations 6-8 from the manuscript and depending on the desired lifetime of edges.
- **UNITS** describes the time needed to elapse after the last CogSNet weight update for the next update to be possible. This mechanism helps avoid a very fast weight increase if interactions are very frequent. Usually set to 3600 (1 hour).
- **TIME_INTERVAL** set individually. This parameter describes how often CogSNet snapshots will be created to compute the cognitive vector. If UNITS is set to 1 hour, then TIME_INTERVAL has to be expressed in hours; if UNITS is set to 1 minute, then TIME_INTERVAL should be in minutes, etc.
- **ALPHA** $\in [0,1]$, controls the influence of neighborhood similarity and behavioral synchrony components.


## Results
Cognitive vectors:
- **cognset-*-avg.csv** contains computed cognitive vectors for avg agreagation
- **cognset-*-sum.csv** contains computed cognitive vectors for sum agreagation
- **cognset-*-weights.csv** contains computed final cogsnet weights after all events

Link prediction:
- **lp-*-auc.csv** - contains computed AUC for each method (_NSCV, CNSCV, NSCTV, CNSCTV, CNSTV, CNS_) and baseline (_CN, NSTV_).
- **lp-*-prec.csv** - contains computed precision for each ratio (from 1 to 0.1) for each method and baseline.

## Reference
[1] Michalski, R., Szymanski, B. K., Kazienko, P., Lebiere, C., Lizardo, O., & Kulisiewicz, M. (2021). Social networks through the prism of cognition. Complexity, 2021(1), 4963903.

[2] Nurek, M., & Michalski, R. (2020). Combining machine learning and social network analysis to reveal the organizational structures. Applied Sciences, 10(5), 1699.

## License
Both dataset samples and code are distributed under the General Public License v3 (GPLv3) license, a copy of which is included in this repository, in the LICENSE file. If you require a different license and for other questions, please contact us at mateusz.nurek@pwr.edu.pl
