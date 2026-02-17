# RecSys Challenge 2024 exercise

## Overview

This repository contains my experimental solution to the RecSys Challenge 2024 (EB-NeRD dataset).
The repository by Ekstra-Bladet is [here](https://github.com/ebanalyse/ebnerd-benchmark).
The goal is to predict user click behavior on news articles using the NRMSDocVec architecture, extended through ablation studies, preprocessing strategies, and user clustering.

Main contributions:
- Structured experimentation framework (generate → run → evaluate)
- Time-decay enhancement of NRMSDocVec
- User segmentation features improving performance
- PCA → GMM → BIC clustering analysis

## Tech Stack

- Python
- PyTorch
- Tensorflow
- Polars
- Scikit-learn
- NumPy

## Repository Structure

The repository contains:
- A numbered folder for each phase of the project
- A data folder containing the dataframes
- A new_data folder containing the new data extracted from the dataframes and the scripts implementing the queries to extract it.

## Used methodologies

### Generate → Run → Evaluate

When dealing with a large amount of experiments, it is not recommended to execute all of them at once in the same script.
If we did so, a single problem would delete all the results of the previously executed experiments.
Not to mention the fact that running a long series of experiments could take a large amount of time.
For this reason, it becomes necessary to have a permanent storage of the experiments execution status and results.

This is exactly the problem addressed using the generate → run → evaluate approach.
- The generate phase generates all the possible combination of hyperparameters and an experiment for each. It also generates a manifest containing the status of each experiment which could be "PENDING", "DONE" or "FAILED"
- The run phase consults the manifest and executes all the experiments labeled as "PENDING" in a separate process to avoid saturating the RAM
- The evaluation phase retrieves the results from the folders of the single experiments and evaluate the outcome of the experiments

I have also inserted an additional phase between the run phase and the evaluate phase to generate a static dataframe from
the data contained in the experiments folders. This phase is called "parquet generation" because this project is based
on Polars and its purpose is avoiding the evaluation script to retrieve static data from the experiments folders every
time it is executed.

### Ablation study

The ablation study is a process consisting of systematically modifying or removing components from a neural network in
order to determine their contribution to the end result.

### PCA → GMM → BIC

The pipeline consisting of Principal Components Analysis, Gaussian Mixture Model and Bayesian Information Criterion is a
standard unsupervised machine learning methodology to respectively:
1. Extract the most informative features from a dataset
2. Derive clusters from the selected features
3. Choose the optimal number of clusters as the configuration minimizing the number of clusters and maximizing the contained information.

## Phases of the exercise

The exercise was made in different phases:
1. Baseline test
2. Ablation study
3. Preprocessing study
4. Creation of the definitive model
5. Implementation of the model as a class with a fit-predict interface (similarly to the classes contained in Scikit-learn)
6. User clustering (as a bonus step)

### 1) Baseline test

This phase uses the NRMSDocVec model as done in the reproducibility script by the original authors.
The baseline study is done by changing the hyperparameters of the model and each experiment corresponds to a different
hyperparameter combination.

At the end of the baseline study we have the best combination of hyperparameters for the NRMSDocVec model which defines
the baseline for the following phases of the exercise.
We also have crucial findings about the metrics and the hyperparameters:
1. Metrics increase and decrease together
2. The relative order of the correlation between the hyperparameters and a metric is the same for all the metrics
3. We can identify hyperparameters combinations with a lower result a priori and so, reduce the granularity of the search

The best metrics obtained at the end of the baseline study are:
- AUC = 0.6928303358140224
- MRR = 0.46395148360881905
- NDCG@5 = 0.5172255459108421
- NDCG@10 = 0.5729085223844624

### 2) Ablation study

In the ablation study I have analyzed the code of NRMSDocvec and tried to change it adding the following components:
1. Addition of a positional embedding (as suggested by the authors of NRMS in the conclusion of the [paper](https://aclanthology.org/D19-1671.pdf))
2. Addition of residual connection and layer normalization between the two self-attention layers
3. Addition of a time decay on the history of the user to make the recent history of the user more relevant than the remote one

At the end of the ablation study we discover that the only modification that increases the performance of the model is
the time decay and, therefore, we add it permanently to the model to define a new baseline.

The reason why the other modifications reduce the performance of the model is the fact that they are not coherent with
the context of the project and the goal of the model. Our goal is to predict the recent future of the user. The order of
the user history items does not carry strong semantic meaning in this setting and the model never had exploding
gradients problems. For these reasons the other modifications introduce noise in the neural network.

The results of the ablation study are as follows:

| Metric | Baseline | Ablation study | Increment |
| ------ | -------- | -------------- | --------- |
| AUC | 0.6928303358140224 | 0.6981578666512015 | 0.8% |
| MRR | 0.46395148360881905 | 0.4687362174258258 | 1% |
| NDCG@5 | 0.5172255459108421 | 0.5237042689898062 | 1.2% |
| NDCG@10 | 0.5729085223844624 | 0.5770825547585428 | 0.7% |

The increments are small because this exercise was made using the demo dataset. For this reason, one has to be interested
in the methodology of the exercise rather than the results.

### 3) Preprocessing study

In the preprocessing study I create the following three user categories:
1. Categorization based on the user propensity to read premium articles (premium propensity)
2. Categorization based on the user propensity to read often and for long (reading engagement)
3. Categorization based on the user propensity to read long articles (article length propensity)

At the end of the preprocessing study we have that the combination of the premium propensity and the reading engagement
increases the performance of the model. This means that we can add them to the model permanently to further increase
the baseline.

The results of the preprocessing study are as follows:

| Metric | Ablation study     | Preprocessing study | Ablation to preprocessing increment | Overall increment (w.r.t the baseline) |
| ------ |--------------------|--------------------|-------------------------------------|----------------------------------------|
| AUC | 0.6981578666512015 | 0.7045915980921428 | 0.9%                                | 1.7%                                   |
| MRR | 0.4687362174258258 | 0.47527500522798993 | 1.4%                                | 2.4%                                   |
| NDCG@5 | 0.5237042689898062 | 0.5287402344152481 | 1%                                  | 2.2%                                   |
| NDCG@10 | 0.5770825547585428 | 0.5818183081184242 | 0.8%                                | 1.5%                                   |

### 4) Creation of the definitive model

In this phase I build the final model by including all the modifications which increased the baseline in the previous
phases. The folder of this phase also contains other scripts to make tests with the final model.

### 5) Implementation of the model as a class with a fit-predict interface

In this phase I transformed the model into a class with a fit-predict interface similarly to the ones contained in scikit-learn.
This makes the model easier to use using an external script especially to people who don't know the internal code of the
class.

### 6) User clustering

In this phase I use the PCA → GMM → BIC pipeline to derive clusters from the users. At the end of the clustering I calculate
the statistical indices of the clusters and the global dataframe to distinguish the descriptive features from the discriminating
features.

The descriptive features are the ones with statistical indices which are very similar to the ones of the global dataframe
and the discriminating features are the ones with statistical indices which are different from the ones of the global dataframe.
Descriptive features are called so because the features of the cluster **describe** the ones of the global dataframe.
Discriminating features are called this way because they are different from the ones of the global dataframe and, therefore,
they contain the reason why that cluster is different from the rest of the data.

To tell the discriminating features from the descriptive ones I use two values:
1. Relative difference $$\frac{|stat_{cluster} - stat_{global}|}{stat_{global}} > \tau$$
2. Statistical relevance $$\frac{|\mu_{cluster} - \mu_{global}|}{\sigma_{global}} > \gamma$$

## Installation

To use this project, after you download the repository and install the python requirements, you have to download some
complementary material. I am not sure if I can put that material in my repository for copyright reasons so you will have
to download it separately.
- [ebnerd-benchmark folder](https://github.com/ebanalyse/ebnerd-benchmark) (To be put in the main folder of the project)
- [ebnerd_demo & ebnerd_testset](https://docs.google.com/forms/d/e/1FAIpQLSdo6YZ1mVewLmqhsqqOjXTKsSp3OmCMHbMjEpsW0t_j-Hjtbg/viewform) (To be put in the data folder)
- Contrastive vector (From the same website of the previous two items, to be put in the 5<sup>th</sup> folder)

You are also encouraged to download all the files from Ekstra Bladet after filling the form to experiment with the dataset. 

## Citations of the original work

The authors of the original repository asked to be cited in derived works. Here are the desired citations to respect their will:

```bibtex
@inproceedings{kruse2024recsys_challenge,
  author    = {Kruse, Johannes and Lindskow, Kasper and Kalloori, Saikishore and Polignano, Marco and Pomo, Claudio and Srivastava, Abhishek and Uppal, Anshuk and Andersen, Michael Riis and Frellsen, Jes},
  title     = {RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations},
  booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems},
  series    = {RecSys '24},
  year      = {2024},
  pages     = {1195--1199},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3640457.3687164},
  url       = {https://doi.org/10.1145/3640457.3687164},
  keywords  = {Beyond-Accuracy, Competition, Dataset, Editorial Values, News Recommendations, Recommender Systems}
}
```

```bibtex
@inproceedings{kruse2024ebnerd,
  author    = {Kruse, Johannes and Lindskow, Kasper and Kalloori, Saikishore and Polignano, Marco and Pomo, Claudio and Srivastava, Abhishek and Uppal, Anshuk and Andersen, Michael Riis and Frellsen, Jes},
  title     = {EB-NeRD: A Large-scale Dataset for News Recommendation},
  booktitle = {Proceedings of the Recommender Systems Challenge 2024},
  series    = {RecSysChallenge '24},
  year      = {2024},
  pages     = {1--11},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3687151.3687152},
  url       = {https://doi.org/10.1145/3687151.3687152},
  keywords  = {Beyond-Accuracy, Dataset, Editorial Values, News Recommendations, Recommender Systems}
}
```

```bibtex
@article{kruse2025design_choices,
  author    = {Kruse, Johannes and Lindskow, Kasper and Andersen, Michael Riis and Frellsen, Jes},
  title     = {Why Design Choices Matter in Recommender Systems},
  journal   = {Nature Machine Intelligence},
  year      = {2025},
  volume    = {7},
  number    = {6},
  pages     = {979--980},
  doi       = {10.1038/s42256-025-01043-5},
  url       = {https://doi.org/10.1038/s42256-025-01043-5},
  publisher = {Nature},
  note      = {In press}
}
```