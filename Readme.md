<h1>[TOSEM 2025] FairGenerate: Enhancing Fairness Through Synthetic Data Generation and Two-Fold Biased Labels Removal </h1> 

Welcome to the homepage of our work, “FairGenerate: Enhancing Fairness Through Synthetic Data Generation and Two-Fold Biased Label Removal”, published at TOSEM 2025. This work introduces a novel preprocessing method, 'FairGenerate', designed to address imbalanced data and biased labels in training datasets.

********************************************************************************************************
<h2> Datasets</h2>

In this study, we utilized nine publicly available datasets. All datasets are provided in the Dataset folder, except Meps15 and Meps16, which are excluded due to size limitations. These two datasets can be accessed through the provided URL.

1. Adult Income dataset - http://archive.ics.uci.edu/ml/datasets/Adult
2. COMPAS - https://github.com/propublica/compas-analysis
3. German Credit - https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
4. Bank Marketing - https://archive.ics.uci.edu/ml/datasets/bank+marketing
5. Default Credit - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
6. Heart - https://archive.ics.uci.edu/ml/datasets/Heart+Disease
7. MEPS15 - https://meps.ahrq.gov/mepsweb/
8. MEPS16 - https://meps.ahrq.gov/mepsweb/
9. Student - https://archive.ics.uci.edu/ml/datasets/Student+Performance

**MEPS15** - https://gitlab.liris.cnrs.fr/otouat/MEPS-HC/-/blob/main/h181.csv <br />
**MEPS16** - https://gitlab.liris.cnrs.fr/otouat/MEPS-HC/-/blob/main/h192.csv
********************************************************************************************************

Baselines
-----------------------------------------------------
**Fair-Smote: Proposed in the paper: Bias in Machine Learning Software: Why? How? What to Do?**
Fair-Smote is a pre-processing method that uses the modified SMOTE method to make the distribution of sensitive features in the data set consistent, and then deletes biased data through situation testing.
We use the code they provided in the code repository: https://github.com/joymallyac/Fair-SMOTE

**FairMask: Proposed in the paper: FairMask: Better Fairness via Model-Based Rebalancing of Protected Attributes**
Fairway is a hybrid algorithm that combines pre-processing and post-processing methods. 
We use the code they provided in the code repository: https://github.com/anonymous12138/biasmitigation 

**LTDD: Linear Regression Based Training Data Debugging**
LTDD is preprocessing algorithm that finds and removes the biased portion present in the features of the training data.
We use the code they provided in the code repository: https://github.com/fairnesstest/LTDD

**MirrorFair**
MirrorFair is a preprocessing method that employs an ensemble approach to address fairness issues, grounded in the principles of
counterfactual inference. It creates a counterfactual dataset from the original dataset and trains two
separate models, one on the original dataset and the other on the counterfactual dataset. Finally, it
adaptively combines the predictions from both models to produce fairer final decisions.
We use the code they provided in the code repository: https://github.com/XY-Showing/FSE2024-MirrorFair
 
**Reweighing: Data preprocessing techniques for classification without discrimination**
Reweighing is a pre-processing method that calculates a weight value for each data point based on the expected probability and the observed probability, to help the unprivileged class have a greater chance of obtaining favorable prediction results. 
We use the following python's AIF360 module to achieve it:

<code>from aif360.algorithms.preprocessing import Reweighing</code>
-----------------------------------------------------

- **Baselines Results at MirrorFair Settings**
- We have also run the baselines, including FairGenerate, on the MirrorFair Settings. Its results are available below.
- 
- https://drive.google.com/drive/folders/177g0Z6-TRBTxQSCePclyT3TvsJH2UMCL?usp=sharing
- -----------------------------------------------------
- 
- <!>**Code and Generated Data** -  https://drive.google.com/drive/folders/1AUKoHZ2sPWzTogNX4YV9D9hJPymHWIQV?usp=sharing
- 
- **Results** - https://drive.google.com/drive/folders/1X3RdUNN07Vcum1Sh7HpUKCMfTrQ_HHC6 
- 
- The codes in the folder are named for the applicable scenarios. The Adult and COMPAS data sets include two protected attributes, so we divide them into two scenarios: Adult_sex and Adult_race, similarly COMPAS_sex and COMPAS_race.
- 
- <h1> Code description</h1>
- 
- In the above replicate package -
- * <b>generated_data</b> folder contains the final synthetic data samples produced by the proposed approach, which were used to report the results.
- * <b>FairGenerate.py </b> contains the code of the proposed approach to generate synthetic data samples.
- * For each scenario, a Jupyter notebook is provided, allowing for the replication of results. Directly running them will result in the replication of results
- * <b>Stats.py</b> file contains the code of Scott-Knott results.
- 
- ********************************************************************************************************
- 
- 
- The following folder contains an Excel sheet that contains the Scott-Knott Test and Fairea evaluation for Logistic Regression, Decision Tree, Support Vector Machine, and Deep Learning Models Used in this study. 

https://drive.google.com/drive/folders/1X3RdUNN07Vcum1Sh7HpUKCMfTrQ_HHC6


## 📄 Citation

If you use this work, please cite our TOSEM 2025 paper:

```bibtex
@article{fairgenerate2025,
  title     = {FairGenerate: Enhancing Fairness Through Synthetic Data Generation and Two-Fold Biased Label Removal},
  author    = {Your Name and Coauthor Name and Another Coauthor},
  journal   = {ACM Transactions on Software Engineering and Methodology (TOSEM)},
  year      = {2025},
  volume    = {XX},
  number    = {XX},
  pages     = {XX--XX},
  doi       = {10.1145/XXXXXXX}
}
