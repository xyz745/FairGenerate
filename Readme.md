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

<h2> Baselines </h2>

**Fair-Smote: Proposed in the paper: Bias in Machine Learning Software: Why? How? What to Do?**
Fair-SMOTE is a pre-processing method that uses the modified SMOTE method to balance the distribution of sensitive features and class labels in the dataset consistently. Then, biased data labels are removed through situation testing tactics.
We use the code they provided in the code repository: https://github.com/joymallyac/Fair-SMOTE

**FairMask: Proposed in the paper: FairMask: Better Fairness via Model-Based Rebalancing of Protected Attributes**
Fairway is a hybrid algorithm that combines pre-processing and post-processing methods. 
We use the code they provided in the code repository: https://github.com/anonymous12138/biasmitigation 

**LTDD: Linear Regression Based Training Data Debugging**
LTDD is a preprocessing algorithm that finds and removes the biased portion present in the features of the training data.
We use the code they provided in the code repository: https://github.com/fairnesstest/LTDD

**MirrorFair**
MirrorFair is a preprocessing method that employs an ensemble approach to address fairness issues, grounded in the principles of counterfactual inference. It creates a counterfactual dataset from the original dataset and trains two separate models, one on the original dataset and the other on the counterfactual dataset. Finally, it adaptively combines the predictions from both models to produce fairer final decisions. We use the code they provided in the code repository: https://github.com/XY-Showing/FSE2024-MirrorFair
 
**Reweighing: Data preprocessing techniques for classification without discrimination**
Reweighing is a pre-processing method that calculates a weight value for each data point based on the expected probability and the observed probability, to help the unprivileged class have a greater chance of obtaining favorable prediction results. 
We use the following python's AIF360 module to achieve it:

<code>from aif360.algorithms.preprocessing import Reweighing</code>
-----------------------------------------------------

 <!-- Section heading for MirrorFair baseline results -->
**Baselines Results at MirrorFair Settings**

<!-- Description of what was done in this section -->
- We have also run the baselines, including FairGenerate, on the MirrorFair Settings. Its results are available below.

<!-- Empty line for spacing -->
- 

<!-- Link to MirrorFair baseline results -->
- https://drive.google.com/drive/folders/177g0Z6-TRBTxQSCePclyT3TvsJH2UMCL?usp=sharing

<!-- Separator line -->
- -----------------------------------------------------

<!-- Empty line for spacing -->
- 

<!-- Link to code and generated data, marked with a symbol to draw attention -->
- <!>**Code and Generated Data** -  https://drive.google.com/drive/folders/1AUKoHZ2sPWzTogNX4YV9D9hJPymHWIQV?usp=sharing

<!-- Link to result files used in the paper -->
- **Results** - https://drive.google.com/drive/folders/1X3RdUNN07Vcum1Sh7HpUKCMfTrQ_HHC6 

<!-- Explanation of how datasets with multiple protected attributes are split -->
- The codes in the folder are named for the applicable scenarios. The Adult and COMPAS data sets include two protected attributes, so we divide them into two scenarios: Adult_sex and Adult_race, similarly COMPAS_sex and COMPAS_race.

<!-- HTML header used in markdown to indicate a new section: code description -->
- <h1> Code description</h1>

<!-- Intro line to explain what’s in the replication package -->
- In the above replicate package -

<!-- List item: folder containing final generated synthetic datasets -->
- * <b>generated_data</b> folder contains the final synthetic data samples produced by the proposed approach, which were used to report the results.

<!-- List item: main implementation file for FairGenerate -->
- * <b>FairGenerate.py </b> contains the code of the proposed approach to generate synthetic data samples.

<!-- List item: Jupyter notebooks for each scenario to allow easy replication -->
- * For each scenario, a Jupyter notebook is provided, allowing for the replication of results. Directly running them will result in the replication of results

<!-- List item: script for statistical test used in evaluation -->
- * <b>Stats.py</b> file contains the code of Scott-Knott results.

<!-- Decorative separator line -->
- ********************************************************************************************************

<!-- Empty line for spacing -->
- 

<!-- Line explaining what the Excel file contains -->
- The following folder contains an Excel sheet that contains the Scott-Knott Test and Fairea evaluation for Logistic Regression, Decision Tree, Support Vector Machine, and Deep Learning Models Used in this study. 

<!-- Link to folder containing the Excel summary of evaluation results -->
<!-- https://drive.google.com/drive/folders/1X3RdUNN07Vcum1Sh7HpUKCMfTrQ_HHC6 -->

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
