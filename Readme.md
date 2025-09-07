<h1>[TOSEM 2025] FairGenerate: Enhancing Fairness Through Synthetic Data Generation and Two-Fold Biased Labels Removal </h1>  This work introduces a novel preprocessing method, 'FairGenerate', designed to address imbalanced data and biased labels in training datasets.

-----------------------------------------------------
<h2> Code </h2>

The file **FairGenerate.ipynb** contains the complete implementation of the FairGenerate method, including data preprocessing, synthetic data generation, and evaluation steps.

-----------------------------------------------------

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

Reweighing is a pre-processing method that calculates a weight value for each data point based on the expected probability and the observed probability, to help the unprivileged class have a greater chance of obtaining favourable prediction results. We use the following python's AIF360 module to achieve it: 

<code>from aif360.algorithms.preprocessing import Reweighing</code>

-----------------------------------------------------


## 📄 Citation

If you use this work, please cite our TOSEM 2025 paper:

```bibtex
@article{10.1145/3730579,
author = {Joshi, Hem Chandra and Kumar, Sandeep},
title = {FairGenerate: Enhancing Fairness Through Synthetic Data Generation and Two-Fold Biased Labels Removal},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1049-331X},
url = {https://doi.org/10.1145/3730579},
doi = {10.1145/3730579},
note = {Just Accepted},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = apr,
keywords = {ML software, Software fairness, Bias mitigation, Imbalanced Data, Biased Labels}
}
