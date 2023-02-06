## Dataset

In this work, we used the publically available dataset. We uploaded the datasets in the **Dataset**  folder except for the below, due to size limit constraints. They can be obtained from the below URLs.

Home Credit - https://www.kaggle.com/c/home-credit-default-risk <br />
MEPS15 - https://gitlab.liris.cnrs.fr/otouat/MEPS-HC/-/blob/main/h181.csv <br />
MEPS16 - https://gitlab.liris.cnrs.fr/otouat/MEPS-HC/-/blob/main/h192.csv

********************************************************************************************************

1. Adult Income dataset - http://archive.ics.uci.edu/ml/datasets/Adult
2. COMPAS - https://github.com/propublica/compas-analysis
3. German Credit - https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
4. Bank Marketing - https://archive.ics.uci.edu/ml/datasets/bank+marketing
5. Default Credit - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
6. Heart - https://archive.ics.uci.edu/ml/datasets/Heart+Disease
7. MEPS - https://meps.ahrq.gov/mepsweb/
8. Student - https://archive.ics.uci.edu/ml/datasets/Student+Performance
9. Home Credit - https://www.kaggle.com/c/home-credit-default-risk

## Baseline

FairMASK  - Proposed in the paper: 
FairMask: Better Fairness via Model-based Rebalancing of Protected Attributes is a pre-processing and post-processing method that uses the extrapolation method to replace protected attributes from the testing data. 

We use the code they provided in the code repository: https://github.com/anonymous12138/biasmitigation


 Fair-Smote: Proposed in the paper: Bias in Machine Learning Software: Why? How? What to Do? Fair-Smote is a pre-processing method that uses the modified SMOTE method to make the distribution of sensitive features in the data set consistent, and then deletes biased data through situation testing.

We use the code they provided in the code repository: https://github.com/joymallyac/Fair-SMOTE
