# Machine Learning Cancer Diagnosis (Random Forest)
This project creates a Random Forest machine learning model to determine which genes best predict whether a patient will develop two cancer symptoms: 

    1. Perineural Invasion (PNI)      - Invasion of cancer to the space surrounding a nerve
    2. Lymphovascular Invasion (LVI)  - Invasion of cancer to lymphatics or blood vessels

The user will input two datasets 
(row: patient, col: normalized TPM counts per gene, 1st col: whether patient has PNI/LVI or not):

    1. 80% Training Dataset           - To train the Random Forest model and determine the optimal parameters
    2. 20% Testing Dataset            - To test the Random Forest and evaluate the ROC AUC (Reciever Operating Characteristic Area Under Curve) metric

This project will output four files:

    1. Probability Matrix (Training)  - According to Random Forest, % likelihood that a given patient (in the training dataset) has PNI/LVI
    2. Probability Matrix (Testing)   - According to Random Forest, % likelihood that a given patient (in the testing dataset) has PNI/LVI
    3. Feature Importance             - Ranking of genes, according to its ability to predict PNI/LVI
    4. ROC Curve                      - X-axis: False positive rate, Y-axis: True positive rate

## Best Practices

1. Accuracy significantly improves when DESeq2 differential gene expression analysis is performed on the 80% training dataset before running this project. After running DESeq2, user should remove all genes whose P-adjusted values are lower than a threshold (e.g. 0.001, 0.005, 0.1, 0.5). This will ensure that only optimal genes are used for the Random Forest model. An example DESeq2 R script can be found in my GitHub.
2. Use tune_parameters() to determine the optimal parameters for the Random Forest. Make sure to update parameters in the random_forest() function after running the tuning function.
3. Use find_feature_importance_threshold() to remove the least significant genes, one-by-one, until number of genes = 0. This function will return the best feature importance threshold / set of genes to train the highest accuracy Random Forest model.

## Results (ROC AUC)

    PNI:
        P-adjusted Value < 0.001:   0.7166400196777765
        P-adjusted Value < 0.005:   0.7285696716271061
        P-adjusted Value < 0.01:    0.7095068257286926
        P-adjusted Value < 0.05:    0.7281392202681097
    LVI:
        P-adjusted Value < 0.001:   0.6113251155624037
        P-adjusted Value < 0.005:   0.6116782229070366
        P-adjusted Value < 0.01:    0.6269581407293271
        P-adjusted Value < 0.05:    0.621276322547509

## Best Parameters (Using RandomizedSearchCV)
    PNI:
      P-adjusted Value < 0.001
        eta = 0.7
        max_depth= 3
        subsample = 1
        colsample_bytree = 0.6
        min_chil_weight=1
      P-adjusted Value < 0.005
          eta = 0.1
          max_depth= 4
          subsample = 0.6
          colsample_bytree = 0.3
          min_chil_weight=1
      P-adjusted Value < 0.01
          eta = 0.4
          max_depth = 5
          subsample = 0.6
          colsample_bytree = 0.3
          min_chil_weight=1
      P-adjusted Value < 0.05
          eta = 0.1
          max_depth = 3
          subsample = 0.8
          colsample_bytree = 0.6
          min_chil_weight= 1
    
    LVI:
      P-adjusted Value < 0.001
          eta = 0.5
          max_depth = 9
          subsample = 0.5
          colsample_bytree = 0.6
          min_chil_weight= 1
      P-adjusted Value < 0.005
          eta = 0.1
          max_depth = 9
          subsample = 0.5
          colsample_bytree = 0.6
          min_chil_weight= 1
      P-adjusted Value < 0.01
          eta = 0.1
          max_depth = 9
          subsample = 0.5
          colsample_bytree = 0.6
          min_chil_weight= 1
      P-adjusted Value < 0.05
          eta = 0.4
          max_depth = 6
          subsample = 1
          colsample_bytree = 1
          min_chil_weight= 1

## Author

* **Peter Hwang** - [pghwang](https://github.com/pghwang)

## Acknowledgments

* This project has been created for an undergraduate research project under the Broad Institute of MIT and Harvard.
* Special thanks to Jimmy Guo and Hannah Hoffman for the guidance and support!

## References

    Random Forest Feature Importance: https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
    Random Forest Parameter Tuning:   https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    Random Forest ROC Curve Plotting: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
