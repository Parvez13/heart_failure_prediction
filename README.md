# HEAR FAILURE PREDICTION

This is a project, replicating a paper from **BMC Medical Informatics and Decision Making**.

## Problem Statement
- Machine Learning  can predict survival of patients with Heart Failure from Serum Creatinine and Ejection Fraction Alone.

### Keywords
* Cardiovascular heart diseases 
* Heart failure
* Serum creatinine
* Ejection fraction
* Medical records 
* Feature ranking
* Feature selection 
* Biostatistics
* Machine learning 
* Data mining 
* Biomedical informatics

### Background:
Cardiovascular diseases kill approximately 17 million people globally every year, and they mainly
exhibit as myocardial infarctions and heart failures. Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body.
Available electronic medical records of patients quantify symptoms, body features, and clinical laboratory test values, which can be used to perform biostatistics analysis aimed at highlighting patterns and correlations otherwise undetectable by medical doctors. 
Machine learning, in particular, can predict patients’ survival from their data and can
individuate the most important features among those included in their medical records

### Mehtods:
The methods we gonna use

1. Machine Learning methods we used for the binary classification **(Survival prediction Classifier)**.
2. Biostatistics and machine learning methods we employed for the feature ranking, discarding each patients’ 
3. Survival machine learning prediction on serum creatinine and ejection fraction alone.

## 1. Survival Prediction Classifier

This part of our analysis focuses on the binary prediction of the survival of the patients in the follow-up period.

To predict patients survival, we employed ten different methods from different machine learning areas.
The classifiers include

* One linear statistical method(Linear Regression).
* Three tree-based methods (Random Forests, One Rule, Decision Tree).
* One Artificial Neural Network (perceptron).
* Two Support Vector Machines (Linear and Gaussian radial kernel)
* One instance-based learning model (K-Nearest Neighbors).
* One probabilistic classifier (Naive Bayes).
* An ensemble boosting method (Gradient Boosting).

We measured the prediction results through
* **Matthews correlation coefficient(MCC)** : The MCC takes into account the dataset is imbalance and generates a high score only if the predictor performed well both on the majority of negative data instances and on the majority of positive data instances . Therefore, we give more importance to the MCC than to the other confusion matrix metrics, and rank the results based on the MCC
* **Receiver operating characteristic(ROC) area under curve**.
* **Precision-recall (PR) area under curve**.

### Survival Prediction Classifier Results
| Models      | Matthews correlation coefficient (MCC)  | F1-Score | Accuracy | TPR(True Positive Rate| TNR(True Negative Rate) | PR Auc | ROC Curve|
| ------------- | ---------- | ----------- | -------------| -----------------| -------------| -----------| ------------| 
|  DecisionTreeClassifier  |  0.351657 	| 0.529412 	| 0.733333 |	0.663671 |	0.663671 |	0.620175 |	0.663671   |   
| RandomForestClassifier   |   0.599886 |	0.705882 |	0.833333 |	0.779204| 	0.779204 |	0.774123 |	0.891528   |     
| XGBoost  |  0.598633 	| 0.687500 |	0.833333 |	0.765083 |	0.765083 |	0.779217 |	0.833119 |  
| Logistic Regression |  0.598633 |	0.687500 |	0.833333 |	0.765083 |	0.765083 |	0.779217 |	0.856226 |  
| SVM(Linear Kernel) |  0.418221 |	0.533333 |	0.766667 |	0.673941 |	0.673941 |	0.665829 |	0.831836 |  
| SVM(Gaussian Kernel) |  0.000000 	|0.000000 |	0.683333 |	0.500000 |	0.500000 |	0.658333 |	0.471117 |  
| Naive Bayes |  0.368531 |	0.482759 |	0.750000 |	0.647625 |	0.647625| 	0.634211 |	0.856226 |  
| KNearestNeighbors |  0.080115 |	0.275862 |	0.650000 |	0.532092| 	0.532092 |	0.430263 |	0.485879 |  
| ArtificialNeuralNetwork | 0.399702 |	0.571429 |	0.750000 |	0.689987| 	0.689987 |	0.650658 |	0.689987  | 
| OneRClassifier | 0.105332 |	0.173913 |	0.683333 |	0.528241 |	0.528241 |	0.444298|	0.528241  | 

## 2. Feature Ranking

For the feature ranking, two traditional approaches are used in the paper

   1.Biostatistics.
   
   2. Machine Learning.

   1.In Biostatistics approach three different approachs are used

   * *Mann-Whitney U test.*
   * *Pearson correlation coefficient.*
   * *Chi Square test.*
   * To compare the distribution of each feature between the two groups (survived individuals and dead patients), plus the *Shapiro–Wilk* test to check the distribution of each feature.

   1.**Machine Learning**

   In this approach, only Random Forest is used.

**Feature Ranking With Random Forest**

Feature importance is another way of asking, "Which features contributed most to the outcomes of the model and how did they contribute?"

Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".

Let's find the feature importance for our RandomForest model.

| Feature | Score |
| ------- | ----- |
| age| 0.07441141887635033|
| anaemia| 0.013753042993957803|
| creatinine_phosphokinase|0.07711649390950762|
| diabetes| 0.01309907993041349|
| ejection_fraction| 0.13125294674109833|
| high_blood_pressure| 0.013732259553809729|
| platelets| 0.07798931742071989|
| serum_creatinine|0.14908356500095482|
| serum_sodium| 0.06383486867800002|
| sex| 0.012093316542865712|
| smoking| 0.013558390604857338|
| time|0.3600752997474648|
