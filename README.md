
<h1 align="center">❤HEART FAILURE PREDICTION❤</h1>

This is a project, replicating a paper from **BMC Medical Informatics and Decision Making**.

## 📝Problem Statement
- Machine Learning  can predict survival of patients with Heart Failure from Serum Creatinine and Ejection Fraction Alone.

### 🔑Keywords
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

### ✍ Background:
Cardiovascular diseases kill approximately 17 million people globally every year, and they mainly
exhibit as myocardial infarctions and heart failures. Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body.
Available electronic medical records of patients quantify symptoms, body features, and clinical laboratory test values, which can be used to perform biostatistics analysis aimed at highlighting patterns and correlations otherwise undetectable by medical doctors. 
Machine learning, in particular, can predict patients’ survival from their data and can
individuate the most important features among those included in their medical records

### ⚒ Mehtods:
The methods we gonna use

1. Machine Learning methods we used for the binary classification **(Survival prediction Classifier)**.✅
2. Biostatistics and machine learning methods we employed for the feature ranking, discarding each patients’ ✅
3. Survival machine learning prediction on serum creatinine and ejection fraction alone.✅

## ⛏ 1. Survival Prediction Classifier

This part of our analysis focuses on the binary prediction of the survival of the patients in the follow-up period.

To predict patients survival, we employed ten different methods from different machine learning areas.
The classifiers include

* One linear statistical method(Linear Regression).✅
* Three tree-based methods (Random Forests, One Rule, Decision Tree).✅
* One Artificial Neural Network (perceptron).✅
* Two Support Vector Machines (Linear and Gaussian radial kernel)✅
* One instance-based learning model (K-Nearest Neighbors).✅
* One probabilistic classifier (Naive Bayes).✅
* An ensemble boosting method (Gradient Boosting).✅

We measured the prediction results through
* **Matthews correlation coefficient(MCC)** : The MCC takes into account the dataset is imbalance and generates a high score only if the predictor performed well both on the majority of negative data instances and on the majority of positive data instances . Therefore, we give more importance to the MCC than to the other confusion matrix metrics, and rank the results based on the MCC✅
* **Receiver operating characteristic(ROC) area under curve**.✅
* **Precision-recall (PR) area under curve**.✅

### Survival Prediction Classifier Results📈
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

## ⛏2. Feature Ranking

For the feature ranking, two traditional approaches are used in the paper

   1.Biostatistics📚.
   
   2. Machine Learning📚.

   1.In Biostatistics approach three different approachs are used

   * **Mann-Whitney U test**✅
   * **Pearson correlation coefficient**✅
   * **Chi Square test**✅
   * To compare the distribution of each feature between the two groups (survived individuals and dead patients), plus the *Shapiro–Wilk* test to check the distribution of each feature.✅

   1.**Machine Learning**📚

   In this approach, only Random Forest is used.

**Feature Ranking With Random Forest**✅

Feature importance is another way of asking, "Which features contributed most to the outcomes of the model and how did they contribute?"

Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".

Let's find the feature importance for our RandomForest model.📈

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

  2. **Biostatistics** 📚
 
 **Pearson Correlation Coefficient**📈
 
 The Pearson correlation coefficient (or Pearson product-moment correlation coefficient, PCC) indicates the linear correlation between elements of two lists, showing the same elements on different positions. The absolute value of PCC generates a high value (close to 1) if the elements of the two lists have linear correlation, and a low value (close to 0) otherwise.

| Features| PCC|
| --------| ---|
|    Serum creatinine |	0.294278|
|Ejection fraction |	-0.268603|
|Age 	|0.253729|
|Serum sodium |	-0.195204|
|High blodd pressure| 	0.079351|
|Anaemia 	|0.066270|
|Creatinine phosphokinase |	0.062728|
|Plateles |	-0.049139|
|Smoking |	-0.012623|
|Sex 	|-0.004316|
|Diabetes |	-0.001943  |   

**Chi-Square Test**📈

The chi square test (or χ2 test) between two features checks how likely an observed distribution is due to chance. A low p-value (close to 0) means that the two features have a strong relation; a high p-value (close to 1) means, instead, that the null hypothesis of independence cannot be discarded.

| Feature | ChiSquare(p-value)|
| ------- | -----------|
|Serum creatinine |	9.988989e-01|
|Ejection fraction |	1.196437e-91|
|Age |	1.641847e-33|
|Serum sodium |	1.000000e+00|
|High blodd pressure |	9.999994e-01|
|Anaemia |	1.000000e+00|
|Creatinine phosphokinase 	|0.000000e+00|
|Plateles 	|0.000000e+00|
|Smoking |	9.999940e-01|
|Sex |	1.000000e+00|
|Diabetes |	1.000000e+00|

**Shapiro**📈

The Shapiro–Wilk test to check the distribution of each feature (to assess if was feature was extracted from a normal distribution).

| Feature | Shapiro(p-value)|
| --------| ------ |
|Serum creatinine |	5.392758e-27|
|Ejection fraction |	7.215433e-09|
|Age 	|5.350570e-05|
|Serum sodium| 	9.210248e-10|
|High blodd pressure |	1.168618e-25|
|Anaemia |	6.209964e-25|
|Creatinine phosphokinase |	7.050336e-28|
|Plateles |	2.883745e-12|
|Smoking |	4.581843e-26|
|Sex |	1.168500e-25|
|Diabetes 	|5.115524e-25|

**Mann-Whitney U-Test**📈

The Mann–Whitney U test (or Wilcoxon rank–sum test), applied to each feature in relation to the death event target, detects whether we can reject the null hypothesis that the distribution of the each feature for the groups of samples defined by death event are the same. A low p-value of this test (close to 0) means that the analyzed feature strongly relates to death event, while a high p-value (close to 1) means the opposite.

| Feature | P-value|
| ------ | ------ |
|Serum Creatinine |	0.000000|
|Ejection Fraction |	0.000000|
|Age |	0.000000|
|Serum Sodium |	0.000000|
|Creatinine_phosphokinase |	0.000000|
|Platelets |	0.000000|
|Sex |	0.000000|
|Anaemia |	0.005386|
|Diabetes 	|0.014107|
|High Blood Pressure |	0.436468|
|Smoking |	1.000000|


## ⛏ 3. Survival machine learning prediction on serum creatinine and ejection fraction alone

To investigate if machine learning can precisely predict patients survival by using the top two ranked features alone. They therefore elaborated another computational pipeline with an initial phase of feature ranking, followed by a binary classification phase based on the top two features selected.

All the different methods employed for feature ranking identified serum creatinine and ejection fraction as the top two features.So we then performed a survival prediction on these two features by employing three algorithms:

   * Random Forests✅
   * Gradient Boosting✅
   * SVM radial.✅

Model Results📈

| Models      | Matthews correlation coefficient (MCC)  | F1-Score | Accuracy | TPR(True Positive Rate| TNR(True Negative Rate) | PR Auc | ROC Curve|
| ------------- | ---------- | ----------- | -------------| -----------------| -------------| -----------| ------------| 
| RandomForestClassifier |	0.286630 |	0.451613 |	0.716667 |	0.623235 |	0.623235 |	0.575877 |	0.719512|
|XGBClassifier| 	0.302136 	|0.484848 |	0.716667 |	0.637356 |	0.637356 |	0.587907 |	0.729140|
|SVM with Gaussian Kernel |	0.199034 |	0.307692 |	0.700000 |	0.568678 |	0.568678 |	0.515977 |	0.651476|
