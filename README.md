
## A. Problem Statement

The aim of this assignment is to build and compare different machine learning classification models on the same dataset.

For this project, I have used the **Spambase dataset** from the UCI Machine Learning Repository. The task is to classify an email as either spam or non-spam based on the features available in the dataset.

I implemented the following classification models:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

The models were compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

The trained models are also used in a Streamlit application where test data can be uploaded and the performance of the selected model can be viewed.

---

## B. Dataset Description

**Dataset:** Spambase
**Source:** UCI Machine Learning Repository
**Problem Type:** Binary Classification

The dataset contains **4,601 records and 57 input features**.

The target variable has two classes:

* 0 - Non-Spam
* 1 - Spam

The features are numerical and mainly represent different characteristics of emails such as word frequency and character frequency.

There were no missing values in the dataset.

I used an **80:20 train-test split** for model training and evaluation.

* Training records: 3,680
* Testing records: 921

Stratified sampling was used while splitting the dataset so that the class distribution remained similar in both training and testing data.

The test portion of the dataset was saved as `test_data.csv` and is also used for testing the models through the Streamlit application.

---

## C. GitHub Repository Link

GitHub Repository:

https://github.com/Lalitv7/ML_Assignment_2


## D. Models Used

Five classification algorithms were trained using the same training and testing data.

For Logistic Regression and kNN, feature scaling was done using `StandardScaler` because these algorithms can be affected by differences in feature scale.

The models used were:

* Logistic Regression
* Decision Tree Classifier
* k-Nearest Neighbors
* Gaussian Naive Bayes
* Random Forest Classifier

---

## Model Performance Comparison

The following results were obtained on the test dataset.

| ML Model            | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.9294 | 0.9702 |    0.9209 | 0.8981 |   0.9093 | 0.8518 |
| Decision Tree       |   0.9110 | 0.9078 |    0.8828 | 0.8926 |   0.8877 | 0.8140 |
| kNN                 |   0.9077 | 0.9506 |    0.8861 | 0.8788 |   0.8824 | 0.8065 |
| Naive Bayes         |   0.8339 | 0.9449 |    0.7178 | 0.9532 |   0.8189 | 0.6941 |
| Random Forest       |   0.9457 | 0.9833 |    0.9510 | 0.9091 |   0.9296 | 0.8860 |

---

## Observations

### Logistic Regression

Logistic Regression gave good results on this dataset with an accuracy of **92.94%**. Its AUC was **0.9702**, which shows that the model was able to separate spam and non-spam emails quite well.

The precision and recall were also close to each other, resulting in an F1 score of **0.9093**. Overall, the model gave balanced performance.

### Decision Tree

Decision Tree achieved an accuracy of **91.10%** and an F1 score of **0.8877**.

The performance was good, but it was lower than Logistic Regression and Random Forest. Its AUC value of **0.9078** was also the lowest among the five models.

### k-Nearest Neighbors

kNN achieved an accuracy of **90.77%** with an AUC of **0.9506**.

Its precision and recall were **0.8861** and **0.8788** respectively. The results were reasonably good, although it did not perform as well as Logistic Regression or Random Forest.

### Naive Bayes

Naive Bayes behaved differently from the other models.

It achieved the **highest recall of 0.9532**, which means that it detected a large proportion of the actual spam emails.

However, its precision was only **0.7178**. This means that it also marked more non-spam emails as spam compared with the other models.

Because of this, its overall accuracy was **83.39%**, which was the lowest among the models tested.

### Random Forest

Random Forest gave the best overall results.

It achieved:

* Accuracy: **94.57%**
* AUC: **0.9833**
* Precision: **95.10%**
* Recall: **90.91%**
* F1 Score: **92.96%**
* MCC: **0.8860**

It obtained the highest Accuracy, AUC, Precision, F1 Score and MCC among the five models.

---

## Overall Winner

For this dataset, I selected **Random Forest** as the best-performing model.

Random Forest gave an accuracy of **94.57%** and the highest F1 score of **0.9296**. It also achieved the highest AUC and MCC values.

Naive Bayes had a better recall, but its precision and overall accuracy were much lower.

Therefore, considering all the evaluation metrics together, Random Forest gave the most balanced performance on the Spambase dataset.

---

## Streamlit Application

A Streamlit application was developed to test the trained models.

The application provides the following options:

* Upload the test CSV file
* Select the required machine learning model
* View Accuracy, AUC, Precision, Recall, F1 Score and MCC
* View the confusion matrix
* View the classification report

The results for different models can therefore be checked using the same test dataset.

---

## Live Streamlit App

Streamlit App Link:

https://mlassignment2-ds8mhzjc9kyvwcq7y5mxvn.streamlit.app/

---

## Project Files

The main files used in the project are:

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## Conclusion

In this assignment, I implemented five different classification models on the Spambase dataset and compared them using six evaluation metrics.

Most of the models gave good classification results. Naive Bayes was particularly good at detecting spam because it had the highest recall, but it also produced more false positive predictions.

Random Forest gave the best overall results and achieved the highest accuracy, AUC, precision, F1 score and MCC.

Based on these results, **Random Forest was selected as the best model for this dataset**.
