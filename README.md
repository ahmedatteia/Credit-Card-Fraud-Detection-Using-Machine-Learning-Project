Credit Card Fraud Detection Using Machine Learning
This project focuses on detecting credit card fraud using machine learning techniques. Fraud detection in credit card transactions is a critical issue in financial services, and this project aims to address it by implementing machine learning algorithms to identify fraudulent transaction patterns and anomalies.
Project Overview
In this project, we explored various machine learning models to determine the most effective one for detecting fraudulent credit card transactions. With the increasing complexity of financial systems, the need for adaptive fraud detection models becomes more important to ensure the safety of transactions.
Machine Learning Models Used:
•	K-Nearest Neighbors (KNN)
•	Logistic Regression
•	Random Forest
•	Support Vector Machines (SVM)
•	Naïve Bayes
•	Neural Networks (CNN & RNN)
Dataset Overview
The dataset contains credit card transactions from European cardholders in September 2013. It includes 284,807 transactions, out of which 492 (0.172%) are labeled as fraudulent. The dataset is highly imbalanced, and the features are transformed using Principal Component Analysis (PCA).
Key Features:
•	Time: Seconds elapsed between each transaction and the first transaction in the dataset.
•	Amount: The transaction amount.
•	V1 to V28: Principal components obtained with PCA.
•	Class: Target variable where 1 indicates fraud, and 0 indicates a legitimate transaction.
Given the class imbalance, the performance of the models is evaluated using the Area Under the Precision-Recall Curve (AUPRC), as accuracy alone would be misleading.
Project Objectives
1.	Detect Fraudulent Transactions: Evaluate different machine learning models to identify fraudulent transactions from the given dataset.
2.	Compare Model Performance: Use metrics such as AUPRC, precision, recall, and accuracy to compare models.
3.	Address Data Imbalance: Handle the highly imbalanced dataset using SMOTE (Synthetic Minority Oversampling Technique).
4.	Real-Time Implementation (Future Plan): Deploy the model for real-time fraud detection using streaming frameworks like Apache Kafka and Apache Flink.
Data Preprocessing and Feature Engineering
To handle missing values and skewed features, we applied the following preprocessing steps:
•	Missing Value Imputation: Missing values were imputed using the mean.
•	Feature Transformation: Skewed features were log-transformed to reduce skewness.
•	Scaling: The dataset was normalized using StandardScaler.
Model Evaluation
We evaluated the following machine learning models:
1.	K-Nearest Neighbors (KNN): An instance-based learning algorithm used to classify transactions based on the nearest neighbors.
2.	Logistic Regression: A linear model that estimates the probability of fraud.
3.	Random Forest: An ensemble learning method that constructs multiple decision trees and outputs the class with the most votes.
4.	Support Vector Machines (SVM): A model that finds the optimal hyperplane for separating legitimate and fraudulent transactions.
5.	Naïve Bayes: A probabilistic model based on Bayes’ theorem.
6.	Neural Networks: We built Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to handle more complex relationships in the data.
Key Results
The best-performing model was K-Nearest Neighbors (KNN) with an AUPRC of 0.8583. Other models also showed competitive results:
•	Logistic Regression AUPRC: 0.7063
•	Support Vector Machine AUPRC: 0.8040
•	Random Forest AUPRC: 0.8262
•	Naïve Bayes AUPRC: 0.4073
•	CNN AUPRC: 0.7296
•	RNN AUPRC: 0.7390
Future Work
1.	Real-Time Fraud Detection: Implement real-time data streaming using Apache Kafka and Apache Flink for continuous fraud detection as transactions occur.
2.	Deep Learning Models: Explore more advanced deep learning techniques such as Transformer models to improve detection accuracy.
3.	Cloud Deployment: Move the fraud detection system to the cloud for scalability and real-time processing of larger datasets.
4.	User Behavior Analytics: Incorporate user behavior analysis for personalized fraud detection models.
Conclusion
This project demonstrates how machine learning algorithms can be effectively used to detect credit card fraud. The KNN model performed the best, but future enhancements such as real-time data processing, advanced deep learning models, and user behavior analytics could further improve fraud detection accuracy.
How to Run the Code
1.	Clone the repository:
bash
Copy code
git clone https://github.com/ahmedatteia/Credit-Card-Fraud-Detection-Using-Machine-Learning-Project
2.	Install the required Python libraries:
pip install -r requirements.txt
3.	Run the Jupyter notebook or Python script to execute the fraud detection models:
jupyter notebook fraud_detection.ipynb
4.	Download the dataset from Kaggle:  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
