# A Machine Learning Framework for COVID-19 Mortality Prediction and Clinical Risk Stratification

**Authors:** Raghavendra Chekuri, Srujith Vempati, Rohit Emmadi, Jadhav Govind, Dr. Krishna Bhargavi Yerraganti, Dr. Konda Adilakshmi

**Institution:** Department of Computer Science & Engineering, Gokaraju Rangaraju Institute of Engineering and Technology, Hyderabad, India

## 📌 Overview

This project presents an improved machine learning (ML) framework designed to predict mortality risk among COVID-19 patients for clinical triage and resource allocation. Developed using a comprehensive dataset of 220,657 confirmed COVID-19 patients, this framework systematically addresses common limitations in previous studies, such as small dataset sizes, severe class imbalance, and an over-reliance on accuracy as a sole evaluation metric.

The study evaluates seven ML models and identifies Logistic Regression as the best-performing classifier, achieving an ROC-AUC score of 0.8914 ± 0.0018. By adjusting the decision threshold to θ = 0.20, the model achieves a maximum sensitivity of 95.9%. The system also introduces an advanced five-level clinical risk stratification method and utilizes SHAP values for high model interpretability.

## 📊 Dataset

* **Source:** Mexican COVID-19 Patient Pre-condition Dataset, released by the Mexican Ministry of Health.


* **Size:** 220,657 confirmed COVID-19 positive records utilized out of 566,602 total records.


* **Class Imbalance:** 193,536 (87.7%) survivors vs. 27,121 (12.3%) deaths, creating a 7:1 ratio.



## 🏗 System Architecture & Methodology

The proposed framework operates through a six-phase sequential pipeline:

<img width="693" height="882" alt="Picture1" src="https://github.com/user-attachments/assets/6dda94e0-c73c-4e67-a246-ddd558bd665f" />

Figure : Proposed System Architecture

### Phase 1 & 2: Preprocessing and EDA

* Replicated prior studies to establish a baseline performance.


* Handled domain-specific missing values (encoded as 97, 98, 99).


* Dropped features with >30% missing data (intubed, icu, pregnancy, contact_other_covid) and applied mode imputation to chronic disease features with <1% missing data.



<img width="1254" height="570" alt="Picture6" src="https://github.com/user-attachments/assets/5a740f48-3341-41a0-a0de-89e334beb57f" />
Figure : Exploratory Data Analysis - Overall death distribution, age group mortality, and sex-wise mortality

<img width="1160" height="537" alt="Picture7" src="https://github.com/user-attachments/assets/f35dfddb-6b30-42bb-aa9c-723392f1b49b" />
Figure 7: Chronic disease impact analysis

<img width="1326" height="596" alt="Picture8" src="https://github.com/user-attachments/assets/88d05a8e-bb0f-4506-9078-f9131b2192fe" />
Figure 8: Age distribution analysis

### Phase 3: Class Imbalance Handling

* Addressed the severe 7:1 class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to achieve a 50:50 training distribution.


* The SMOTE interpolation formula used for synthetic sample generation:



$$\tilde{x} = x_i + \lambda \cdot (x_{nn} - x_i)$$




<img width="1303" height="578" alt="Picture3" src="https://github.com/user-attachments/assets/8bfe808d-cf70-4a20-8333-1394af03f203" />
Figure 3: Balancing methods comparison

### Phase 4: Systematic Feature Selection

* Ten clinically significant features were systematically selected over 13 candidates.


* Selected features: pneumonia, age, hypertension, diabetes, renal chronic disease, obesity, sex, COPD, other disease, and cardiovascular disease.


* Selection was based on the combined average rank of three independent methods:


* **Chi-Square Test** ($\chi^2$): Measured statistical dependence.



$$\chi^2 = \sum_i \sum_j \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$



* **Mutual Information** ($MI$): Measured the decrease in target variable uncertainty.



$$MI(X, Y) = \sum_x \sum_y P(x, y) \cdot \log\left[\frac{P(x, y)}{P(x) \cdot P(y)}\right]$$



* **Random Forest Importance**: Based on average decrease in Gini impurity ($G(t)$).



$$G(t) = 1 - \sum_k p_k^2$$






<img width="1441" height="946" alt="Picture2" src="https://github.com/user-attachments/assets/0ad5b176-63c0-4dae-bc23-5ab8c99742f8" />
Figure : Visualization of feature selection based on Chi-Square, MI, and RF Importance

### Phase 5: Model Training

* Seven classifiers were evaluated on the SMOTE-balanced training set: Logistic Regression, Random Forest, Decision Tree, KNN, XGBoost, LightGBM, and a Stacking Ensemble.


* Logistic Regression optimizes the probability of death using binary cross-entropy loss:



$$P(y = 1 \vert{} x) = \frac{1}{1 + e^{-(w^T x + b)}}$$




### Phase 6: Threshold Optimization & Evaluation

* Decreased the classification decision threshold from standard θ = 0.50 to an optimal θ = 0.20 to prioritize clinical sensitivity (reducing false negatives).


* Achieved a 5-fold cross-validated statistical stability.



<img width="992" height="697" alt="Picture5" src="https://github.com/user-attachments/assets/4c339fa8-5ded-4f01-9f85-08219d7473a9" />
Figure : Cross-validation stability plots

## 🚀 Key Results

### Model Performance

* **Logistic Regression** outperformed complex ensemble models, yielding an ROC-AUC of 0.8930 (held-out test set) and correctly identifying 4,421 out of 5,335 death cases at the standard threshold.



<img width="547" height="487" alt="Picture9_300" src="https://github.com/user-attachments/assets/7b1f32cb-b937-4f32-87d9-747d8fed91e6" />

Figure : Model comparison and Confusion Matrix for Logistic Regression

### Threshold Optimization Impact

* Reducing the threshold to θ = 0.20 increased sensitivity from 82.8% to 95.9%.


* This successfully caught 5,114 out of 5,335 deaths (reducing missed deaths from 919 to 221).



<img width="1435" height="752" alt="Picture11" src="https://github.com/user-attachments/assets/03cec7c3-68f4-4266-8d93-ea6490b044ab" />
Figure: Threshold optimization analysis

### Five-Tier Clinical Risk Stratification

The framework translates predictions into actionable hospital protocols:

1. **Very Low (< 10% prob):** 0.5% observed mortality. Action: Home monitoring.


2. **Low (10–25% prob):** 2.7% observed mortality. Action: Routine ward admission.


3. **Moderate (25–50% prob):** 9.7% observed mortality. Action: Close observation, oxygen therapy.


4. **High (50–75% prob):** 22.1% observed mortality. Action: ICU prepared.


5. **Critical (> 75% prob):** 46.5% observed mortality. Action: Immediate ICU.



### SHAP Explainability

* SHAP (SHapley Additive exPlanations) values were used to ensure model transparency.


* **Top Risk Drivers:** Pneumonia, age, hypertension, and diabetes were identified as the strongest individual feature contributors to mortality risk predictions.
