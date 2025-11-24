# 📡 Telecom Customer Churn Prediction App

## Project Overview
This project demonstrates a complete Machine Learning workflow, from **Exploratory Data Analysis (EDA)** and **Feature Engineering** to building a production-ready **Random Forest Classifier** to predict customer churn in a telecommunications dataset.

The main goal is to identify customers at **"HIGH RISK"** of leaving the company (Churn = 1) so that the marketing and retention teams can intervene proactively.

The final model is deployed as an interactive, real-time web application using **Streamlit**, ensuring a professional and engaging portfolio experience.

## 📈 Model Performance & Business Metrics

In Churn Prediction, **Recall** (identifying true churners) is prioritized to maximize retention chances, but it must be balanced with **Precision** (avoiding costly offers to stable customers).

We used **Randomized Search CV (RSCV)** to optimize the **Random Forest Classifier** and fine-tuned the probability threshold from the default 0.50 to **0.55** to achieve an optimal balance.

### Final Metrics (Threshold = 0.55)

| Metric | Class 0 (No Churn) | Class 1 (Churn Risk) | Weighted Avg |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.89 | **0.53** | 0.80 |
| **Recall** | 0.77 | **0.75** | 0.76 |
| **F1-Score** | 0.82 | **0.62** | 0.77 |

**Business Insight:**
By using a threshold of **0.55**, the model successfully identifies **75% of actual churning customers (High Recall)**, while ensuring that the outreach efforts are correct **53% of the time (Improved Precision)**.

## 📂 Project Architecture & Files

The project is structured for easy setup and deployment:

| File Name | Purpose | Notes |
| :--- | :--- | :--- |
| **`app.py`** | **Streamlit Application** | Contains all the UI, input logic, model loading, and real-time prediction/result display. |
| **`churn_deployment_assets.joblib`** | **Trained ML Pipeline** | The saved `joblib` file containing the `ColumnTransformer` (preprocessor) and the optimized `RandomForestClassifier` object. |
| **`requirements.txt`** | **Dependencies** | Lists all necessary Python libraries (Streamlit, Scikit-learn==1.6.1, pandas, joblib). |
| **`.streamlit/config.toml`** | **UI Configuration** | Sets the forced **Dark Theme** and custom colors for a professional look. |
| **`Churn prediction.ipynb`** | **Full ML Notebook** | Complete record of EDA, Feature Engineering, Model Selection, and Hyperparameter Tuning. |

## 🧠 Machine Learning Workflow Details

### 1. Data Cleaning & Selection
* **Leakage Removal:** Columns causing **Data Leakage** (`Churn Score`, `CLTV`, `Churn Reason`, etc.) and unnecessary identifiers (`CustomerID`, location data) were dropped.
* **Handling Missing Values:** The `Total Charges` column was converted to a numeric type, revealing $\text{NaN}$ values which were handled by the preprocessing pipeline.

### 2. Feature Selection
* **Numerical Features:**
    * **Tenure Months** ($\text{R} = -0.35$)
    * **Monthly Charges** ($\text{R} = 0.19$)
    * **Total Charges** ($\text{R} = -0.20$)
* **Categorical Features:**
    * **Contract**, **Online Security**, **Tech Support**, and **Payment Method** showed the highest statistical significance ($\text{low p-values}$ in the $\text{Chi}^2$ test) against the target variable.

### 3. Preprocessing Pipeline
A `ColumnTransformer` was used inside a $\text{Pipeline}$ to ensure consistency:
* **Numerical:** `SimpleImputer(strategy='mean')` followed by `StandardScaler()` for normalization.
* **Categorical:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(sparse_output=False)` for conversion.

### 4. Model & Tuning
* **Base Model:** $\text{RandomForestClassifier}$ was selected over $\text{Logistic Regression}$ due to its non-linear nature, using `class_weight='balanced'` to handle the inherent class imbalance in churn data.
* **Hyperparameter Optimization:** $\text{RandomizedSearchCV}$ was used to efficiently search the optimal combinations for `n_estimators`, `max_depth`, and `min_samples_split`, focusing on maximizing the **Recall** score.

## 🚀 How to Run Locally

1.  **Clone the Repository** and navigate to the project directory.
2.  **Activate Virtual Environment** (e.g., `.\venv\Scripts\activate`).
3.  **Install Dependencies** (ensure you are using `scikit-learn==1.6.1`):
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

The application will automatically open in your web browser, ready for testing.

---
Developed by **Moizz**