# SBA Loan Default Prediction and Profit Optimization
Machine learning framework for predicting Small Business Administration loan defaults with profit-maximizing decision thresholds

## Project Description  
The project focuses on building models that not only predict loan defaults but also optimize lender profitability using a cost-sensitive profit framework.  

## Objective  
- Predict the likelihood of loan default using historical SBA loan data (~890K records).  
- Incorporate business costs into model evaluation to maximize net profit from loan approvals.  
- Recommend an approval strategy (cutoff probability and approval rate) that balances profitability and risk.  

## Data & Preprocessing
Dataset: SBA loan records (899K original → 857K final, 95.3% retention rate)
Target variable: Loan status (MIS_Status: Paid in Full vs Default)

## Data Quality & Cleaning:
* Missing value analysis across 27 variables with strategic treatment
* Temporal outlier removal (1987-2014 timeframe filtering)
* Currency formatting and data type conversions

## Advanced Feature Engineering:
* Correlation-based selection for numerical variables (Term: -0.314, GrAppv: -0.118)
* Default rate analysis for categorical variables (avoiding correlation bias with dummy variables)
* Grouped 1,311 NAICS codes into 4 business risk categories using statistical methodology (>1000 loan threshold for reliability)
* Risk-based state clustering using natural breakpoints in default rates

## Methodological Framework:

* Train-test split: 60-40 with preserved class distribution
* Feature scaling using StandardScaler for distance-based algorithms

## Modeling Approach
Implemented and compared 14 algorithms, including:
* Ensemble methods: Bagging, Random Forest, Gradient Boosting, AdaBoost, XGBoost
* Regularized regression: Logistic Regression (Standard, Lasso, Ridge, ElasticNet)
* Others: k-Nearest Neighbors, Neural Networks, LDA, QDA, Single Decision Tree

* Hyperparameter optimization via GridSearchCV with 5-fold cross-validation
* Evaluation metrics: Accuracy, Recall, Precision, F1, ROC-AUC, Gains & Lift charts

## Profit framework:  
Approve + repaid loan → +5% of disbursement
Approve + defaulted loan → –25% of disbursement
  - Denied loan → **$0**  

## Results  
- **Best Model:** Bagging (Decision Trees)  
- **Performance:**  
  - ROC-AUC: 0.97  
  - Net Profit: **$2.64 Billion**  
  - Optimal cutoff threshold: **0.40**  
  - Recommended approval rate: **~76% of least risky applicants**
  - **Competitive Analysis**: Bagging achieved the highest net profit of ~$2.64B, outperforming Gradient Boosting by ~$150M, XGBoost by ~$200M, and Random Forest by ~$280M, making it the most cost-effective model under the profit-sensitive framework. Bagging outperformed Gradient Boosting ($2.49B) and XGBoost ($2.44B).  

##  Business Impact  
- Provides lenders with a **data-driven framework** to maximize profit while controlling risk exposure.  
- Recommends approving only the safest ~76% of applicants, aligning lending practices with financial sustainability.  
- Demonstrates the value of **profit-sensitive evaluation** beyond traditional accuracy-based metrics.  


##  Tech Stack  
- **Languages/Tools:** Python, Jupyter Notebook  
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn  


