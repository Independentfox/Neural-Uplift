# Neural Uplift 🧠📈  
**Neural Uplift is a machine learning project that focuses on understanding which user actions (like seeing an ad or interacting with a product) actually influence purchasing decisions — and by how much. This helps businesses make smarter, data-driven decisions to boost conversions.**

---

## 🔍 Overview  
**Neural Uplift** is a comprehensive uplift modeling pipeline built around the **S-learner strategy**, using neural networks to estimate the causal impact of user features on conversion outcomes. It aims to uncover the **incremental effect** of different touchpoints (like impressions or visits) on a user's likelihood to convert.

Key applications include:  
- **Causal Inference:** Differentiating causal effects from simple correlations.  
- **Maximum likelihood in market:** Discovering which factors genuinely drive conversions.  
- **Targetting customers:** Targeting users more effectively based on predicted behavioral change.

The core model is implemented using the **S-learner** from the `causalml` library, paired with `MLPRegressor` from scikit-learn for the neural network backbone.

---

## 🧪 Key Components  
- **Dataset:** Based on the Criteo Uplift dataset, which includes anonymized user data across 12 numerical features and conversion labels. ([Available on Kaggle](https://www.kaggle.com/datasets/arashnic/uplift-modeling))  
- **Feature Attribution:** Uses permutation-based importance to evaluate the contribution of each feature to the estimated uplift.  
- **Uplift Modeling:** Implements the S-learner strategy with a neural network (`MLPRegressor`) to model treatment effects.  
- **Model Persistence:** Saves the trained uplift model using `Pickle` for reuse and deployment.  
- **Evaluation Metrics:** Supports metrics like **Conversion Lift**, **Uplift Gain**, and feature-wise **incremental effect** to measure model effectiveness.

---

## 🛠 Technologies Used  
- **Programming Language:** Python 3.12+  
- **Development Environment:** Jupyter Notebooks  
- **Libraries:**  
  - **Data Handling:** Pandas, NumPy  
  - **Visualization:** Matplotlib, Seaborn  
  - **Machine Learning:** scikit-learn  
  - **Uplift Modeling:** causalml (S-learner with neural network)  
  - **Neural Networks:** MLPRegressor from scikit-learn  

