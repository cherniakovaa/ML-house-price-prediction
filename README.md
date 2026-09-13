# Machine Learning — House Price Prediction

A Machine Learning project focused on predicting rental housing prices using the **XGBoost** algorithm.

---

## Project Overview

The model predicts rental housing prices based on data from the Warsaw rental housing market.

During the training process, hyperparameters such as `n_estimators` and `learning_rate` were optimized to improve the model's prediction accuracy.

The project includes the trained model, preprocessing components, the dataset, and a Jupyter Notebook containing the analysis and model development process.

---

## Dataset

The model was trained using the **Warsaw Flat Rent Prices** dataset available on Kaggle:

[Warsaw Flat Rent Prices — Kaggle](https://www.kaggle.com/datasets/beksultankarimov/warsaw-flat-rent-prices/data)

---

## Machine Learning Model

The project uses **XGBoost** for rental price prediction.

Selected hyperparameters were optimized during model training, including:

* `n_estimators`
* `learning_rate`

---

## Technologies Used

* Python
* Jupyter Notebook
* XGBoost
* Pandas
* Scikit-learn

---

## Project Files

* `FinalProject.ipynb` — Jupyter Notebook containing the project analysis and model development
* `data_clean.csv` — processed dataset
* `model_xgb.pkl` — trained XGBoost model
* `scaler.pkl` — saved data scaler
* `test_pipeline.py` — model pipeline testing
