# Glucose Level Prediction - Model Stacking and Ensembling

This repository contains multiple machine learning models and techniques used to predict glucose levels based on various features. The project focuses on experimenting with different algorithms, model stacking, and ensemble learning approaches to improve prediction accuracy.

## Project Overview

In this project, I explored different machine learning models and their combinations to create an ensemble model that can predict glucose levels effectively. The main approaches include:

1. **Individual Models**: Training different models like XGBoost, LightGBM, Random Forest, and CatBoost on the dataset to assess their performance individually.
2. **Model Stacking**: Combining predictions from multiple models (XGBoost, LightGBM, and CatBoost) with a Ridge Regression meta-model to improve overall performance.
3. **Ensemble Learning with K-Folds**: Implementing K-Fold cross-validation to train and ensemble models like XGBoost, LightGBM, Random Forest, and TabNet for final predictions.

## Key Techniques

### 1. **Individual Models**:
- **XGBoost**: Gradient boosting framework for high performance with structured data.
- **LightGBM**: High-performance gradient boosting framework based on decision trees.
- **Random Forest**: Ensemble of decision trees to improve prediction accuracy.
- **CatBoost**: Gradient boosting framework optimized for categorical features.

### 2. **Model Stacking**:
Stacking combines multiple model predictions into a meta-model to boost accuracy. The steps include:
- Training base models: XGBoost, LightGBM, and CatBoost.
- Using a Ridge Regression meta-model to combine predictions from base models.

### 3. **Ensemble Learning with K-Folds**:
- **K-Fold Cross-Validation**: The dataset is divided into 5 folds. Models are trained on 4 folds and validated on the remaining fold. This is repeated for all folds.
- **Ensembling**: Predictions from models like XGBoost, LightGBM, Random Forest, and TabNet are averaged for each fold, and final predictions are averaged across folds.

### 4. **Final Ensemble Predictions**:
The ensemble predictions from all models and folds are averaged to create the final predictions submitted as `submission.csv`.

## Dataset

The dataset used for this project can be downloaded from Kaggle:  
[BRIST1D Kaggle Competition Dataset](https://www.kaggle.com/competitions/brist1d)  


## Installation and Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/SheemaMasood381/Blood-Glucose-Levels-Prediction-2024-kaggle-competition.git
    cd Blood-Glucose-Levels-Prediction-2024-kaggle-competition
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from the [Kaggle BRIST1D Competition](https://www.kaggle.com/competitions/brist1d) and place the files in the appropriate folder (e.g., `data/`).

4. Run the Jupyter notebooks step-by-step to train models and generate predictions.

## Results and Evaluation

- **Stacking Approach**: Combining XGBoost, LightGBM, and CatBoost with a Ridge Regression meta-model provided robust predictions.
- **K-Fold Ensembling**: Combining XGBoost, LightGBM, Random Forest, and TabNet predictions across folds resulted in the best generalization.
- **Metrics**: Models were evaluated using RMSE, R², and MAE.

## Submission

The final ensemble predictions are saved in `submission.csv`. This file contains the predicted glucose levels for the test data.

## Future Work

- Experiment with additional models like Neural Networks for further improvement.
- Perform hyperparameter optimization using Grid Search or Random Search.
- Explore advanced ensembling techniques like weighted averages or Bayesian model averaging.

## Acknowledgments

- Dataset provided by the [BRIST1D Kaggle Competition](https://www.kaggle.com/competitions/brist1d).
- Libraries used: [XGBoost](https://xgboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), and others.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
