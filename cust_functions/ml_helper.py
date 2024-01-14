import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV


def gridsearch(models, grid, X_train, y_train, scoring, refit = 'roc_auc'):
    """ Performs grid search for a list of models and returns the best parameters and scores """

    best_scores = {metric: {} for metric in scoring}
    best_params = {}
    fitted_models = {}  # Dictionary to store fitted models

    for model_name, name in zip(models, grid.keys()):
        print(f"Training model {name}")

        # Perform grid search
        model = GridSearchCV(model_name, grid[name], cv=5, scoring=scoring, refit=refit, n_jobs=8)
        model.fit(X_train, y_train)

        # Store the best parameters and the fitted model
        best_params[name] = model.best_params_
        fitted_models[name] = model.best_estimator_  # Storing the fitted model

        # Store the best scores for each metric
        for metric in scoring:
            mean_key = f"mean_test_{metric}"
            std_key = f"std_test_{metric}"
            best_scores[metric][name] = {
                "mean": model.cv_results_[mean_key][model.best_index_],
                "std": model.cv_results_[std_key][model.best_index_]
            }

        # Print best parameters and best score for ROC AUC
        print(f"Best parameters for {name}: {model.best_params_}")
        print(f"Best ROC AUC score for {name}: {best_scores['roc_auc'][name]}")

    # Creating a DataFrame for each metric
    best_models = {metric: pd.DataFrame({"Model": list(best_scores[metric].keys()),
                                        "Best Score": [scores["mean"] for scores in best_scores[metric].values()],
                                        "St. Dev.": [scores["std"] for scores in best_scores[metric].values()],
                                        "Best Params": [best_params[model] for model in best_scores[metric]]})
                   for metric in scoring}
    
    return best_models, best_params, fitted_models

def predict_ml_model(model, X_train, y_train, X_test, y_test):
    """ Fits a model and returns the predictions and the confusion matrix """

    # Fitting the model
    model.fit(X_train, y_train)

    # Predicting the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Creating the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, y_pred_proba, cm

def print_ml_metrics(cm, y_test, y_pred_proba):
    """ Prints the metrics for a given confusion matrix """

    # Calculating metrics
    recall_pheno2 = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-10)
    recall_pheno1 = cm[0,0] / (cm[0,0] + cm[0,1] + 1e-10)
    precision_pheno1 = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-10)
    precision_pheno0 = cm[0,0] / (cm[0,0] + cm[1,0] + 1e-10)
    f1_pheno2 = 2 * precision_pheno1 * recall_pheno2 / (precision_pheno1 + recall_pheno2)
    f1_pheno1 = 2 * precision_pheno0 * recall_pheno1 / (precision_pheno0 + recall_pheno1)
    f1_macro = (f1_pheno2 + f1_pheno1) / 2
    accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    
    # Printing metrics
    print("Accuracy: %f" % accuracy)
    print("AUC: %f" % roc_auc_score(y_test, y_pred_proba))
    print("F1 Macro: %f" % f1_macro)
    print("F1 pheno2: %f" % f1_pheno2)
    print("F1 pheno1: %f" % f1_pheno1)
    print("Recall pheno2: %f" % recall_pheno2)
    print("Recall pheno1: %f" % recall_pheno1)
    print("Precision pheno2: %f" % precision_pheno1)
    print("Precision pheno1: %f" % precision_pheno0)

def extract_top_features(fitted_models, X_train, input_data_preprocessed, top_n=30):
    """ Extracts the top features for RF and AdaBoost models """

    feature_importances = {}
    top_features = {}
    top_features_with_names = {}

    # Extracting feature importances
    for model_name, model in fitted_models.items():
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importances[model_name] = model.feature_importances_
            # normalizing to 0-1 range
            feature_importances[model_name] = feature_importances[model_name] / feature_importances[model_name].sum()

    # Sorting and selecting top features
    for model_name, importances in feature_importances.items():
        feature_names = X_train.columns
        features_importance = zip(feature_names, importances)
        sorted_features = sorted(features_importance, key=lambda x: x[1], reverse=True)
        top_features[model_name] = sorted_features[:top_n]

    # Adding protein names to top features
    for model_name, features in top_features.items():
        feature_names = [feature[0] for feature in features]
        feature_scores = [round(feature[1], 3) for feature in features]
        protein_names = [input_data_preprocessed['Protein'].tolist()[feature] for feature in feature_names]
        combined_features = zip(feature_names, protein_names, feature_scores)
        top_features_with_names[model_name] = list(combined_features)

    return top_features_with_names


def find_common_features(top_features_with_names, model1_name, model2_name):
    """ Finds the common features between two models """

    # Extract top features for each model
    model1_top_features = top_features_with_names[model1_name]
    model2_top_features = top_features_with_names[model2_name]

    # Create sets of top feature names for comparison
    model1_top_feature_names = set([feature[0] for feature in model1_top_features])
    model2_top_feature_names = set([feature[0] for feature in model2_top_features])

    # Find common features
    common_features = model1_top_feature_names.intersection(model2_top_feature_names)

    # Dictionary to store common features with their positions, importance values, and protein names
    common_features_info = {}

    # Find positions, importance values, and protein names of common features in each model
    for feature in common_features:
        model1_feature_info = next((item for item in model1_top_features if item[0] == feature), None)
        model2_feature_info = next((item for item in model2_top_features if item[0] == feature), None)

        if model1_feature_info and model2_feature_info:
            common_features_info[feature] = {
                f'{model1_name}_Position': model1_top_features.index(model1_feature_info) + 1,
                f'{model1_name}_Importance': model1_feature_info[2],
                f'{model1_name}_Protein': model1_feature_info[1],
                f'{model2_name}_Position': model2_top_features.index(model2_feature_info) + 1,
                f'{model2_name}_Importance': model2_feature_info[2],
                f'{model2_name}_Protein': model2_feature_info[1]
            }

    return common_features_info