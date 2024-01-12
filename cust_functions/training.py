import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_cv(create_model_fn, loss_fn, optimizer_fn, scheduler_fn, train_graph_data, train_labels, batch_size, num_epochs, device, save_path = None, 
                 save = True, use_scheduler = True, early_stopping_patience = 30, SEED = 42, FOLDS = 3, hierarchical = False):
    results = {}
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


    for fold, (train_idx, test_idx) in enumerate(skf.split(train_graph_data, train_labels)):
        print(f"Fold: {fold + 1}")
        train_loader = DataLoader([train_graph_data[i] for i in train_idx], batch_size=batch_size)
        val_loader = DataLoader([train_graph_data[i] for i in test_idx], batch_size=batch_size)

        model = create_model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        if use_scheduler:
            scheduler = scheduler_fn(optimizer)

        best_val_loss = np.inf
        best_model = None
        patience = early_stopping_patience
        best_epoch = 0

        for epoch in range(num_epochs):
            if hierarchical:
                train_loss, train_cm, train_roc_auc = train(train_loader, model, optimizer, loss_fn, device, hierarchical = True)
            else:
                train_loss, train_cm, train_roc_auc = train(train_loader, model, optimizer, loss_fn, device)
            if hierarchical:
                val_loss, val_cm, val_roc_auc = validate(val_loader, model, loss_fn, device, hierarchical = True)
            else:
                val_loss, val_cm, val_roc_auc = validate(val_loader, model, loss_fn, device)
            results = update_results(results, fold, epoch, train_loss, train_cm, train_roc_auc, val_loss, val_cm, val_roc_auc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                patience = early_stopping_patience
                best_epoch = epoch + 1
                results[fold + 1]['best_val_epoch'] = best_epoch
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if use_scheduler:
                scheduler.step(val_loss)

            
            if epoch % 5 == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {results[fold + 1]['val_accuracy'][-1]:.4f}, Val F1 Macro: {results[fold + 1]['val_f1_macro'][-1]:.4f}")

        if save:
            model.load_state_dict(best_model)
            save_fold_path = f"{save_path}_fold_{fold + 1}.pt"
            torch.save(model.state_dict(), save_fold_path)

    return results

def run_training(create_model_fn, loss_fn, optimizer_fn, scheduler_fn, train_graph_data, batch_size, num_epochs, device, save_path, use_scheduler = True):

    train_loader = DataLoader(train_graph_data, batch_size=batch_size)
    model = create_model_fn().to(device)
    optimizer = optimizer_fn(model.parameters())
    if use_scheduler:
        scheduler = scheduler_fn(optimizer)

    for epoch in range(num_epochs):
        train_loss, _, _ = train(train_loader, model, optimizer, loss_fn, device)

        if use_scheduler:
            scheduler.step()

    save_fold_path = f"{save_path}.pt"
    torch.save(model.state_dict(), save_fold_path)

    return model


def train(train_data, model, optimizer, loss_fn, device, hierarchical = False):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    for batch in train_data:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        if hierarchical:
            out = model(batch.x, batch.edge_index, batch.batch, batch.node_hierarchy)
            print(out)
        else:
            out = model(batch.x, batch.edge_index, batch.batch)
        train_loss = loss_fn(out, batch.y)
        train_loss.backward()
        optimizer.step()
        running_loss += train_loss.item()

        preds = torch.softmax(out, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch.detach().y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])

    pred_labels = np.argmax(all_preds, axis=1)
    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])

    return running_loss / len(train_data), cm, roc_auc


def validate(test_data, model, loss_fn, device, hierarchical = False):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            if hierarchical:
                out = model(batch.x, batch.edge_index, batch.batch, batch.node_hierarchy)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            running_loss += loss.item()

            preds = torch.softmax(out, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])

    pred_labels = np.argmax(all_preds, axis=1)
    cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])

    return running_loss / len(test_data), cm, roc_auc


def test(test_data, models, device):
    all_preds = []
    all_prob_pos_class = []

    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            batch_preds = []
            for model in models:
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = torch.softmax(out, dim=1)
                batch_preds.append(preds.cpu().numpy())

            batch_preds = np.array(batch_preds)
            avg_prob_pos_class = np.mean(batch_preds[:, :, 1], axis=0)
            all_prob_pos_class.append(avg_prob_pos_class)

            # Majority voting
            majority_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), 0, np.argmax(batch_preds, axis=2))
            all_preds.append(majority_vote)

    all_preds = np.concatenate(all_preds)
    all_prob_pos_class = np.concatenate(all_prob_pos_class)
    return all_preds, all_prob_pos_class


def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    recall_phen1 = tp / (tp + fn + 1e-6)
    recall_phen2 = tn / (tn + fp + 1e-6)
    precision_phen1 = tp / (tp + fp + 1e-6)
    precision_phen2 = tn / (tn + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_phen1 = 2 * (precision_phen1 * recall_phen1) / (precision_phen1 + recall_phen1 + 1e-6)
    f1_phen2 = 2 * (precision_phen2 * recall_phen2) / (precision_phen2 + recall_phen2 + 1e-6)
    f1_macro = (f1_phen1 + f1_phen2) / 2
    return recall_phen1, recall_phen2, precision_phen1, precision_phen2, accuracy, f1_phen1, f1_phen2, f1_macro

def update_results(result_dic, fold, epoch, train_loss, train_cm, train_roc_auc, val_loss, val_cm, val_roc_auc):
    # Calculate metrics for train and val
    train_metrics = calculate_metrics(train_cm)
    val_metrics = calculate_metrics(val_cm)

    # Extend metric names to include ROC AUC and f1_macro
    metric_names = ['recall_phen1', 'recall_phen2', 'precision_phen1', 'precision_phen2', 
                    'accuracy', 'f1_phen1', 'f1_phen2', 'f1_macro']

    # Initialize or update result_dic
    if epoch == 0:
        result_dic[fold + 1] = {'epoch': [epoch + 1]}
        for metric_type in ['train', 'val']:
            result_dic[fold + 1][f'{metric_type}_loss'] = [train_loss if metric_type == 'train' else val_loss]
            result_dic[fold + 1][f'{metric_type}_cm'] = [train_cm if metric_type == 'train' else val_cm]
            for i, name in enumerate(metric_names):
                result_dic[fold + 1][f'{metric_type}_{name}'] = [train_metrics[i] if metric_type == 'train' else val_metrics[i]]
            result_dic[fold + 1][f'{metric_type}_roc_auc'] = [train_roc_auc if metric_type == 'train' else val_roc_auc]
    else:
        result_dic[fold + 1]['epoch'].append(epoch + 1)
        for metric_type in ['train', 'val']:
            result_dic[fold + 1][f'{metric_type}_loss'].append(train_loss if metric_type == 'train' else val_loss)
            result_dic[fold + 1][f'{metric_type}_cm'].append(train_cm if metric_type == 'train' else val_cm)
            for i, name in enumerate(metric_names):
                result_dic[fold + 1][f'{metric_type}_{name}'].append(train_metrics[i] if metric_type == 'train' else val_metrics[i])
            result_dic[fold + 1][f'{metric_type}_roc_auc'].append(train_roc_auc if metric_type == 'train' else val_roc_auc)
            result_dic[fold + 1][f'{metric_type}_roc_auc'].append(train_roc_auc if metric_type == 'train' else val_roc_auc)

    return result_dic


def plot_results(results, folds):
    fig, axes = plt.subplots(folds, 3, figsize=(15, 5 * folds))

    for fold in range(folds):
            
        metrics = results[fold + 1]
        epochs = range(1, len(metrics["train_loss"]) + 1)

        epochs_for_val_loss = range(1, len(metrics["val_loss"]) + 1)

        # Plot training and validation loss on first column
        axes[fold][0].plot(epochs, metrics["train_loss"], 'b-', label='Train Loss')
        axes[fold][0].plot(epochs_for_val_loss, metrics["val_loss"], 'r-', label='Validation Loss')
        axes[fold][0].set_xlabel('Epochs')
        axes[fold][0].set_ylabel('Loss')
        axes[fold][0].legend()
        axes[fold][0].set_title(f"Loss for fold {fold + 1}")

        epochs_for_val_accuracy = range(1, len(metrics["val_accuracy"]) + 1)

        # Plot training and validation accuracy on second column
        axes[fold][1].plot(epochs, metrics["train_accuracy"], 'b-', label='Train Accuracy')
        axes[fold][1].plot(epochs_for_val_accuracy, metrics["val_accuracy"], 'r-', label='Validation Accuracy')
        axes[fold][1].set_xlabel('Epochs')
        axes[fold][1].set_ylabel('Accuracy')
        axes[fold][1].legend()
        axes[fold][1].set_title(f"Accuracy for fold {fold + 1}")

        # Plot training and validation F1 Macro on third column
        axes[fold][2].plot(epochs, metrics["train_f1_macro"], 'b-', label='Train F1 Macro')
        axes[fold][2].plot(epochs_for_val_accuracy, metrics["val_f1_macro"], 'r-', label='Validation F1 Macro')
        axes[fold][2].set_xlabel('Epochs')
        axes[fold][2].set_ylabel('F1 Macro')
        axes[fold][2].legend()
        axes[fold][2].set_title(f"F1 Macro for fold {fold + 1}")

    plt.tight_layout()
    plt.show()

def print_val_results(results, metrics = None):
    if metrics is None:
        metrics = ['accuracy', 'roc_auc', 'f1_macro', 'f1_phen1', 'f1_phen2', 'recall_phen1', 'recall_phen2', 'precision_phen1', 'precision_phen2']
    for metric in metrics:
        avg_metric = np.mean([results[fold]['val_' + metric][results[fold]['best_val_epoch'] - 1] for fold in results.keys()])
        std_metric = np.std([results[fold]['val_' + metric][results[fold]['best_val_epoch'] - 1] for fold in results.keys()])
        print(f"Average validation {metric}: {np.round(avg_metric, 3)} +/- {np.round(std_metric, 2)}")

def plot_confusion_matrix(results, use = 'val'):
    # Plot confusion matrix for best validation epoch
    if use == 'val':
        cm = np.mean([results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()], axis=0)
        cm_std = np.std([results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()], axis=0)
    elif use == 'test':
        cm = results

   # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if use == 'val':
                axes[0].text(x=j, y=i, s=f"{cm[i, j]:.0f} +/- {cm_std[i, j]:.0f}", 
                             va='center', ha='center', size=15)
            elif use == 'test':
                axes[0].text(x=j, y=i, s=f"{cm[i, j]:.0f}", va='center', ha='center', size=15)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")


    if use == 'val':
        # Normalize each confusion matrix and collect them
        normalized_cms = []
        for fold in results.keys():
            cm = results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1].astype(np.float64)
            norm_cm = cm / cm.sum(axis=1, keepdims=True)
            normalized_cms.append(norm_cm)

        # Calculate mean and standard deviation of normalized confusion matrices
        mean_normalized_cm = np.mean(normalized_cms, axis=0)
        std_normalized_cm = np.std(normalized_cms, axis=0)
    elif use == 'test':
        mean_normalized_cm = cm / cm.sum(axis=1, keepdims=True)

    # Normalise confusion matrix
    axes[1].matshow(mean_normalized_cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(mean_normalized_cm.shape[0]):
        for j in range(mean_normalized_cm.shape[1]):
            if use == 'val':
                axes[1].text(x=j, y=i, s=f"{mean_normalized_cm[i, j]:.2f} +/- {std_normalized_cm[i, j]:.2f}", 
                             va='center', ha='center', size=15)
            elif use == 'test':
                axes[1].text(x=j, y=i, s=f"{mean_normalized_cm[i, j]:.2f}", 
                             va='center', ha='center', size=15)
    axes[1].set_title("Normalised Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

def gridsearch(models, grid, X_train, y_train, scoring, refit = 'roc_auc'):
    best_scores = {metric: {} for metric in scoring}
    best_params = {}
    fitted_models = {}  # Dictionary to store fitted models

    for model_name, name in zip(models, grid.keys()):
        print(f"Training model {name}")
        model = GridSearchCV(model_name, grid[name], cv=5, scoring=scoring, refit=refit, n_jobs=8)
        model.fit(X_train, y_train)

        # Store the best parameters and the fitted model
        best_params[name] = model.best_params_
        fitted_models[name] = model.best_estimator_  # Storing the fitted model

        # Store the best scores for each metric
        for metric in scoring:
            key = f"mean_test_{metric}"
            best_scores[metric][name] = model.cv_results_[key][model.best_index_]

        # Print best parameters and best score for ROC AUC
        print(f"Best parameters for {name}: {model.best_params_}")
        print(f"Best ROC AUC score for {name}: {best_scores['roc_auc'][name]}")

    # Creating a DataFrame for each metric
    best_models = {metric: pd.DataFrame({"Model": list(best_scores[metric].keys()),
                                        "Best Score": list(best_scores[metric].values()),
                                        "Best Params": [best_params[model] for model in best_scores[metric]]})
                   for metric in scoring}
    
    return best_models, best_params, fitted_models

def predict_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, y_pred_proba, cm

def print_ml_metrics(cm, y_test, y_pred_proba):
    recall_pheno1 = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-10)
    recall_pheno0 = cm[0,0] / (cm[0,0] + cm[0,1] + 1e-10)
    precision_pheno1 = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-10)
    precision_pheno0 = cm[0,0] / (cm[0,0] + cm[1,0] + 1e-10)
    f1_pheno1 = 2 * precision_pheno1 * recall_pheno1 / (precision_pheno1 + recall_pheno1)
    f1_pheno0 = 2 * precision_pheno0 * recall_pheno0 / (precision_pheno0 + recall_pheno0)
    accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    
    print("Recall pheno1: %f" % recall_pheno1)
    print("Recall pheno0: %f" % recall_pheno0)
    print("Precision pheno1: %f" % precision_pheno1)
    print("Precision pheno0: %f" % precision_pheno0)
    print("F1 pheno1: %f" % f1_pheno1)
    print("F1 pheno0: %f" % f1_pheno0)
    print("Accuracy: %f" % accuracy)
    print("AUC: %f" % roc_auc_score(y_test, y_pred_proba))

def extract_top_features(fitted_models, X_train, input_data_preprocessed, top_n=30):
    feature_importances = {}
    top_features = {}
    top_features_with_names = {}

    # Extracting feature importances
    for model_name, model in fitted_models.items():
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importances[model_name] = model.feature_importances_

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