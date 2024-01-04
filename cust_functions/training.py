import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def train(train_data, model, optimizer, loss_fn, device, update_confusion_matrix=False):

    model.train()
    running_loss = 0
    if update_confusion_matrix:
        running_confusion_matrix = np.zeros((2, 2))


    for batch in train_data:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        train_loss = loss_fn(out, batch.y)
        train_loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += train_loss.item()

        if update_confusion_matrix:
            pred = out.argmax(dim=1)
            running_confusion_matrix += confusion_matrix(batch.y.cpu(), pred.cpu(), labels=[0, 1])
        
    if update_confusion_matrix:
        return running_loss / len(train_data), running_confusion_matrix
    else:
        return running_loss / len(train_data)


def validate(test_data, model, loss_fn, device, update_confusion_matrix=False):

    model.eval()
    running_loss = 0
    if update_confusion_matrix:
        running_confusion_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            running_loss += loss.item()

            if update_confusion_matrix:
                pred = out.argmax(dim=1)
                running_confusion_matrix += confusion_matrix(batch.y.cpu(), pred.cpu(), labels=[0, 1])
    
    if update_confusion_matrix:
        return running_loss / len(test_data), running_confusion_matrix
    else:
        return running_loss / len(test_data)

def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    recall_phen1 = tp / (tp + fn + 1e-6)
    recall_phen2 = tn / (tn + fp + 1e-6)
    precision_phen1 = tp / (tp + fp + 1e-6)
    precision_phen2 = tn / (tn + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_phen1 = 2 * (precision_phen1 * recall_phen1) / (precision_phen1 + recall_phen1 + 1e-6)
    f1_phen2 = 2 * (precision_phen2 * recall_phen2) / (precision_phen2 + recall_phen2 + 1e-6)
    roc_auc = (recall_phen1 + recall_phen2) / 2
    return recall_phen1, recall_phen2, precision_phen1, precision_phen2, accuracy, f1_phen1, f1_phen2, roc_auc

def update_results(result_dic, fold, epoch, train_loss, train_cm, val_loss, val_cm):
    # Calculate metrics for train and val
    train_metrics = calculate_metrics(train_cm)
    val_metrics = calculate_metrics(val_cm)

    # Mapping of metric names to their values
    metrics = {
        'train': (train_loss, train_metrics, train_cm),
        'val': (val_loss, val_metrics, val_cm)
    }

    # If first epoch, initialize result_dic
    if epoch == 0:
        result_dic[fold + 1] = {'epoch': [epoch + 1]}
        for metric_type in ['train', 'val']:
            loss, calculated_metrics, cm = metrics[metric_type]
            result_dic[fold + 1][f'{metric_type}_loss'] = [loss]
            metric_names = ['recall_phen1', 'recall_phen2', 'precision_phen1', 'precision_phen2', 'accuracy', 'f1_phen1', 'f1_phen2', 'roc_auc']
            for i, name in enumerate(metric_names):
                result_dic[fold + 1][f'{metric_type}_{name}'] = [calculated_metrics[i]]
            result_dic[fold + 1][f'{metric_type}_cm'] = [cm]

    # Append metrics for subsequent epochs
    else:
        result_dic[fold + 1]['epoch'].append(epoch + 1)
        for metric_type in ['train', 'val']:
            loss, calculated_metrics, cm = metrics[metric_type]
            result_dic[fold + 1][f'{metric_type}_loss'].append(loss)
            metric_names = ['recall_phen1', 'recall_phen2', 'precision_phen1', 'precision_phen2', 'accuracy', 'f1_phen1', 'f1_phen2', 'roc_auc']
            for i, name in enumerate(metric_names):
                result_dic[fold + 1][f'{metric_type}_{name}'].append(calculated_metrics[i])
            result_dic[fold + 1][f'{metric_type}_cm'].append(cm)

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

        # Plot training and validation ROC AUC on third column
        axes[fold][2].plot(epochs, metrics["train_roc_auc"], 'b-', label='Train ROC AUC')
        axes[fold][2].plot(epochs_for_val_accuracy, metrics["val_roc_auc"], 'r-', label='Validation ROC AUC')
        axes[fold][2].set_xlabel('Epochs')
        axes[fold][2].set_ylabel('ROC AUC')
        axes[fold][2].legend()
        axes[fold][2].set_title(f"ROC AUC for fold {fold + 1}")

    plt.tight_layout()
    plt.show()

def print_val_results(results):
    print(f"Average validation accuracy: {np.round(np.mean([results[fold]['val_accuracy'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 3)} +/- {np.round(np.std([results[fold]['val_accuracy'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 2)}")
    print(f"Average validation ROC_AUC: {np.round(np.mean([results[fold]['val_roc_auc'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 3)} +/- {np.round(np.std([results[fold]['val_roc_auc'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 2)}")
    print(f"Average validation F1 Phen1: {np.round(np.mean([results[fold]['val_f1_phen1'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 3)} +/- {np.round(np.std([results[fold]['val_f1_phen1'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 2)}")
    print(f"Average validation F1 Phen2: {np.round(np.mean([results[fold]['val_f1_phen2'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 3)} +/- {np.round(np.std([results[fold]['val_f1_phen2'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()]), 2)}")

def plot_confusion_matrix(results):
    # Plot confusion matrix for best validation epoch
    cm = np.mean([results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()], axis=0)
    cm_std = np.std([results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1] for fold in results.keys()], axis=0)
    
   # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(x=j, y=i, s= f"{round(cm[i, j],1)} +/- {round(cm_std[i, j],1)}" , va='center', ha='center', size = 15)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")


    # Normalize each confusion matrix and collect them
    normalized_cms = []
    for fold in results.keys():
        cm = results[fold]['val_cm'][results[fold]['best_val_epoch'] - 1].astype(np.float64)
        norm_cm = cm / cm.sum(axis=1, keepdims=True)
        normalized_cms.append(norm_cm)

    # Calculate mean and standard deviation of normalized confusion matrices
    mean_normalized_cm = np.mean(normalized_cms, axis=0)
    std_normalized_cm = np.std(normalized_cms, axis=0)


    # Normalise confusion matrix
    axes[1].matshow(mean_normalized_cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(mean_normalized_cm.shape[0]):
        for j in range(mean_normalized_cm.shape[1]):
            axes[1].text(x=j, y=i, s=f"{mean_normalized_cm[i, j]:.2f} +/- {std_normalized_cm[i, j]:.2f}", 
                         va='center', ha='center', size=15)
    axes[1].set_title("Normalised Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()