import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def evaluate_model_detailed(model, test_loader, device, class_names):
    """
    Perform detailed evaluation of a trained PyTorch model
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probas.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Basic metrics
    accuracy = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Create a DataFrame for better display
    metrics_df = pd.DataFrame(report).transpose()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Per-class performance analysis
    class_performance = []
    for i, cls in enumerate(class_names):
        # Extract samples for this class
        class_indices = (y_true == i)
        class_samples = sum(class_indices)
        correctly_classified = sum((y_pred == i) & class_indices)
        accuracy = correctly_classified / class_samples if class_samples > 0 else 0
        
        # Most common misclassifications
        if class_samples > 0:
            misclassified_indices = (y_true == i) & (y_pred != i)
            if sum(misclassified_indices) > 0:
                misclassified_as = y_pred[misclassified_indices]
                common_misclass = [(class_names[label], count) 
                                   for label, count in zip(*np.unique(misclassified_as, return_counts=True))]
                common_misclass = sorted(common_misclass, key=lambda x: x[1], reverse=True)
                common_misclass = common_misclass[:3]  # Top 3 misclassifications
            else:
                common_misclass = []
        else:
            common_misclass = []
        
        class_performance.append({
            'class': cls,
            'samples': class_samples,
            'accuracy': accuracy,
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1_score': report[cls]['f1-score'],
            'common_misclassifications': common_misclass
        })
    
    # Create DataFrame for better display
    class_perf_df = pd.DataFrame(class_performance)
    
    # Plot per-class metrics
    plt.figure(figsize=(14, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    bar_width = 0.2
    index = np.arange(len(class_names))
    
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, class_perf_df[metric], bar_width, 
                label=metric.replace('_', ' ').title())
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_performance.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'classification_report': metrics_df,
        'confusion_matrix': cm,
        'class_performance': class_perf_df,
    }

def analyze_misclassifications(model, test_loader, device, class_names, feature_names=None):
    """
    Analyze misclassifications to understand model weaknesses
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        class_names: List of class names
        feature_names: List of feature names (optional)
        
    Returns:
        DataFrame with misclassification analysis
    """
    model.eval()
    misclassifications = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            mask = (predicted != y_batch)
            if torch.any(mask):
                misclassified_X = X_batch[mask].cpu().numpy()
                misclassified_y_true = y_batch[mask].cpu().numpy()
                misclassified_y_pred = predicted[mask].cpu().numpy()
                misclassified_probs = probas[mask].cpu().numpy()
                
                for i in range(len(misclassified_X)):
                    true_class = class_names[misclassified_y_true[i]]
                    pred_class = class_names[misclassified_y_pred[i]]
                    confidence = misclassified_probs[i][misclassified_y_pred[i]]
                    true_class_prob = misclassified_probs[i][misclassified_y_true[i]]
                    
                    sample_info = {
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'true_class_probability': true_class_prob,
                        'confidence_gap': confidence - true_class_prob
                    }
                    
                    if feature_names:
                        for j, feat_name in enumerate(feature_names):
                            if j < misclassified_X.shape[1]:
                                sample_info[feat_name] = misclassified_X[i, j]
                    
                    misclassifications.append(sample_info)
    
    misclass_df = pd.DataFrame(misclassifications)

    if not misclass_df.empty:
        confusion_pairs = misclass_df.groupby(['true_class', 'predicted_class']).size().reset_index()
        confusion_pairs.columns = ['True Class', 'Predicted Class', 'Count']
        confusion_pairs = confusion_pairs.sort_values(by='Count', ascending=False)

        print()
        
        print("Top 10 misclassification patterns:")
        print(confusion_pairs.head(10))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='true_class', y='confidence', data=misclass_df)
        plt.title('Model Confidence in Misclassifications by True Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('misclassification_confidence.png') 
        plt.close()
        
        print()

        y_true_all = []
        y_pred_all = []

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true_all.extend(y_batch.cpu().numpy())
                y_pred_all.extend(predicted.cpu().numpy())

        # Generate and display classification report
        report_dict = classification_report(y_true_all, y_pred_all, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(3)

        print("\nFull Classification Report:")
        print(report_df)
        
        print()
    return misclass_df

def visualize_learned_features(model, test_loader, device, class_names, n_samples=500):
    """
    Visualize learned features from the penultimate layer
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        class_names: List of class names
        n_samples: Maximum number of samples to visualize
    """
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    handle = model.model[-3].register_forward_hook(get_activation('penultimate'))
    model.eval()
    features = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if len(features) >= n_samples:
                break
                
            data, target = data.to(device), target.to(device)
            _ = model(data)
            
            features.append(activations['penultimate'].cpu().numpy())
            targets.append(target.cpu().numpy())
    
    handle.remove()

    features = np.vstack(features)[:n_samples]
    targets = np.concatenate(targets)[:n_samples]

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    for i, cls in enumerate(class_names):
        mask = targets == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], alpha=0.7, label=cls)
    
    plt.legend()
    plt.title('PCA of Learned Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('pca_learned_features.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for i, cls in enumerate(class_names):
        mask = targets == i
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], alpha=0.7, label=cls)
    
    plt.legend()
    plt.title('t-SNE of Learned Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('tsne_learned_features.png')
    plt.close()
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    try:
        silhouette = silhouette_score(features, targets)
        calinski = calinski_harabasz_score(features, targets)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.4f}")
    except:
        print("Could not calculate clustering metrics (possibly too few samples)")