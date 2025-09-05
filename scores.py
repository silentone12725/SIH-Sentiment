import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json
from datetime import datetime

def calculate_prf_scores(df: pd.DataFrame) -> dict:
    """
    Calculate Precision, Recall, and F1-score for sentiment analysis results
    
    Args:
        df: DataFrame with 'expected_sentiment' and 'predicted_sentiment' columns
        
    Returns:
        Dictionary containing detailed metrics
    """
    # Filter out rows with prediction errors
    valid_predictions = df[df['predicted_sentiment'] != 'Error'].copy()
    
    if len(valid_predictions) == 0:
        return {"error": "No valid predictions found"}
    
    # Get the expected and predicted sentiments
    y_true = valid_predictions['expected_sentiment']
    y_pred = valid_predictions['predicted_sentiment']
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=['Positive', 'Negative', 'Neutral', 'Mixed']
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Create detailed results
    results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1_score': float(overall_f1),
            'total_samples': len(valid_predictions)
        },
        'class_metrics': {},
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Class-wise metrics
    labels = ['Positive', 'Negative', 'Neutral', 'Mixed']
    for i, label in enumerate(labels):
        if i < len(precision):
            results['class_metrics'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
    
    return results

def main():
    # Load the results file
    results_file = input("Enter the path to your sentiment analysis results CSV file: ")
    
    try:
        df = pd.read_csv(results_file)
        
        # Calculate PRF scores
        metrics = calculate_prf_scores(df)
        
        # Save metrics to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = f"prf_scores_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print results
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        overall = metrics['overall_metrics']
        print(f"Overall Accuracy: {overall['accuracy']:.4f}")
        print(f"Overall Precision: {overall['precision']:.4f}")
        print(f"Overall Recall: {overall['recall']:.4f}")
        print(f"Overall F1-Score: {overall['f1_score']:.4f}")
        print(f"Total Samples: {overall['total_samples']}")
        
        print("\nClass-wise Metrics:")
        print("-" * 30)
        for class_name, metrics_dict in metrics['class_metrics'].items():
            print(f"{class_name}:")
            print(f"  Precision: {metrics_dict['precision']:.4f}")
            print(f"  Recall: {metrics_dict['recall']:.4f}")
            print(f"  F1-Score: {metrics_dict['f1_score']:.4f}")
            print(f"  Support: {metrics_dict['support']}")
            print()
        
        print(f"Detailed metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
