import os
import argparse
import json
import csv
import glob
from openai import OpenAI
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenAI model on classification datasets")
    parser.add_argument("--prompt_file", type=str, help="Path to file containing the prompt")
    parser.add_argument("--prompt", type=str, help="Prompt to send to OpenAI (alternative to prompt_file)")
    parser.add_argument("--dataset", type=str, help="Specific dataset to use (optional)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CSV datasets")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--output_predictions", action="store_true", help="Output predictions to CSV file")
    return parser.parse_args()

def load_prompt(file_path):
    """Load prompt from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_dataset(dataset_path):
    """Load dataset from CSV with 'text' and 'label' columns"""
    texts = []
    labels = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                texts.append(row[0].strip('"'))
                labels.append(row[1].strip('"'))
    
    return texts, labels

def get_available_datasets(data_dir):
    """Get list of available CSV datasets"""
    return glob.glob(os.path.join(data_dir, "*.csv"))

def select_dataset(data_dir, dataset=None):
    """Let user select a dataset if not specified"""
    available_datasets = get_available_datasets(data_dir)
    
    if not available_datasets:
        raise FileNotFoundError(f"No CSV datasets found in {data_dir}")
    
    if dataset:
        dataset_path = os.path.join(data_dir, dataset)
        if dataset_path not in available_datasets and dataset not in available_datasets:
            raise FileNotFoundError(f"Dataset {dataset} not found in {data_dir}")
        return dataset_path if dataset_path in available_datasets else dataset
    
    print("Available datasets:")
    for i, dataset_path in enumerate(available_datasets):
        print(f"{i+1}. {os.path.basename(dataset_path)}")
    
    selection = int(input("Select dataset (number): ")) - 1
    return available_datasets[selection]

def get_openai_predictions(client, texts, prompt, model):
    """Get predictions from OpenAI for each text"""
    all_predictions = []
    
    for text in texts:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # Validate response format
            if "labels" not in result or "scores" not in result:
                print(f"Warning: Invalid response format for text: {text[:50]}...")
                continue
            
            all_predictions.append(result)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response for text: {text[:50]}...")
            continue
    
    return all_predictions

def get_highest_score_labels(predictions):
    """Extract predicted labels with highest scores"""
    predicted_labels = []
    
    for pred in predictions:
        max_score_idx = np.argmax(pred["scores"])
        if pred["scores"][max_score_idx] == 1.0:
            predicted_labels.append(pred["labels"][max_score_idx])
        else:
            print(pred["scores"])
            predicted_labels.append("n/a")


    
    return predicted_labels

def calculate_metrics(true_labels, predicted_labels):
    """Calculate precision, recall, and F1 for each label"""
    unique_labels = sorted(set(true_labels))
    results = {}
    
    for avg in ['micro', 'macro', 'weighted']:
        results[f'{avg}_precision'] = precision_score(true_labels, predicted_labels, average=avg, zero_division=0)
        results[f'{avg}_recall'] = recall_score(true_labels, predicted_labels, average=avg, zero_division=0)
        results[f'{avg}_f1'] = f1_score(true_labels, predicted_labels, average=avg, zero_division=0)
    
    # Per-class metrics
    results['per_class'] = {}
    for label in unique_labels:
        true_binary = [1 if l == label else 0 for l in true_labels]
        pred_binary = [1 if l == label else 0 for l in predicted_labels]
        
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        
        results['per_class'][label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': true_binary.count(1)
        }
    
    return results

def save_predictions_to_csv(texts, true_labels, predicted_labels, dataset_name):
    """Save text, true labels, and predicted labels to a CSV file"""
    output_file = f"predictions_{os.path.basename(dataset_name).replace('.csv', '')}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "true_label", "predicted_label"])
        
        for text, true_label, pred_label in zip(texts, true_labels, predicted_labels):
            writer.writerow([text, true_label, pred_label])
    
    return output_file

def main():
    args = parse_args()
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Get prompt from file or command line
    if args.prompt_file and os.path.exists(args.prompt_file):
        prompt = load_prompt(args.prompt_file)
        print(f"Loaded prompt from {args.prompt_file}")
    elif args.prompt:
        prompt = args.prompt
    else:
        raise ValueError("Either --prompt or --prompt_file must be provided")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Select dataset
    dataset_path = select_dataset(args.data_dir, args.dataset)
    print(f"Using dataset: {dataset_path}")
    
    # Load dataset
    texts, true_labels = load_dataset(dataset_path)
    print(f"Loaded {len(texts)} examples")
    
    # Get predictions from OpenAI
    print("Getting predictions from OpenAI...")
    predictions = get_openai_predictions(client, texts, prompt, args.model)
    
    # Extract predicted labels with highest scores
    predicted_labels = get_highest_score_labels(predictions)
    
    # Save predictions to CSV file
    predictions_file = save_predictions_to_csv(texts, true_labels, predicted_labels, dataset_path)
    print(f"Predictions saved to {predictions_file}")
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Display results
    print("\n===== RESULTS =====")
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Model: {args.model}")
    print(f"Examples: {len(texts)}")
    
    print("\nAggregate Metrics:")
    for metric, value in {k: v for k, v in metrics.items() if k != 'per_class'}.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nPer-Class Metrics:")
    class_metrics_df = pd.DataFrame.from_dict(
        {label: metrics['per_class'][label] for label in metrics['per_class']},
        orient='index'
    )
    print(class_metrics_df.sort_values('support', ascending=False))
    
    # Save results to file
    results_file = f"results_{os.path.basename(dataset_path).replace('.csv', '')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "dataset": os.path.basename(dataset_path),
            "model": args.model,
            "examples": len(texts),
            "metrics": metrics,
            "prompt": prompt,
            "predictions_file": predictions_file
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()