import os
import time
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from google.cloud import storage

def load_data():
    dataset = load_dataset("ccdv/patent-classification")
    texts = dataset['train']['text'] + dataset['test']['text']
    labels = dataset['train']['labels'] + dataset['test']['labels']
    return texts, labels

def run_vllm_baseline(texts, labels):
    # Start timing the entire process
    start_time = time.time()
    
    # Initialize vLLM with a smaller model suitable for T4
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, gpu_memory_utilization=0.5, max_model_len=1024)
    
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    # Prepare prompts for classification
    class_names = ["Human Necessities", "Chemistry; Metallurgy", "Textiles; Paper", 
                  "Fixed Constructions", "Mechanical Engineering", "Physics", 
                  "Electricity", "General tagging", "Unknown"]
    
    prompts = []
    for text in texts:  # Process all samples
        prompt = f"Classify this patent abstract into one of the following categories: {', '.join(class_names)}\n\nAbstract: {text[:512]}\n\nCategory:"
        prompts.append(prompt)
    
    # Run inference
    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    predictions = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        # Simple matching to class names
        pred_class = 8  # Default to "Unknown"
        for i, class_name in enumerate(class_names):
            if class_name.lower() in generated_text.lower():
                pred_class = i
                break
        predictions.append(pred_class)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # End timing the entire process
    end_time = time.time()
    total_time = end_time - start_time
    
    return accuracy, total_time, predictions

def save_results_to_gcs(pod_index, texts, labels, predictions, accuracy, total_time):
    """Save results to GCS for aggregation"""
    bucket_name = os.environ.get("BUCKET_NAME", "gdrive-410709-vllm-13b-demo-bucket")
    
    # Create results data
    results_data = {
        "pod_index": pod_index,
        "num_samples": len(texts),
        "accuracy": accuracy,
        "processing_time": total_time,
        "labels": labels,
        "predictions": predictions
    }
    
    # Save results to GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob_name = f"text_classification_results/pod_{pod_index}_results.json"
        blob = bucket.blob(blob_name)
        
        # Convert results to JSON
        results_json = json.dumps(results_data)
        blob.upload_from_string(results_json, content_type='application/json')
        print(f"Results saved to GCS: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"Error saving results to GCS: {e}")

def aggregate_results():
    """Aggregate results from all pods"""
    bucket_name = os.environ.get("BUCKET_NAME", "gdrive-410709-vllm-13b-demo-bucket")
    num_pods = int(os.environ.get("NUM_PODS", 1))
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Collect results from all pods
        all_predictions = []
        all_labels = []
        total_processing_time = 0
        
        for pod_index in range(num_pods):
            blob_name = f"text_classification_results/pod_{pod_index}_results.json"
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                # Download and parse results
                results_json = blob.download_as_string()
                results_data = json.loads(results_json)
                
                # Add to aggregated results
                all_predictions.extend(results_data.get("predictions", []))
                all_labels.extend(results_data.get("labels", []))
                total_processing_time += results_data.get("processing_time", 0)
                print(f"Loaded results from pod {pod_index}")
            else:
                print(f"Results not found for pod {pod_index}")
        
        # Calculate overall accuracy
        if all_labels and all_predictions:
            overall_accuracy = accuracy_score(all_labels, all_predictions)
            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            print(f"Total Processing Time: {total_processing_time:.2f} seconds")
            print(f"Total Samples Processed: {len(all_labels)}")
            
            # Save aggregated results
            aggregated_data = {
                "overall_accuracy": overall_accuracy,
                "total_processing_time": total_processing_time,
                "total_samples": len(all_labels)
            }
            aggregated_blob = bucket.blob("text_classification_results/aggregated_results.json")
            aggregated_blob.upload_from_string(json.dumps(aggregated_data), content_type='application/json')
            print("Aggregated results saved to GCS")
        else:
            print("No results found to aggregate")
            
    except Exception as e:
        print(f"Error aggregating results: {e}")

def main():
    print("Loading data...")
    texts, labels = load_data()
    
    # Distribute the work among the pods
    pod_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
    num_pods = int(os.environ.get("NUM_PODS", 1))
    files_per_pod = len(texts) // num_pods
    start_index = pod_index * files_per_pod
    end_index = start_index + files_per_pod
    
    if pod_index == num_pods - 1:
        end_index = len(texts)
    
    texts_to_process = texts[start_index:end_index]
    labels_to_process = labels[start_index:end_index]
    
    print(f"Processing samples {start_index} to {end_index} (total {len(texts_to_process)} samples)")
    
    accuracy, total_time, predictions = run_vllm_baseline(texts_to_process, labels_to_process)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Wall Clock Time: {total_time:.2f} seconds")
    print(f"Samples Processed: {len(texts_to_process)}")
    
    # Save results to GCS for aggregation
    save_results_to_gcs(pod_index, texts_to_process, labels_to_process, predictions, accuracy, total_time)
    
    # If this is the last pod, aggregate results
    if pod_index == num_pods - 1:
        print("Aggregating results from all pods...")
        aggregate_results()

if __name__ == "__main__":
    main()