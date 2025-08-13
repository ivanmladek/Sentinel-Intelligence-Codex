import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import time

def load_data():
    dataset = load_dataset("ccdv/patent-classification")
    texts = dataset['train']['text'] + dataset['test']['text']
    labels = dataset['train']['labels'] + dataset['test']['labels']
    return texts, labels

def run_vllm_baseline(texts, labels):
    # Initialize vLLM with a small model
    model_name = "NousResearch/Hermes-2-Pro-Llama-3-8B"
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
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
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
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    return accuracy, processing_time

def main():
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples")
    
    accuracy, processing_time = run_vllm_baseline(texts, labels)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Samples Processed: {len(texts)}")

if __name__ == "__main__":
    main()