import torch
import numpy as np
import time
from collections import Counter
from typing import List
import sys
import math 

sys.path.append('../audio_preprocessing')

import preprocessing_func_2

def most_common(my_list):
    data = Counter(my_list)
    return data.most_common(1)[0][0]
    

def get_device():
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    return device

def evaluation(model, gen, device=get_device()):
    model = model.to(device)
    model.eval()
    true_labels=[]
    predictions=[]
    
    for image, label in gen:
        pred = model(image.type(torch.cuda.FloatTensor).to(device).unsqueeze(0))
        predictions.append(pred.detach().cpu().numpy().argmax(axis=1).flatten()[0])  
        true_labels.append(label)
        
    accuracy = np.mean(np.array(true_labels)==np.array(predictions))
    accuracy = float(accuracy) * 100
    print(f"accuracy = {accuracy:.2f}%")
    print(f"predictions = {predictions}")
    final_prediction = most_common(predictions)
    
    return predictions, final_prediction

    
def evaluate_audio_classes(
    models, 
    paths:List[str], 
    labels:List[int], 
    mean, 
    std, 
    device=get_device(),
    image_preprocess_fn=preprocessing_func_2.resize_function(),
    mel_transform_fn=preprocessing_func_2.calculate_melsp,
):
    result_dict = {
        "paths": paths,
        "true_label": labels,
        "predicted_labels": [],
        "predicted_class_id": [],
    }
    for path, label in zip(paths, labels):
        predictions = []
        for model in models:
            non_normalised_generator = preprocessing_func_2.non_normalised_data_generator(
                paths=[path], 
                labels=[label],
                image_preprocess_fn=image_preprocess_fn,
                mel_transform_fn=mel_transform_fn,
            )
            normalised_generator = preprocessing_func_2.normalised_data_generator(
                non_normalised_generator, 
                mean, 
                std,
            )
            current_predictions, _ = evaluation(model, normalised_generator, device)
            predictions += current_predictions
        final_prediction = most_common(predictions)
        result_dict["predicted_labels"].append(predictions)
        result_dict["predicted_class_id"].append(final_prediction)
    return result_dict






