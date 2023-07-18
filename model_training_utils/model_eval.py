import torch
import numpy as np
import time
from collections import Counter
from typing import List
import sys
import math 
import librosa

sys.path.append('../audio_preprocessing')

import preprocessing_func_2

def most_common(my_list):
    data = Counter(my_list)
    return data.most_common(1)[0][0]
    
def resume_model(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])

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

def evaluate_wav_list(
    models, 
    wav_list, 
    mean, 
    std, 
    device=get_device(), 
    image_preprocess_fn=preprocessing_func_2.resize_function(),
    mel_transform_fn=preprocessing_func_2.calculate_melsp,
):
    for model in models:
        model = model.to(device)
        model.eval()
    predictions=[]
    
    for wav in wav_list:
        image = mel_transform_fn(wav)
        image = preprocessing_func_2.to_reshaped_tensor(image)
        image = image_preprocess_fn(image)
        image = preprocessing_func_2.normalise_image(image, mean, std)
        for model in models:
            pred = model(image.to(device, dtype=torch.float).unsqueeze(0))
            predictions.append(pred.detach().cpu().numpy().argmax(axis=1).flatten()[0])  
        
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

def evaluate_audio_classes_model_most_common(
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
            predictions.append(most_common(current_predictions))
        final_prediction = most_common(predictions)
        result_dict["predicted_labels"].append(predictions)
        result_dict["predicted_class_id"].append(final_prediction)
    return result_dict


def evaluate_audio_classes_one_sec(
    models, 
    paths:List[str], 
    labels:List[int], 
    mean, 
    std, 
    device=get_device(),
    image_preprocess_fn=preprocessing_func_2.resize_function(output_shape=(128, 345)),
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
            non_normalised_generator = preprocessing_func_2.non_normalised_data_generator_one_sec(
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


def evaluate_audio_classes_new(
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
            non_normalised_generator = preprocessing_func_2.non_normalised_data_generator_new(
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


def add_white_noise(x):
    rate = np.random.uniform(low=0.01, high=0.02)
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x):
    rate = np.random.randint(low=2, high=math.floor(len(x)/10000))
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x):
    rate = np.random.uniform(low=0.8, high=1.2)
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate=rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")
    
def combined_sound(x):
    return add_white_noise(shift_sound(stretch_sound(x)))

def evaluate_audio_classes_with_augmentations(
    models, 
    paths:List[str], 
    labels:List[int], 
    mean, 
    std, 
    device=get_device(),
    augmentations=[add_white_noise, shift_sound, stretch_sound, combined_sound],
    image_preprocess_fn=preprocessing_func_2.resize_function(),
    mel_transform_fn=preprocessing_func_2.calculate_melsp,
    chunk_size:int=66150,
    wav_max_amplitude:float=0.5,
):
    result_dict = {
        "paths": paths,
        "true_label": labels,
        "predicted_labels": [],
        "predicted_class_id": [],
    }
    for path, label in zip(paths, labels):
        predictions = []
        wav = preprocessing_func_2.load_wav(path=path)
        wav = preprocessing_func_2.normalise_wav(wav=wav, wav_max_amplitude=wav_max_amplitude)
        peaks = preprocessing_func_2.find_wav_peaks(wav=wav, distance_between_peaks=chunk_size)
        if peaks is None:
            wav = preprocessing_func_2.process_small_wav(wav=wav, chunk_size=chunk_size)
            wav_list = [wav]
        else:
            wav_list = preprocessing_func_2.split_wav_by_peaks(wav=wav, peaks=peaks, chunk_size=chunk_size)
        
        # predict non-augmented wav
        current_predictions, _ = evaluate_wav_list(models, wav_list, mean, std)
        predictions += current_predictions
        # predict augmented wavs
        for augmentation in augmentations:
            augmented_wav_list = []
            for _ in range(5):
                for wav in wav_list:
                    augmented_wav_list.append(augmentation(wav))
            current_predictions, _ = evaluate_wav_list(models, augmented_wav_list, mean, std)
            predictions += current_predictions
        
        true_labels = [label]*len(predictions)
        accuracy = np.mean(np.array(true_labels)==np.array(predictions))
        accuracy = float(accuracy) * 100
        print(f"accuracy = {accuracy:.2f}%")
        print(f"predictions = {predictions}")
        
        final_prediction = most_common(predictions)
        result_dict["predicted_labels"].append(predictions)
        result_dict["predicted_class_id"].append(final_prediction)
    return result_dict







