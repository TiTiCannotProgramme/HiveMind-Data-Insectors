import torch
import torchaudio
import torchaudio.functional
import torchaudio.transforms
import torchvision.transforms
from scipy import signal

import math
import numpy as np
import sklearn

import librosa
from typing import List

def load_wav(path:str) -> List[float]:
    '''
    returned wav is an array of floats, not a tensor of shape (1, #points)
    '''
    wav, sr = librosa.load(path, sr=44100)
    return wav

def normalise_wav(wav, wav_max_amplitude:float=0.5):
    '''
    wav is numpy array
    '''
    if wav.max() < 0:
        wav = wav - 0.5*(wav.max() + wav.min())
    max_value: float = max(wav.max(), abs(wav.min()))
    return wav_max_amplitude*(wav/max_value)

def find_wav_peaks(wav:List[float], distance_between_peaks:int=66150) -> List[int]|None:
    '''
    wav is array of floats
    '''
    if len(wav) <= distance_between_peaks * 1.05:
        return None
    
    height:float = wav.max()/4
    start_point:int = math.floor(distance_between_peaks/2)
    end_point:int = len(wav) - 1 - start_point
    peaks, _ = signal.find_peaks(wav[start_point:end_point], distance=distance_between_peaks, height=height)
    
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(wav, distance=distance_between_peaks, height=height)
        for i in range(len(peaks)):
            if peaks[i] < start_point:
                peaks[i] = start_point + 1
            if peaks[i] + math.floor(distance_between_peaks/2) > len(wav):
                peaks[i] = len(wav) - math.floor(distance_between_peaks/2) - 1
        return peaks
        
    return peaks + start_point

def find_wav_peaks_with_overlap(wav:List[float], distance_between_peaks:int=66150) -> List[int]|None:
    '''
    wav is array of floats
    '''
    if len(wav) <= distance_between_peaks * 1.05:
        return None
    
    height:float = wav.max()/4
    start_point:int = math.floor(distance_between_peaks/2)
    end_point:int = len(wav) - 1 - start_point
    distance_with_overlap:int = math.floor(distance_between_peaks * 0.55)
    peaks, _ = signal.find_peaks(wav[start_point:end_point], distance=distance_with_overlap, height=height)
    
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(wav, distance=distance_with_overlap, height=height)
        for i in range(len(peaks)):
            if peaks[i] < start_point:
                peaks[i] = start_point + 1
            if peaks[i] + math.floor(distance_between_peaks/2) >= len(wav):
                peaks[i] = len(wav) - math.floor(distance_between_peaks/2) - 1
        return peaks
        
    return peaks + start_point

def split_wav_by_peaks(wav:List[float], peaks:List[int], chunk_size:int=66150) -> List[List[float]]:
    results: List[List[float]] = []
    half_chunk_size:int = math.floor(chunk_size/2)
    for peak in peaks:
        results.append(wav[peak-half_chunk_size:peak+half_chunk_size])
    return results

def process_small_wav(wav, chunk_size:int=66150):
    if len(wav) < chunk_size:
        wav = np.array(wav)
        wav = np.concatenate([wav, np.zeros(chunk_size-len(wav))])
        return wav
    if len(wav) > chunk_size:
        return wav[:chunk_size]
    return wav

def resize_function(output_shape=(128,512)):
    return torchvision.transforms.Resize(size=output_shape, antialias=False)

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

def to_reshaped_tensor(image):
    image = torch.tensor(image)
    return image.unsqueeze(0)

def non_normalised_data_generator(
    paths: List[str], 
    labels: List[int], 
    image_preprocess_fn=resize_function(),
    mel_transform_fn=calculate_melsp,
    chunk_size:int=66150,
    wav_max_amplitude:float=0.5
):
    for (path, label) in zip(paths, labels):
        wav = load_wav(path=path)
        wav = normalise_wav(wav=wav, wav_max_amplitude=wav_max_amplitude)
        peaks = find_wav_peaks(wav=wav, distance_between_peaks=chunk_size)
        if peaks is None:
            wav = process_small_wav(wav=wav, chunk_size=chunk_size)
            db_mel_spec = mel_transform_fn(wav)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            if len(wav) < 1024:
                print("wav")
                print(path)
            yield image_preprocess_fn(db_mel_spec), label
            continue
        for chunks in split_wav_by_peaks(wav=wav, peaks=peaks, chunk_size=chunk_size):
            if len(chunks) < 1024:
                print("chunk")
                print(len(chunks))
                print(path)
            db_mel_spec = mel_transform_fn(chunks)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            yield image_preprocess_fn(db_mel_spec), label

def non_normalised_data_generator_new(
    paths: List[str], 
    labels: List[int], 
    image_preprocess_fn=resize_function(),
    mel_transform_fn=calculate_melsp,
    chunk_size:int=66150,
    wav_max_amplitude:float=0.5
):
    for (path, label) in zip(paths, labels):
        wav = load_wav(path=path)
        normalised_wav = normalise_wav(wav=wav, wav_max_amplitude=wav_max_amplitude)
        peaks = find_wav_peaks(wav=normalised_wav, distance_between_peaks=chunk_size)
        if peaks is None:
            wav = process_small_wav(wav=wav, chunk_size=chunk_size)
            db_mel_spec = mel_transform_fn(wav)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            if len(wav) < 1024:
                print("wav")
                print(path)
            yield image_preprocess_fn(db_mel_spec), label
            continue
        for chunks in split_wav_by_peaks(wav=wav, peaks=peaks, chunk_size=chunk_size):
            if len(chunks) < 1024:
                print("chunk")
                print(len(chunks))
                print(path)
            db_mel_spec = mel_transform_fn(chunks)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            yield image_preprocess_fn(db_mel_spec), label

def non_normalised_data_generator_one_sec(
    paths: List[str], 
    labels: List[int], 
    image_preprocess_fn=resize_function(),
    mel_transform_fn=calculate_melsp,
    chunk_size:int=44100,
    wav_max_amplitude:float=0.5
):
    for (path, label) in zip(paths, labels):
        wav = load_wav(path=path)
        wav = normalise_wav(wav=wav, wav_max_amplitude=wav_max_amplitude)
        peaks = find_wav_peaks_with_overlap(wav=wav, distance_between_peaks=chunk_size)
        if peaks is None:
            wav = process_small_wav(wav=wav, chunk_size=chunk_size)
            db_mel_spec = mel_transform_fn(wav)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            if len(wav) < 1024:
                print("wav")
                print(path)
            yield image_preprocess_fn(db_mel_spec), label
            continue
        for chunks in split_wav_by_peaks(wav=wav, peaks=peaks, chunk_size=chunk_size):
            if len(chunks) < 1024:
                print("chunk")
                print(len(chunks))
                print(path)
            db_mel_spec = mel_transform_fn(chunks)
            db_mel_spec = to_reshaped_tensor(db_mel_spec)
            yield image_preprocess_fn(db_mel_spec), label

def normalise_image(image, mean, std, eps=1e-6):
    """
    Return a normalised image that has mean 0 and standard deviation 1.
    
    Args:
        image: A pytorch tensor
        mean: Precalculated mean of all images
        std: Precalculated standard deviation of all images

    Returns:
        The normalised image (pytorch tensor)
    """
    return (image - mean) / (std + eps)


def normalised_data_generator(generator, mean, std):
    """
    Return a generator with normalised images.
    
    Args:
        generator: A generator outputing (non-normalised image, label)
        mean: Precalculated mean of all non-normalised images
        std: Precalculated standard deviation of all non-normalised images

    Returns:
        generator outputing (normalised image with mean 0 and std 1, label)
    """
    for image, label in generator:
        yield normalise_image(image, mean, std), label

def get_stats_and_class_weights_of_non_normalised_data_gen(data_gen, image_width_height):
    total_sum = torch.Tensor([0.0])
    total_squared_sum = torch.Tensor([0.0])
    num_of_images: int = 0
    labels = []
    
    for image, label in data_gen:
        labels.append(label)
        num_of_images += 1
        total_sum += image.sum()
        total_squared_sum += (image ** 2).sum()
    
    num_of_data = num_of_images * image_width_height[0] * image_width_height[1]
    
    mean = total_sum / num_of_data
    variance = total_squared_sum / num_of_data - mean ** 2
    std = torch.sqrt(variance)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return mean, std, class_weights


def calculate_num_of_labels(gen):
    result_dict = {}
    for _, label in gen:
        if label not in result_dict:
            result_dict[label] = 1
        else:
            result_dict[label] += 1
    return result_dict