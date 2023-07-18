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
    wav_max_amplitude:float=0.5
):
    for (path, label) in zip(paths, labels):
        wav = load_wav(path=path)
        db_mel_spec = mel_transform_fn(wav)
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