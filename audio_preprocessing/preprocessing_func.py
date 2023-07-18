import torch
import torchaudio
import torchaudio.functional
import torchaudio.transforms
import torchvision.transforms

import math
import numpy as np
import sklearn

import librosa
from typing import List

def change_wav_sample_rate(path:str, sample_rate:int=16000):
    """
    Change the sample rate of the wav file. By default it down sample the sample rate to 16000, so that the later
    processing process steps are hopefully faster.

    Args:
        path (str): The path to the wav file.
        sample_rate (int): how many points to sample per 1 second of audio

    Returns:
        (Torch tensor, sample_rate): The torch tensor is the resampled waveform in shape 
        (number of channels, amplitudes), where in our case number of channels = 1. So for a 2 second wav 
        file resampled to 16000, the output will be ((1, 32000), 16000).
    """
#     effects = [
#       ["rate", f'{sample_rate}'],
#     ]
#     return torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    wav, sr = librosa.load(path, sr=44100)
    resampled_wav = torch.tensor(librosa.resample(wav, orig_sr=sr, target_sr=sample_rate))
    reshaped_tensor_wav = torch.reshape(resampled_wav, (1,-1))
    return (reshaped_tensor_wav, sample_rate)


def split_wav_to_n_seconds_chunks(waveform, n_seconds: float=3, keep_remainder_threshold: float = 0.25, sample_rate: int=16000):
    """
    Split a waveform in the shape (number of channels, amplitudes) into a list of tensors in the shape 
    (number of channels, n_seconds*sample_rate). 
    
    If amplitudes < n_seconds*sample_rate, then the waveform is padded with zeros.
    For example, if input is a tensor Tensor([[1, 2, 3]]) and n_seconds*sample_rate = 5, then the output will be
    [Tensor([[1, 2, 3, 0, 0]])].
    
    If amplitudes > n_seconds*sample_rate, depends on the remainder is greater than keep_remainder_threshold or not, 
    there are two different logics for splitting. Where the remainder is the none integer part of 
    waveform[1]/(n_seconds*sample_rate). For example, the remainder of 5.6 is 0.6.
    I did some maths for the two logics, so I'm not gonna explain it in detail here, you guys can ask me if you have
    questions.

    Args:
        waveform (pytorch tensor): waveform is a tensor with shape (number of channels, amplitudes).
        sample_rate (int): how many points to sample per 1 second of audio
        keep_remainder_threshold(float): will keep remainder if remainder is greater than this threshold. Else
        discard reminder.

    Returns:
        A list of pytorch tensors, where each of the tensor will be of the shape 
        (number of channels, n_seconds*sample_rate).
    """
    chunk_length: int = math.floor(sample_rate * n_seconds)
    waveform_length: int = waveform.shape[1]
    
    #pad the pytorch tensor with zeros
    if waveform_length < chunk_length:
        return [torch.nn.functional.pad(waveform, pad=(0, chunk_length - waveform_length, 0, 0))]
    
    num_of_chunks: int = math.ceil(waveform_length / chunk_length)
    reminder: float = waveform_length / chunk_length - num_of_chunks + 1
    
    # num_of_chunks == 1 if and only if waveform_length == chunk_length
    if num_of_chunks == 1:
        return [waveform[:, 0: chunk_length]]
    
    if reminder < keep_remainder_threshold:
        num_of_chunks -= 1
        return [waveform[:, i * chunk_length : (i+1) * chunk_length] for i in range(num_of_chunks)]
    else:
        gap: int = math.ceil(1/(num_of_chunks-1) * (num_of_chunks * chunk_length - waveform_length))
        result = [waveform[:, 0: chunk_length]]
        for i in range(1, num_of_chunks):
            result.append(waveform[:, i*(chunk_length-gap): i*(chunk_length-gap) + chunk_length])
        return result

def mel_spec(n_mels:int=128, n_fft:int=1024, sample_rate:int=16000, f_min:float=300.0, hop_length:int=128):
    """
    Obtain a melspectrogram object, see the following link for more detail on this object:
    https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
    
    These preset parameters is what I think work the best, but you are free to experiment.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min = f_min,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        n_mels=n_mels,
        mel_scale="htk",
    )

def waveform_to_db_scale_mel_spec(waveform, mel_transform):
    """
    Transform a waveform of shape (number of channels, amplitudes) into a melspectrogram with db scale of the shape
    (number of channels, n_mels, times), where n_mels is an argument in mel_transform. I don't know how times works
    though, but it doesn't matter cos we will reshape this output later in the pipline anyways.
    
    Args:
        waveform (pytorch tensor): waveform is a tensor with shape (number of channels, amplitudes).
        mel_transform: A torchaudio.transforms.MelSpectrogram object that can transform waveform into melsepctrogram.

    Returns:
        A pytorch tensor of shape (number of channels, n_mels, times).
    
    """
    mel_spec = mel_transform(waveform)
    return torchaudio.functional.amplitude_to_DB(mel_spec, 10, 1e-10, np.log10(max(mel_spec.max(), 1e-10)))


def resize_function(output_shape=(64,64)):
    """
    Obtain a torchvision.transforms.Resize object, see the following link for more detail on this object:
    https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html
    
    These preset parameters is what I think work the best, but you are free to experiment.
    """
    return torchvision.transforms.Resize(size=output_shape, antialias=False)

def resize_image(image, resize_fun):
    """
    Resize the image to the desired output shape using the resize_fun. The output shape is defined in resize_fun.
    For example, if the output_shape of the resize_fun is (64, 64), and the image has the shape (1, 128, 128), then
    the output will be a tensor of shape (1, 64, 64).
    
    Args:
        image (pytorch tensor): image of shape (number of channels, height, width).
        resize_fun: A torchvision.transforms.Resize object that can transform image into desired shape.

    Returns:
        A pytorch tensor of shape (number of channels, reshaped height, reshaped width).
    """
    return resize_fun(image)


def non_normalised_data_generator(
    paths: List[str], labels: List[int], mel_transform=mel_spec(), resize_fun=resize_function(), n_seconds: float=3, keep_remainder_threshold: float = 0.25, sample_rate:int=16000
):
    """
    This is a data pipline for generating un-normalised mel spec, label pairs using the functions in this file.
    
    Args:
        paths (List[str]): A list of str path pointing to the wav files.
        labels (List[int]): The labels of each wav files.
        mel_transform: A torchaudio.transforms.MelSpectrogram object.
        resize_fun: A torchvision.transforms.Resize object
        
        n_seconds, keep_remainder_threshold, sample_rate are arguments used in functions change_wav_sample_rate and split_wav_to_n_seconds_chunks.

    Returns:
        A generator generating (resized_image, label) pair.
    """
    for (path, label) in zip(paths, labels):
        wav, _ = change_wav_sample_rate(path=path, sample_rate=sample_rate)
        for chunked_wav in split_wav_to_n_seconds_chunks(waveform=wav, n_seconds=n_seconds, keep_remainder_threshold=keep_remainder_threshold):
            db_mel_spec = waveform_to_db_scale_mel_spec(chunked_wav, mel_transform)
            resized_image = resize_image(db_mel_spec, resize_fun)
            yield(resized_image, label)

            
def load_audio(path:str, sample_rate:int=44100):
    wav, sr = librosa.load(path, sr=sample_rate)
    wav = np.reshape(wav, (1,-1))
    tensor_wav = torch.tensor(wav)
    return tensor_wav, sample_rate

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

def non_normalised_data_generator_new(
    paths: List[str], labels: List[int], 
    image_preprocess_fn,
    mel_transform_fn=calculate_melsp,
    n_seconds: float=3, 
    keep_remainder_threshold:float=0.25, 
    sample_rate:int=44100
):
    for (path, label) in zip(paths, labels):
        wav, _ = load_audio(path=path, sample_rate=sample_rate)
        for chunked_wav in split_wav_to_n_seconds_chunks(
            waveform=wav, 
            n_seconds=n_seconds, 
            keep_remainder_threshold=keep_remainder_threshold, 
            sample_rate=sample_rate
        ):
            db_mel_spec = calculate_melsp(chunked_wav.numpy())
            yield image_preprocess_fn(torch.tensor(db_mel_spec)), label



def get_stats_of_non_normalised_data_gen(data_gen, image_width_height):
    """
    Calculate the mean and std of all images in the data_gen
    
    Args:
        data_gen(Generator): A generator yielding data of the format (image, label). Should be non_normalised_data_generator.
        image_width_height(Tuple): The width and height of the image. If the image is of shape (1, 64, 64), this parameter should
        be (64, 64).

    Returns:
        The mean and standard deviation of all the images. Both values are pytorch tensor objects.
    """
    total_sum = torch.Tensor([0.0])
    total_squared_sum = torch.Tensor([0.0])
    num_of_images: int = 0
    
    for image, _ in data_gen:
        num_of_images += 1
        total_sum += image.sum()
        total_squared_sum += (image ** 2).sum()
    
    num_of_data = num_of_images * image_width_height[0] * image_width_height[1]
    
    mean = total_sum / num_of_data
    variance = total_squared_sum / num_of_data - mean ** 2
    std = torch.sqrt(variance)
    return mean, std


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


def get_normalised_image_generator(generator, mean, std):
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

def pretrained_vision_models_preprocess():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(64,256), antialias=False),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
        
        
def pretrained_vision_models_data_generator(
    paths: List[str], 
    labels: List[int], 
    mel_transform=mel_spec(), 
    image_preprocess_fn=pretrained_vision_models_preprocess(), 
    n_seconds: float=3, 
    keep_remainder_threshold: float = 0.25, 
    sample_rate:int=16000
):
    for (path, label) in zip(paths, labels):
        wav, _ = change_wav_sample_rate(path=path, sample_rate=sample_rate)
        for chunked_wav in split_wav_to_n_seconds_chunks(waveform=wav, n_seconds=n_seconds, keep_remainder_threshold=keep_remainder_threshold):
            db_mel_spec = waveform_to_db_scale_mel_spec(chunked_wav, mel_transform)
            yield image_preprocess_fn(db_mel_spec), label

            
def pretrained_vision_models_data_generator_new(
    paths: List[str], 
    labels: List[int], 
    mel_transform_fn=calculate_melsp, 
    image_preprocess_fn=pretrained_vision_models_preprocess(), 
    n_seconds: float=3, 
    keep_remainder_threshold: float = 0.25, 
    sample_rate:int=44100
):
    for (path, label) in zip(paths, labels):
        wav, _ = load_audio(path=path, sample_rate=sample_rate)
        for chunked_wav in split_wav_to_n_seconds_chunks(
            waveform=wav, 
            n_seconds=n_seconds, 
            keep_remainder_threshold=keep_remainder_threshold,
            sample_rate=sample_rate
        ):
            db_mel_spec = torch.tensor(calculate_melsp(chunked_wav.numpy()))
            yield image_preprocess_fn(db_mel_spec), label


def get_class_weights_from(data_gen):
    """
    Get the class weights from get_stats_of_non_normalised_data_gen, the result returned by this function can be used directly
    in the loss function's weight parameter
    """
    labels = []
    for _, label in data_gen:
        labels.append(label)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights,dtype=torch.float)


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