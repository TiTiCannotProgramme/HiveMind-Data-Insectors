import torch
import pandas as pd
import preprocessing_func_2

    
class NormalisedDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        non_normalised_data_generator_fn, 
        normalised_data_generator_fn,
        df, 
        shuffle:bool,
        mean,
        std,
        image_preprocess_fn=preprocessing_func_2.resize_function(),
        mel_transform_fn=preprocessing_func_2.calculate_melsp,
        chunk_size:int=66150,
        wav_max_amplitude:float=0.5,
    ):
        self.non_normalised_data_generator_fn = non_normalised_data_generator_fn
        self.normalised_data_generator_fn = normalised_data_generator_fn
        self.mel_transform_fn=mel_transform_fn
        self.image_preprocess_fn=image_preprocess_fn
        self.df = df
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.chunk_size = chunk_size
        self.wav_max_amplitude = wav_max_amplitude

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        non_normalised_data_generator = self.non_normalised_data_generator_fn(
            paths=paths,
            labels=labels,
            mel_transform_fn=self.mel_transform_fn, 
            image_preprocess_fn=self.image_preprocess_fn,
            chunk_size=self.chunk_size,
            wav_max_amplitude=self.wav_max_amplitude,
        )
        normalised_data_generator = self.normalised_data_generator_fn(
            non_normalised_data_generator, 
            self.mean, 
            self.std
        )
        
        return normalised_data_generator