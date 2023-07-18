import torch
import pandas as pd
import preprocessing_func


class NormalisedDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        nomralised_gen_fn, 
        non_nomralised_gen_fn, 
        mean, 
        std, 
        df, 
        shuffle:bool, 
        mel_transform=preprocessing_func.mel_spec(), 
        resize_fun=preprocessing_func.resize_function(), 
        n_seconds: float=3, 
        keep_remainder_threshold: float = 0.25, 
        sample_rate:int=16000
    ):
        self.nomralised_gen_fn = nomralised_gen_fn
        self.non_nomralised_gen_fn = non_nomralised_gen_fn
        self.mean = mean
        self.std = std
        self.df = df
        self.shuffle = shuffle
        self.mel_transform = mel_transform
        self.resize_fun = resize_fun
        self.n_seconds = n_seconds
        self.keep_remainder_threshold = keep_remainder_threshold
        self.sample_rate = sample_rate

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        non_nomralised_gen = self.non_nomralised_gen_fn(
            paths=paths, 
            labels=labels, 
            mel_transform=self.mel_transform, 
            resize_fun=self.resize_fun, 
            n_seconds=self.n_seconds, 
            keep_remainder_threshold=self.keep_remainder_threshold, 
            sample_rate=self.sample_rate
        )
        nomralised_gen = self.nomralised_gen_fn(non_nomralised_gen, self.mean, self.std)
        return nomralised_gen


class PretraindModelsDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        pretrained_vision_models_data_gen_fn, 
        df, 
        shuffle:bool, 
        mel_transform=preprocessing_func.mel_spec(), 
        image_preprocess_fn=preprocessing_func.pretrained_vision_models_preprocess(),
        n_seconds: float=3, 
        keep_remainder_threshold: float = 0.25, 
        sample_rate:int=16000
    ):
        self.pretrained_vision_models_data_gen_fn = pretrained_vision_models_data_gen_fn
        self.df = df
        self.shuffle = shuffle
        self.mel_transform = mel_transform
        self.image_preprocess_fn = image_preprocess_fn
        self.n_seconds = n_seconds
        self.keep_remainder_threshold = keep_remainder_threshold
        self.sample_rate = sample_rate

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        pretrained_vision_models_data_gen = self.pretrained_vision_models_data_gen_fn(
            paths=paths,
            labels=labels,
            mel_transform=self.mel_transform, 
            image_preprocess_fn=self.image_preprocess_fn, 
            n_seconds=self.n_seconds, 
            keep_remainder_threshold=self.keep_remainder_threshold, 
            sample_rate=self.sample_rate
        )
        return pretrained_vision_models_data_gen

class PretraindModelsDataSetNew(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        pretrained_vision_models_data_gen_fn, 
        df, 
        shuffle:bool, 
        mel_transform_fn=preprocessing_func.calculate_melsp, 
        image_preprocess_fn=preprocessing_func.pretrained_vision_models_preprocess(),
        n_seconds: float=3, 
        keep_remainder_threshold: float = 0.25, 
        sample_rate:int=44100
    ):
        self.pretrained_vision_models_data_gen_fn = pretrained_vision_models_data_gen_fn
        self.df = df
        self.shuffle = shuffle
        self.mel_transform_fn = mel_transform_fn
        self.image_preprocess_fn = image_preprocess_fn
        self.n_seconds = n_seconds
        self.keep_remainder_threshold = keep_remainder_threshold
        self.sample_rate = sample_rate

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        pretrained_vision_models_data_gen = self.pretrained_vision_models_data_gen_fn(
            paths=paths,
            labels=labels,
            mel_transform_fn=self.mel_transform_fn, 
            image_preprocess_fn=self.image_preprocess_fn, 
            n_seconds=self.n_seconds, 
            keep_remainder_threshold=self.keep_remainder_threshold, 
            sample_rate=self.sample_rate
        )
        return pretrained_vision_models_data_gen    
    
class NonNormalisedTestModelsDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        non_normalised_data_generator_fn, 
        df, 
        shuffle:bool,
        mel_transform_fn=preprocessing_func.calculate_melsp,
        n_seconds:float=3, 
        keep_remainder_threshold:float = 0.25, 
        sample_rate:int=44100
    ):
        self.non_normalised_data_generator_fn = non_normalised_data_generator_fn
        self.mel_transform_fn=mel_transform_fn
        self.df = df
        self.shuffle = shuffle
        self.n_seconds = n_seconds
        self.keep_remainder_threshold = keep_remainder_threshold
        self.sample_rate = sample_rate

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        non_normalised_data_generator = self.non_normalised_data_generator_fn(
            paths=paths,
            labels=labels,
            mel_transform_fn=self.mel_transform_fn, 
            n_seconds=self.n_seconds, 
            keep_remainder_threshold=self.keep_remainder_threshold, 
            sample_rate=self.sample_rate
        )
        return non_normalised_data_generator
    
class NormalisedDataSetNew(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        non_normalised_data_generator_fn, 
        normalised_data_generator_fn,
        df, 
        shuffle:bool,
        mean,
        std,
        image_preprocess_fn,
        mel_transform_fn=preprocessing_func.calculate_melsp,
        n_seconds:float=3, 
        keep_remainder_threshold:float = 0.25, 
        sample_rate:int=44100
    ):
        self.non_normalised_data_generator_fn = non_normalised_data_generator_fn
        self.normalised_data_generator_fn = normalised_data_generator_fn
        self.mel_transform_fn=mel_transform_fn
        self.image_preprocess_fn=image_preprocess_fn
        self.df = df
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.n_seconds = n_seconds
        self.keep_remainder_threshold = keep_remainder_threshold
        self.sample_rate = sample_rate

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["path"].values), list(self.df["label"].values)
        non_normalised_data_generator = self.non_normalised_data_generator_fn(
            paths=paths,
            labels=labels,
            mel_transform_fn=self.mel_transform_fn, 
            image_preprocess_fn=self.image_preprocess_fn,
            n_seconds=self.n_seconds, 
            keep_remainder_threshold=self.keep_remainder_threshold, 
            sample_rate=self.sample_rate
        )
        normalised_data_generator = self.normalised_data_generator_fn(
            non_normalised_data_generator, 
            self.mean, 
            self.std
        )
        
        return normalised_data_generator