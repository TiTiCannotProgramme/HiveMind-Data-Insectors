import torch
import pandas as pd
import preprocessing_func_3

def normalised_image_generator(paths, labels):
    for (path, label) in zip(paths, labels):
        image = torch.load(path)
        yield image, label
    
class NormalisedImageDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        df, 
        shuffle:bool,
        normalised_image_gen=normalised_image_generator
    ):
        self.normalised_image_gen = normalised_image_gen
        self.df = df
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
        paths, labels = list(self.df["file_path"].values), list(self.df["label"].values)
        normalised_image_gen = self.normalised_image_gen(
            paths=paths,
            labels=labels,
        )
        
        return normalised_image_gen