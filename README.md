# The complete workflow
Here we will list the complete workflow, from preprocessing, to model definition/training, to finally model evaluation on the given test data.
# The Proprocessing 

## 1. Split files listed in **data/metadata.csv** into 1.5 second chunks of wav

This whole process is done in the notebook `data/processed_wav_generator_train_and_val.ipynb`. The steps are as follows, for each wav file and it's corresponding labels in the **data/metadata.csv**:
- Read the wav file into a numpy array, and normalised the array's value between -0.5 to 0.5 (when doing data inspection we found that, for the same insect, depending on how close the insect is to the sound detector, the amplitude of the wav varies, so it makes sense to do this normalisation).
- Find a list of peaks (peaks correspond to when the insect makes a noise) in the array, take 1.5 seconds chunk around each peak, save each of those 1.5 seconds chunk into wav in the `data/big_data_processed_train_and_val` directory.
- If the wav file from the **data/metadata.csv** itself is smaller than 1.5 * 1.05 seconds, then process it using `preprocessing_func_2.process_small_wav` function and save the resulting wav in the `data/big_data_processed_train_and_val` directory, where `preprocessing_func_2.py` is under `audio_preprocessing` directory.
- Save the file names and class labels of those 1.5 seconds audio chunks to **data/big_data_processed_train_and_val.csv**.


## 2. Upsample less represented classes using data augmentation

This whole process is done in the notebook `data/big_data_processed_train_and_val.ipynb`. By checking **data/big_data_processed_train_and_val.csv**, one sees that after splitting audio files into 1.5 second chunks using peak detection, some of the classes only have 22 data points (a data point is a 1.5 second wav), while some other classes have more than 1500 data points. So it is important to upsample those less represeted class. Here we upsample those classes using data augmentations. The steps are as follows, for each class in the **data/big_data_processed_train_and_val.csv**:
- If the number of data points in the class is less than 1350 (this number is choosen somewhat randomly, but it's there to make sure most classes can have enough data points for training), then we generate new wavs using the data augmentation functions until we reach 1350 data points.
- We have 5 data augmentation functions defined in `data/big_data_processed_train_and_val.ipynb`, being `add_white_noise`, `shift_sound`, `stretch_sound`, `change_pitch` and `combined_sound`, where `combined_sound` just combines all the other 4 augmentation functions. We apply those augmentation functions to less representated classes to upsample.
- Save all the upsampled wavs in the `data/big_data_upsample_train_and_val` directory.
- Save the file names and class labels of those upsampled wavs to **data/big_argumentation_data_train_and_val.csv**.