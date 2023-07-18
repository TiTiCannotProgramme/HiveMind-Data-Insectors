# The complete workflow
Here we will list the complete workflow, from preprocessing, to model definition/training, to finally model evaluation on the given test data.
# The Proprocessing 

## 1. Split files listed in **metadata.csv** into 1.5 second chunks of wav

This whole process is done in the notebook `data/processed_wav_generator_train_and_val.ipynb`. The steps are as follows, for each wav file and it's corresponding labels in the **metadata.csv**:
- Read the wav file into a numpy array, and normalised the array's value between -0.5 to 0.5 (when doing data inspection we found that, for the same insect, depending on how close the insect is to the sound detector, the amplitude of the wav varies, so it makes sense to do this normalisation).
- Find a list of peaks (peaks correspond to when the insect makes a noise) in the array, take 1.5 seconds chunk around each peak, save each of those 1.5 seconds chunk into wav in the `data/big_data_processed_train_and_val` directory.
- If the wav file from the **metadata.csv** itself is smaller than 1.5 * 1.05 seconds, then process it using `preprocessing_func_2.process_small_wav` function and save the resulting wav in the `data/big_data_processed_train_and_val` directory, where `preprocessing_func_2.py` is under `audio_preprocessing` directory.
- Save the file names and class labels of those 1.5 seconds audio chunks to **data/big_data_processed_train_and_val.csv**.
