import os
import numpy as np
import random
import pickle
from sonopy import mfcc_spec
from scipy.io import wavfile
from tqdm import tqdm


class IntegerEncode:
    """
        Encodes labels into integers
    
    Args:
        labels (list): shape (n_samples, strings)
    """

    def __init__(self, labels):

        self.char2index = {
            "pad":-1
        }
        self.index2char = {
            -1: "pad"
        }
        self.grapheme_count = 0 #字母统计
        self.process(labels)    #完成char2index和index2char
        self.max_label_seq = 6

    def process(self, labels):
        """
            builds the encoding values for labels
        
        Args:
            labels (list): shape (n_samples, strings)
        """
        strings = "".join(labels)
        for s in strings:
            if s not in self.char2index:
                self.char2index[s] = self.grapheme_count
                self.index2char[self.grapheme_count] = s
                self.grapheme_count += 1

    def convert_to_ints(self, label):
        """
            Convert into integers
        
        Args:
            label (str): string to encode
        
        Returns:
            list: shape (max_label_seq)
        """
        y = []
        for char in label:
            y.append(self.char2index[char])
        if len(y) < self.max_label_seq:
            diff = self.max_label_seq - len(y)
            pads = [self.char2index["pad"]] * diff
            y += pads
        return y

    def save(self, file_path):
        """
            Save integer encoder model as a pickle file

        Args:
            file_path (str): path to save pickle object
        """
        file_name = os.path.join(file_path, "int_encoder.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f)


def normalize(values):
    """
        Normalize values to mean 0 and std 1
    
    Args:
        values (np.array): shape (frame_len, features)
    
    Returns:
        np.array: normalized features
    """
    return (values - np.mean(values)) / np.std(values)


class GoogleSpeechCommand():
    """
        Data set can be found here 
        https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
    """

    def __init__(self, data_path="speech_data/speech_commands_v0.01", sample_rate=16000):
        self.data_path = data_path
        self.labels = [
            'right', 'eight', 'cat', 'tree', 'bed', 'happy', 'go', 'dog', 'no', 
            'wow', 'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero',
            'seven', 'up', 'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 
            'five', 'off', 'four'
        ]
        self.intencode = IntegerEncode(self.labels)
        self.sr = sample_rate
        self.max_frame_len = 225

    def get_data(self, progress_bar=True):
        """
            Currently returns mfccs and integer encoded data

        Returns:
            (list, list): 
                inputs shape (sample_size, frame_len, mfcs_features)
                targets shape (sample_size, seq_len)  seq_len is variable
        """
        pg = tqdm if progress_bar else lambda x: x

        inputs, targets, input_lengths= [], [], []
        meta_data = []
        for label in self.labels:
            path = os.listdir(os.path.join(self.data_path, label))
            for audio in path:
                audio_path = os.path.join(self.data_path, label, audio)
                meta_data.append((audio_path, label))
        
        random.shuffle(meta_data)   #打乱数据集

        for md in pg(meta_data):
            audio_path = md[0]
            label = md[1]
            _, audio = wavfile.read(audio_path)
            mfccs = mfcc_spec(
                audio, self.sr, window_stride=(160, 80),
                fft_size=512, num_filt=20, num_coeffs=13
            )
            mfccs = normalize(mfccs)
            diff = self.max_frame_len - mfccs.shape[0]
            input_lengths.append(mfccs.shape[0])
            mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")#padding
            inputs.append(mfccs)

            target = self.intencode.convert_to_ints(label)
            targets.append(target)
        return inputs, targets, input_lengths

    @staticmethod
    def save_vectors(file_path, x, y, x_length):
        """
            saves input and targets vectors as x.npy and y.npy
        
        Args:
            file_path (str): path to save numpy array
            x (list): inputs
            y (list): targets
        """
        x_file = os.path.join(file_path, "x")
        y_file = os.path.join(file_path, "y")
        length_file = os.path.join(file_path, "x_length")
        np.save(x_file, np.asarray(x))
        np.save(y_file, np.asarray(y))
        np.save(length_file, np.asarray(x_length))

    @staticmethod
    def load_vectors(file_path):
        """
            load inputs and targets
        
        Args:
            file_path (str): path to load targets from
        
        Returns:
            inputs, targets: np.array, np.array
        """
        x_file = os.path.join(file_path, "x.npy")
        y_file = os.path.join(file_path, "y.npy")
        length_file = os.path.join(file_path, "x_length.npy")
        inputs = np.load(x_file)
        targets = np.load(y_file)
        input_lengths = np.load(length_file)
        return inputs, targets, input_lengths


if __name__ == "__main__":
    gs = GoogleSpeechCommand()
    inputs, targets, input_lengths = gs.get_data()
    gs.save_vectors("./speech_data", inputs, targets, input_lengths)
    gs.intencode.save("./speech_data")
    print("preprocessed and saved")
