import argparse
import tensorflow as tf
import numpy as np
from tensorflow import keras
from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand


def train(batch_size, epochs):
    gs = GoogleSpeechCommand()
    inputs, targets, input_lengths= gs.load_vectors("./speech_data")

    # paramters
    batch_size = batch_size
    mfcc_features = 13
    grapheme_count = gs.intencode.grapheme_count
    index2char = gs.intencode.index2char

    print("training google speech dataset")
    print("data size", len(inputs))
    print("batch_size", batch_size)
    print("epochs", epochs)
    print("num_mfcc_features", mfcc_features)
    print("grapheme_count", grapheme_count)
    print("index2char", index2char)


    print("input shape", inputs.shape)
    print("target shape", targets.shape)

    model = Wav2Letter(mfcc_features, grapheme_count)
    
    model.fit(inputs, targets, input_lengths, batch_size, epoch=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='total epochs (default: 100)')
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    # batch_size = 256
    # epochs = 15
    train(batch_size, epochs)

