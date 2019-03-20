"""
Trains Wav2Letter model using speech data
    
    TODO:
        * show accuracy metrics

"""
import argparse
import tensorflow as tf
from Wav2Letter.model_tf import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
from Wav2Letter.decoder import GreedyDecoder


def train(batch_size, epochs):
    # load saved numpy arrays for google speech command
    gs = GoogleSpeechCommand()
    _inputs, _targets = gs.load_vectors("./speech_data")

    # paramters
    batch_size = batch_size
    mfcc_features = 13
    grapheme_count = gs.intencode.grapheme_count
    index2char = gs.intencode.index2char

    print("training google speech dataset")
    print("data size", len(_inputs))
    print("batch_size", batch_size)
    print("epochs", epochs)
    print("num_mfcc_features", mfcc_features)
    print("grapheme_count", grapheme_count)
    print("index2char", index2char)

    # tensors
    inputs = torch.Tensor(_inputs)
    targets = torch.IntTensor(_targets)

    print("input shape", inputs.shape)
    print("target shape", targets.shape)

    # Initialize model
    model = Wav2Letter(mfcc_features, grapheme_count)
    print(model.layers)

    ctc_loss = tf.nn.ctc_loss()
    optimizer = tf.keras.optimizers.Adam()

    # Each mfcc feature is a channel
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    # transpose (sample_size, in_frame_len, mfcc_features)
    # to      (sample_size, mfcc_features, in_frame_len)
    inputs = inputs.transpose(1, 2)
    print("transposed input", inputs.shape)

    model.fit(inputs, targets, optimizer, ctc_loss, batch_size, epoch=epochs)

    sample = inputs[0]
    sample_target = targets[0]
    
    log_probs = model.eval(sample)
    output = GreedyDecoder(log_probs)

    print("sample target", sample_target)
    print("predicted", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='total epochs (default: 100)')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    train(batch_size, epochs)
