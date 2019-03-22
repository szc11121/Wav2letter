from __future__ import print_function
import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Conv1D
from keras.layers import ReLU
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Lambda
from Wav2Letter.decoder import GreedyDecoder


class Wav2Letter():
    """Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals

        TODO: use cuda if available

        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()
        self.inputs = Input(shape=(225, 13))
        self.targets = Input(shape=(6))
        
        # input_lengths = Input(tf.int32, shape=(None,))
        # target_lengths = Input(tf.int32, shape=(None,))
        self.x = Conv1D(filters=250, kernel_size=48, strides=2)(self.inputs)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=2000, kernel_size=32)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=2000, kernel_size=1)(self.x)
        self.x = ReLU()(self.x)
        self.y_pred = Conv1D(filters=num_classes, kernel_size=1)(self.x)
        self.log_probs = tf.nn.log_softmax(self.y_pred)
        self.log_probs = tf.transpose(self.log_probs, perm=[2,1])
        self.log_probs = tf.transpose(self.log_probs, perm=[1,0])
        # self.log_probs = log_probs.transpose(1, 2).transpose(0, 1)
        # mini_batch_size = len(batch)

        self.input_lengths = Input(name='input_length', shape=[1])
        self.target_lengths = Input(name='label_length', shape=[1])
        self.loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(y_pred=self.log_probs,
                                                                                    y_true=self.targets,
                                                                                    input_length=self.input_lengths,
                                                                                    label_length=self.target_lengths)

        self.model = Model(inputs=[self.inputs, self.targets, self.input_lengths, self.target_lengths], outputs=[self.loss_out])
        # Since loss is already calculated in the last layer of the net, we just pass through the results here.
        # The loss dummy labels have to be given to satify the Keras API.
        self.model.compile(loss={'ctc': lambda dummy_labels, ctc_loss: ctc_loss}, optimizer='adam')
        print(self.model.summary())

    def ctc_lambda_func(self, *args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        # y_pred = y_pred[:, 2:, :] 测试感觉没影响
        # y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 
    def forward(self, batch):
        """Forward pass through Wav2Letter network than 
            takes log probability of output

        Args:
            batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)
        log_probs = tf.nn.log_softmax(y_pred)
        print(log_probs)
        return log_probs

    def fit(self, inputs, output, optimizer, ctc_loss, batch_size, epoch, print_every=50):
        """Trains Wav2Letter model.

        Args:
            inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
            output (torch.Tensor): shape (sample_size, seq_len)
            optimizer (nn.optim): pytorch optimizer
            ctc_loss (ctc_loss_fn): ctc loss function
            batch_size (int): size of mini batches
            epoch (int): number of epochs
            print_every (int): every number of steps to print loss
        """
        split_line = math.floor(0.9 * len(inputs))
        inputs_train, output_train = inputs[:split_line], output[:split_line]
        inputs_test, output_test = inputs[split_line:], output[split_line:]

        total_steps = math.ceil(len(inputs_train) / batch_size)
        seq_length = output.shape[1]

        for t in range(epoch):

            samples_processed = 0
            avg_epoch_loss = 0

            for step in range(total_steps):
                optimizer.zero_grad()
                batch = \
                    inputs_train[samples_processed:batch_size + samples_processed]

                # log_probs shape (batch_size, num_classes, output_len)
                log_probs = self.forward(batch)

                # CTC_Loss expects input shape
                # (input_length, batch_size, num_classes)
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                # CTC arguments
                # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
                # better definitions for ctc arguments
                # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
                mini_batch_size = len(batch)
                targets = output_train[samples_processed: mini_batch_size + samples_processed]

                input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.IntTensor([target.shape[0] for target in targets])
                print(input_lengths)
                print(target_lengths)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

                avg_epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                samples_processed += mini_batch_size

                if step % print_every == 0:
                    print("epoch", t + 1, ":" , "step", step + 1, "/", total_steps, ", loss ", loss.item())
                    
            #eval test data every epoch
            samples_processed = 0
            avg_epoch_test_loss = 0
            total_steps_test = math.ceil(len(inputs_test) / batch_size)
            for step in range(total_steps_test):
                batch = inputs_test[samples_processed:batch_size + samples_processed]

                log_probs = self.forward(batch)
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                mini_batch_size = len(batch)
                targets = output_test[samples_processed: mini_batch_size + samples_processed]

                input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.IntTensor([target.shape[0] for target in targets])

                test_loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                avg_epoch_test_loss += test_loss.item()
                samples_processed += mini_batch_size
            print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)
            print("epoch", t + 1, "average epoch test_loss", avg_epoch_test_loss / total_steps_test)

    def eval(self, sample):
        """Evaluate model given a single sample

        Args:
            sample (torch.Tensor): shape (n_features, frame_len)

        Returns:
            log probabilities (torch.Tensor):
                shape (n_features, output_len)
        """
        _input = sample.reshape(1, sample.shape[0], sample.shape[1])
        log_prob = self.forward(_input)
        return log_prob
