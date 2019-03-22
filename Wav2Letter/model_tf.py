from __future__ import print_function
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Sequential
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from .decoder import GreedyDecoder


class Wav2Letter():
    """
        Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals


        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes):

        self.inputs = tf.placeholder(tf.float32, shape=(None, 225, 13))
        self.targets = tf.placeholder(tf.float32, shape=(None, 6))
        self.input_lengths = tf.placeholder(tf.int32, shape=(None,))
        self.mini_batch_size = tf.placeholder(tf.int32)
        self.target_lengths = tf.placeholder(tf.int32, shape=(None,))

        self.x = Conv1D(filters=250, padding='same', kernel_size=48, strides=2, activation='relu')(self.inputs)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=250, padding='same', kernel_size=7, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=2000, padding='same', kernel_size=32, strides=1, activation='relu')(self.x)
        self.x = Conv1D(filters=2000, padding='same', kernel_size=1, strides=1, activation='relu')(self.x)
        self.y_pred = Conv1D(filters=num_classes, padding='same', kernel_size=1, strides=1, activation='relu')(self.x)

        self.log_probs = tf.nn.log_softmax(self.y_pred)
        tf.transpose(self.log_probs, perm=[2, 0, 1])
        self.input_lengths = np.ones((self.mini_batch_size.eval(),)) * self.log_probs.shape[0]
        self.loss = K.ctc_batch_cost(self.targets, self.log_probs, self.input_lengths, self.target_lengths)
        # self.loss = tf.nn.ctc_loss(self.targets, self.log_probs, self.input_lengths)
 	 
        self.train_optim = tf.train.AdamOptimizer().minimize(self.loss)
        

    def fit(self, inputs, output, batch_size, epoch, print_every=50):

        split_line = math.floor(0.9 * len(inputs))
        inputs_train, output_train = inputs[:split_line], output[:split_line]
        inputs_test, output_test = inputs[split_line:], output[split_line:]
        total_steps = math.ceil(len(inputs_train) / batch_size)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for t in range(epoch):

                samples_processed = 0
                avg_epoch_loss = 0

                for step in range(total_steps):
                    batch = \
                        inputs_train[samples_processed:batch_size + samples_processed]

                    # CTC arguments
                    # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
                    # better definitions for ctc arguments
                    # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
                    mini_batch_size = batch.shape[0]
                    targets = output_train[samples_processed: mini_batch_size + samples_processed]

                    # input_lengths = np.ones((mini_batch_size,)) * batch.shape[0]
                    target_lengths = np.asarray([target.shape[0] for target in targets])
                    sess.run(self.train_optim, feed_dict={self.inputs: batch, 
                                                          self.targets: targets,
                                                          self.input_lengths: target_lengths,
                                                          self.mini_batch_size: mini_batch_size,
                                                          self.target_lengths: target_lengths})

                    train_loss = sess.run(self.loss)
                    avg_epoch_loss += train_loss
                    samples_processed += mini_batch_size

                    if step % print_every == 0:
                        print("epoch", t + 1, ":" , "step", step + 1, "/", total_steps, ", loss ", train_loss)
                        
                # #eval test data every epoch
                # samples_processed = 0
                # avg_epoch_test_loss = 0
                # total_steps_test = math.ceil(len(inputs_test) / batch_size)
                # for step in range(total_steps_test):
                #     batch = inputs_test[samples_processed:batch_size + samples_processed]

                #     log_probs = self.forward(batch)
                #     log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                #     mini_batch_size = len(batch)
                #     targets = output_test[samples_processed: mini_batch_size + samples_processed]

                #     input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
                #     target_lengths = torch.IntTensor([target.shape[0] for target in targets])

                #     test_loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                #     avg_epoch_test_loss += test_loss.item()
                #     samples_processed += mini_batch_size
                print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)
                # print("epoch", t + 1, "average epoch test_loss", avg_epoch_test_loss / total_steps_test)

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
