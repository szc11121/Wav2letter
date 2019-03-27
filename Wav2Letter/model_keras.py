from __future__ import print_function
from functools import reduce
import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import keras
import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as K
from keras.layers import Conv1D
from keras.layers import ReLU
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Lambda
from Wav2Letter.decoder import GreedyDecoder


class Wav2Letter():

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()
        self.inputs = Input(name='input', shape=(225, 13))
        
        

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
        self.y_pred = Conv1D(name='pred', filters=num_classes+1, kernel_size=1)(self.x)

        self.log_probs = Activation('softmax', name='log_probs')(self.y_pred)

        self.targets = Input(name='target', shape=(6,))
        self.input_lengths = Input(name='input_length', shape=(1,))
        self.target_lengths = Input(name='label_length', shape=(1,))
        self.loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([self.log_probs, self.targets, self.input_lengths, self.target_lengths])
        
        self.model = Model(inputs=[self.inputs, self.targets, self.input_lengths, self.target_lengths], outputs=[self.loss_out])
  
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        self.predict_model = Model(inputs=self.model.get_layer('input').input, outputs=self.model.get_layer('log_probs').output)
        self.model.summary()

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        return K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)

    def fit(self, inputs, output, input_lengths, batch_size, epoch=15, print_every=50):
        """
        Trains Wav2Letter model.
        """
        split_line = math.floor(0.9 * len(inputs))
        predict_lengths = np.ones((inputs.shape[0], 1)) * (self.model.get_layer('pred').output_shape[1])
        target_lengths = np.asarray([np.argwhere(output_ == -1)[0][0] if np.argwhere(output_ == -1).shape[0]>0 else output_.shape[0] for output_ in output]).reshape(-1,1)
        inputs_train, output_train = inputs[:split_line], output[:split_line]
        inputs_test, output_test = inputs[split_line:], output[split_line:]
        input_lengths_train, input_lengths_test = predict_lengths[:split_line], predict_lengths[split_line:]
        target_lengths_train, target_lengths_test = target_lengths[:split_line], target_lengths[split_line:]

        # self.model.fit(
        #     x=[inputs, output, predict_lengths, target_lengths],
        #     y=np.ones((inputs.shape[0],1)),
        #     batch_size=batch_size,
        #     epochs=epoch,
        #     validation_split=0.1,
        #     )
        total_steps = math.ceil(len(inputs_train) / batch_size)

        for t in range(epoch):

            samples_processed = 0
            avg_epoch_loss = 0

            for step in range(total_steps):

                batch = \
                    inputs_train[samples_processed:batch_size + samples_processed]

                mini_batch_size = len(batch)
                targets = output_train[samples_processed: mini_batch_size + samples_processed]
                input_lengths = input_lengths_train[samples_processed: mini_batch_size + samples_processed]
                target_lengths = target_lengths_train[samples_processed: mini_batch_size + samples_processed]
   
                loss = self.model.train_on_batch([batch, targets, input_lengths, target_lengths], np.ones((mini_batch_size,1)))

                avg_epoch_loss += loss
                samples_processed += mini_batch_size

                if step % print_every == 0:
                    print("epoch", t + 1, ":" , "step", step + 1, "/", total_steps, ", loss ", loss.item())
                    
            #eval test data every epoch
            samples_processed = 0
            avg_epoch_test_loss = 0
            total_steps_test = math.ceil(len(inputs_test) / batch_size)
            for step in range(total_steps_test):
                batch = inputs_test[samples_processed:batch_size + samples_processed]
                mini_batch_size = len(batch)
                targets = output_test[samples_processed: mini_batch_size + samples_processed]

                input_lengths = input_lengths_test[samples_processed: mini_batch_size + samples_processed]
                target_lengths = target_lengths_test[samples_processed: mini_batch_size + samples_processed]
   
                test_loss = self.model.test_on_batch(
                    [batch, targets, input_lengths, target_lengths],
                    np.ones((mini_batch_size,1))
                    )
                avg_epoch_test_loss += test_loss
                samples_processed += mini_batch_size
            print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)
            print("epoch", t + 1, "average epoch test_loss", avg_epoch_test_loss / total_steps_test)
            #visualize sample
            sample = inputs_test[0]
            sample_target = output_test[0]
            
            output = self.eval(sample)

            print("sample target", sample_target)
            print("predicted", output.eval())

    def eval(self, sample):
        """Evaluate model given a single sample

        Args:
            sample (torch.Tensor): shape (n_features, frame_len)

        Returns:
            log probabilities (torch.Tensor):
                shape (n_features, output_len)
        """
        
        _input = sample.reshape(1, sample.shape[0], sample.shape[1])
        log_prob = self.predict_model.predict(_input)
        output = K.ctc_decode(log_prob, input_length=np.asarray(self.model.get_layer('pred').output_shape[1]).reshape(1,))
        return output
