'''
Implementing Speech Commands dataset from tensorflow docs.
Goal is to eventually create a custom model for speech recognition.

author: Omar Barazanji
date: 1/29/21
-
Python 3.7.x
Tensorflow 2.1
-
sources: https://www.tensorflow.org/tutorials/audio/simple_audio#spectrogram
'''

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class Speech():

  def __init__(self):

    self.data_dir = pathlib.Path('data/mini_speech_commands')
    if not self.data_dir.exists():
      tf.keras.utils.get_file(
          'mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True,
          cache_dir='.', cache_subdir='data')

  # returns the WAV-encoded audio as a Tensor and the sample rate.
  def decode_audio(self, audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

  # The label for each WAV file is its parent directory.
  def get_label(self, file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-2]

  # takes in the filename of the WAV file and output a tuple containing
  # the audio and labels for supervised training.
  def get_waveform_and_label(self, file_path):
    label = self.get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = self.decode_audio(audio_binary)
    return waveform, label

  def get_waveform_predict(self, file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = self.decode_audio(audio_binary)
    return waveform

  def get_spectrogram(self, waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

  def plot_spectrogram(self, spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    X = np.arange(16000, step=height + 1)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

  def get_spectrogram_and_label_id(self, audio, label):
    spectrogram = self.get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    id_find = label == self.commands
    id_find = tf.cast(id_find, tf.float32)
    label_id = tf.argmax(id_find)
    return spectrogram, label_id

  def preprocess_dataset(self, files):
    AUTOTUNE = self.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        self.get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds

  def preprocess_dataset_predict(self, files):
    AUTOTUNE = self.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(self.get_waveform_predict, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      self.get_spectrogram_predict,  num_parallel_calls=AUTOTUNE)
    return output_ds

  def get_spectrogram_predict(self, audio):
    spectrogram = self.get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

  def organize_data(self):
    data_dir = self.data_dir
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    self.commands = commands
    print('Commands:', commands)

    # Extracting audio files into a list and shuffling it.
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
          len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])

    # Split data into train, validation, and test samples
    train_files = filenames[:6400]
    self.val_files = filenames[6400: 6400 + 800]
    val_files = self.val_files
    self.test_files = filenames[-800:]
    test_files = self.test_files

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    # apply process_path to build your training set to extract the 
    # audio-label pairs and check the results.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    self.AUTOTUNE = AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    # examine a few audio waveforms with their corresponding labels:
    rows = 3
    cols = 3
    n = rows*cols
    fig1, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      ax.plot(audio.numpy())
      ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
      label = label.numpy().decode('utf-8')
      ax.set_title(label)

    '''
    Audio Processing Note: 

    A Fourier transform (tf.signal.fft) converts a signal to its component frequencies, but loses 
    all time information. The STFT (tf.signal.stft) splits the signal into windows of time
    and runs a Fourier transform on each window, preserving some time information,
    and returning a 2D tensor that you can run standard convolutions on.

    STFT produces an array of complex numbers representing magnitude and phase. 
    However, you'll only need the magnitude for this tutorial, which can be derived 
    by applying tf.abs on the output of tf.signal.stft.
    '''
    for waveform, label in waveform_ds.take(1):
      label = label.numpy().decode('utf-8')
      spectrogram = self.get_spectrogram(waveform)
  
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)

    fig2, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    self.plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')

    spectrogram_ds = waveform_ds.map(
        self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    self.spectrogram_ds = spectrogram_ds

    rows = 3
    cols = 3
    n = rows*cols
    fig3, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      self.plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
      ax.set_title(commands[label_id.numpy()])
      ax.axis('off')

  def build_train(self):
    AUTOTUNE = self.AUTOTUNE
    spectrogram_ds = self.spectrogram_ds
    commands = self.commands

    # repeat the training set preprocessing on the validation and test sets.
    train_ds = spectrogram_ds
    val_ds = self.preprocess_dataset(self.val_files)
    test_ds = self.preprocess_dataset(self.test_files)

    # Batch the training and validation sets for model training.
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Add dataset cache() and prefetch() operations to reduce read latency while training the model.
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    self.test_ds = test_ds

    if not os.path.exists("speech.model"):
      for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
      print('Input shape:', input_shape)
      num_labels = len(commands)
      norm_layer = preprocessing.Normalization()
      norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
      model = models.Sequential([
          layers.Input(shape=input_shape),
          preprocessing.Resizing(32, 32), 
          norm_layer,
          layers.Conv2D(32, 3, activation='relu'),
          layers.Conv2D(64, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Dropout(0.25),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dropout(0.5),
          layers.Dense(num_labels),
      ])

      model.summary()
      model.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'],
      )

      EPOCHS = 10
      history = model.fit(
          train_ds, 
          validation_data=val_ds,  
          epochs=EPOCHS,
          callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
      )

      metrics = history.history
      fig4 = plt.figure()
      plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
      plt.legend(['loss', 'val_loss'])

      model.save('speech.model')

  def test_model(self):
    model = tf.keras.models.load_model('speech.model')
    test_audio = []
    test_labels = []

    for audio, label in self.test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=self.commands, yticklabels=self.commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')

  def predict_word(self, filepath):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    self.AUTOTUNE = AUTOTUNE
    data_dir = self.data_dir
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    self.commands = commands[commands != 'README.md']
    sample_audio = []
    sample_file = filepath
    sample_file = tf.io.gfile.glob(sample_file)
    sample_ds = self.preprocess_dataset_predict(sample_file)
    for audio in sample_ds:
      sample_audio.append(audio.numpy())
    sample_audio = np.array(sample_audio)
    model = tf.keras.models.load_model('speech.model')
    preds = model.predict(sample_audio)
    return preds
    # print(pred, commands)
    # if np.max(pred) < 5:
    #   print('error, that\'s not one of ours!')
    # print(self.commands[np.argmax(pred, axis=1)])

def trial_runs(model, word, count, test):
  results = []
  label = -1
  if word == 'stop': label = 5
  if word == 'go': label = 1
  if word == 'right': label = 4
  for x in range(count+1):
    filepath = 'tests/%s/%s/%d_%s.wav' % (test,word,x,word)
    preds = np.array(model.predict_word(filepath))
    print(model.commands[np.argmax(preds, axis=1)])
    results.append(preds[0][label])
  print(model.commands)
  return results


if __name__ == "__main__":

  model = Speech()

  # model.organize_data()

  # model.build_train()

  # model.test_model()

  test_go = trial_runs(model,'go',8,'distance')
  test_stop = trial_runs(model,'stop',8,'distance')
  fig1 = plt.figure()
  plt.plot(test_go, label='go')
  plt.plot(test_stop, label='stop')
  plt.legend()
  plt.title("Victor's Voice - Distance vs Output Neuron per Label")
  plt.xlabel("Distance (m)")
  plt.ylabel("Output Neuron Confidence")

  test_right = trial_runs(model,'right',6,'noise')
  test_stop2 = trial_runs(model,'stop',6,'noise')
  fig2 = plt.figure()
  plt.plot(test_right, label='right')
  plt.plot(test_stop2, label='stop')
  plt.legend()
  plt.title("Omar's Voice - Noise vs Output Neuron per Label")
  plt.xlabel("Noise (+2dB per interval)")
  plt.ylabel("Output Neuron Confidence")

  plt.show()