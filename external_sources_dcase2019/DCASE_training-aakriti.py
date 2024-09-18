# select a GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import soundfile as sound
import keras
from datetime import datetime
import tensorflow
from keras.optimizers import SGD
from DCASE2019_improvised_network import model_resnet_new
from DCASE_training_functions import LR_WarmRestart, MixupGenerator
from tqdm import tqdm
from keras.utils import plot_model

print("Librosa version = ", librosa.__version__)
print("Pysoundfile version = ", sound.__version__)
print("keras version = ", keras.__version__)
print("tensorflow version = ", tensorflow.__version__)

WhichTask = "1a"

if WhichTask == "1a":
    ThisPath = "/work/aistwal/dataset_tau2019/extracted-files/TAU-urban-acoustic-scenes-2019-development/"
    TrainFile = ThisPath + "evaluation_setup/fold1_train.csv"
    ValFile = ThisPath + "evaluation_setup/fold1_evaluate.csv"
    sr = 48000
    num_audio_channels = 2

SampleDuration = 10

# log-mel spectrogram parameters
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints / 2)
NumTimeBins = int(np.ceil(SampleDuration * sr / HopLength))

# training parameters
max_lr = 0.1
# batch_size = 32
batch_size = 16
num_epochs = 300
# num_epochs = 250
# num_epochs = 120
# num_epochs = 510
mixup_alpha = 0.4
crop_length = 400

# load filenames and labels
dev_train_df = pd.read_csv(TrainFile, sep="\t", encoding="ASCII")
dev_val_df = pd.read_csv(ValFile, sep="\t", encoding="ASCII")
wavpaths_train = dev_train_df["filename"].tolist()
wavpaths_val = dev_val_df["filename"].tolist()
y_train_labels = dev_train_df["scene_label"].astype("category").cat.codes.values
y_val_labels = dev_val_df["scene_label"].astype("category").cat.codes.values

ClassNames = np.unique(dev_train_df["scene_label"])
NumClasses = len(ClassNames)

y_train = keras.utils.to_categorical(y_train_labels, NumClasses)
y_val = keras.utils.to_categorical(y_val_labels, NumClasses)

LM_train = np.zeros(
    (len(wavpaths_train), NumFreqBins, NumTimeBins, num_audio_channels), "float32"
)
for i in tqdm(range(len(wavpaths_train))):
    stereo, fs = sound.read(ThisPath + wavpaths_train[i], stop=SampleDuration * sr)
    for channel in range(num_audio_channels):
        if len(stereo.shape) == 1:
            stereo = np.expand_dims(stereo, -1)
        LM_train[i, :, :, channel] = librosa.feature.melspectrogram(
            y=stereo[:, channel],
            sr=sr,
            n_fft=NumFFTPoints,
            hop_length=HopLength,
            n_mels=NumFreqBins,
            fmin=0.0,
            fmax=sr / 2,
            htk=True,
            norm=None,
        )

LM_train = np.log(LM_train + 1e-8)
print("Training data shape: ", LM_train.shape)

LM_val = np.zeros(
    (len(wavpaths_val), NumFreqBins, NumTimeBins, num_audio_channels), "float32"
)
for i in tqdm(range(len(wavpaths_val))):
    stereo, fs = sound.read(ThisPath + wavpaths_val[i], stop=SampleDuration * sr)
    for channel in range(num_audio_channels):
        if len(stereo.shape) == 1:
            stereo = np.expand_dims(stereo, -1)
        LM_val[i, :, :, channel] = librosa.feature.melspectrogram(
            y=stereo[:, channel],
            sr=sr,
            n_fft=NumFFTPoints,
            hop_length=HopLength,
            n_mels=NumFreqBins,
            fmin=0.0,
            fmax=sr / 2,
            htk=True,
            norm=None,
        )

LM_val = np.log(LM_val + 1e-8)
print("Validation data shape: ", LM_val.shape)

# # Load model from checkpoint if available
# # checkpoint_path = "/work/aistwal/DCASE2019-Task1/asc_checkpoints/model_126-1.13.keras"
# if os.path.exists(checkpoint_path):
#     model = keras.saving.load_model(checkpoint_path)
#     print(f"Loaded model from checkpoint {checkpoint_path}")
# else:
#     model = model_resnet_new(
#         NumClasses, input_shape=[NumFreqBins, None, num_audio_channels]
#     )
#     print(f"Created a new model.")

model = model_resnet_new(
        NumClasses, input_shape=[NumFreqBins, None, num_audio_channels])
print(f"Created a new model.")

model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=max_lr, decay=0, momentum=0.9, nesterov=False),
    metrics=["accuracy"],
)

print(model.summary())
plot_model(model, to_file="large_asc_model.png")

# set learning rate schedule
lr_scheduler = LR_WarmRestart(
    nbatch=np.ceil(LM_train.shape[0] / batch_size),
    Tmult=2,
    initial_lr=max_lr,
    min_lr=max_lr * 1e-4,
    epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0, 511.0],
)

# create data generator
TrainDataGen = MixupGenerator(
    LM_train, y_train, batch_size=batch_size, alpha=mixup_alpha, crop_length=crop_length
)()

# Define the ModelCheckpoint callback,  by default the monitoring of model is based on validation loss, so best model is saved based on validation loss
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='asc_checkpoints/model_{epoch:02d}-{val_loss:.2f}.keras',
    save_best_only=True,
    verbose=1
)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

callbacks = [lr_scheduler, tensorboard_callback, checkpoint_callback]
# train the model
training_history = model.fit(
    x=TrainDataGen,
    validation_data=(LM_val, y_val),
    epochs=num_epochs,
    verbose=1,
    callbacks=callbacks,
    steps_per_epoch=int(np.ceil(LM_train.shape[0] / batch_size)),
)
# model.save("DCASE_" + WhichTask + "_Task_development_1.h5")
model.save("DCASE_" + WhichTask + "_Epochs_"+ str(num_epochs)+ "_Task_development_1.keras")
print("Average test loss: ", np.average(training_history.history["loss"]))
