import pyaudio
import wave
import librosa
import numpy as np
import sys
import tensorflow as tf
import time


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

def extract_feature(data,sr=8000):
    #X, sample_rate = librosa.load(file_name)
    sample_rate=sr
    X=data
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate,fmin=60.0).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def callback(in_data, frame_count, time_info, flag):
    #extraction data
    x=buf_to_float(in_data)
    print(x)
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(x)
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features= np.empty((0,193))
    features = np.vstack([features,ext_features])  
    test_x =features  
    #tensorflow execution
    training_epochs = 5000
    n_dim = 193#features.shape[1]
    n_classes =10
    n_hidden_units_one = 193 
    n_hidden_units_two = 386
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01

    X = tf.placeholder(tf.float32,[None,n_dim])
    
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd),name="Weight_1")#193,280
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd),name="bias_1")
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd),name="Weight_2")#280,300
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd),name="bias_2")
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd),name="W")#300,7
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd),name="b")
    y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)
    saver =  tf.train.Saver()
    y_pred = None
    with tf.Session() as sess:    
        saver.restore(sess, "D:\\source\\python\\UrbanSound8K\\pkl\\audio.ckpt")
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
        print(y_pred)
        if (y_pred==6):
            print('gunshot detected')
    return None, pyaudio.paContinue

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 1


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)



stream.start_stream()
while stream.is_active():
    time.sleep(0.25)
#stream.stop_stream()
stream.close()
p.terminate()



