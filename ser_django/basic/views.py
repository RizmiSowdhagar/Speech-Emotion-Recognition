from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader

import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import sounddevice as sd
# from scipy.io.wavfile import write
import wavio as wv
import pathlib
from django.http import HttpResponse
from django.template import loader

def to_decibles(signal):
    # Perform short time Fourier Transformation of signal and take absolute value of results
    stft = np.abs(librosa.stft(signal))
    # Convert to dB
    D = librosa.amplitude_to_db(stft, ref = np.max) # Set reference value to the maximum value of stft.
    return D # Return converted audio signal

# Function to plot the converted audio signal
def plot_spec( D, filename, sr=16000):
    fig, ax = plt.subplots(figsize = (20,10))
    spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    # ax.set(title = 'Spectrogram of ' + instrument)
    # fig.colorbar(spec)
    plt.savefig(filename)

def convtopng():
    testing,_ = librosa.load('./basic/rec2spec/recording1.wav')
    plot_spec(to_decibles(testing), os.path.join('./basic/rec2spec/','recording1.png'))
    x=pred()
    return x
class_names=['angry', 'happy', 'neutral', 'sad']



def Voice_rec(request):
    freq = 44100
    
    # Recording duration
    duration = 5
    
    # Start recorder with the given values 
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=1)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    # Convert the numpy array to wav file
    path='./basic/rec2spec'
    wv.write(os.path.join(path,"recording1.wav"), recording, freq, sampwidth=2)
    # data= convtopng()
    # print("haye pataka main: ", os.listdir())
    template = loader.get_template('index.html')
    context = convtopng()
    return HttpResponse(template.render(context, request))



# def index(request):
def pred():
    try:
        json_file = open('./basic/model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("./basic/model/model.h5")
        print("Loaded model from disk")
    except:
        print('model load nhi hua')

    rec_path = './basic/rec2spec/recording1.png'

    img = keras.utils.load_img(
    rec_path, target_size=(320, 320))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    data = {'state': class_names[np.argmax(score)], 'conf': 100 * np.max(score)}
    # data= {"state":'saddddd' , "conf": 0.92}
    # template = loader.get_template('index.html')
    return data
    # return HttpResponse(template.render(data, request))
    # return render(request, 'index.html', output=data)

    # return render(request, 'index.html')


####
