# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:02:01 2019

@author: jonat
"""
#Proyecto IB 

import speech_recognition as sr
from pydub import AudioSegment
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import io
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from gensim import corpora, models
import gensim
from nltk import WordNetLemmatizer
from nltk import SnowballStemmer
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves

###################Reconocimiento de Audio##################################### 

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:\spechh-07f6cfbd660c.json"
#client = speech.SpeechClient()
    
#file_name='music\sacrifice1.wav'
   
#with io.open(file_name, 'rb') as audio_file:
 #  content = audio_file.read()
  # audio = types.RecognitionAudio(content=content)
        
   ##       encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
     #      sample_rate_hertz=44100,
      #     audio_channel_count=2,
       #    language_code='en-US')
        
       
#response = client.recognize(config, audio)
#txtreco=""
#for result in response.results:
 #   txtreco=txtreco+(result.alternatives[0].transcript)
  
#print(txtreco)
###############################################################################

###Supuesto reconocimiento de audio 
#txtreco="I know there's something in the wake of your smile.cast ocean from the eyes Love it. Love. You little piece of heaven.Listen to your heart.Listen to your heart.I don't listen to you.You tell him goodbye.Sometimes you wonder if this fight is worthwhile.Precious Moments lost the time Dylan LeBlanc in Listen to you.He's calling for you. Listen to your heart. There's nothing else going on.You tell him goodbye.What's up guys has Kurt and it's been awhile since I sung in one of my videos. So I hope you guys liked it. It's one of my favorite songs and it's on iTunes on a project imagination and what they're doing for that is anyone can submit a 60 second trailer and some of those trailers will be chosen to inspire a full-length Hollywood film. So I'll shut up at the Lincoln Project Imagination. Check it out. If you are it don't make her or any of that some interests you and as always there's more stuff coming soon. So I will see you guys then. 
#######

#####Web Scrapping

def webscrapping(url):

    file = urlopen(url)
    html = file.read()
    file.close()

    soup=BeautifulSoup(html)

    txtsong=""
    for links in soup.find_all("div",{"class": "cnt-letra p402_premium"}):
        txtsong=links.text+txtsong

    print(txtsong)

    return txtsong

#NPL Web Scrapping
    
###########Metodos para NPL 
def limpiarData(texto):
    txt=str(texto)
    txt1=txt.lower()
    lmp1= re.sub('[^A-Za-z0-9]+',' ', txt1)
    lmp1= re.sub(r'\d','',lmp1)
    data=lmp1.split()
    
    return data

def eliminaStopWords(texto1):

    nltk.download('stopwords')
    steam=stopwords.words('english')
    #Eliminacion de StopWords
    for palabras in texto1: 
        if palabras in steam: 
            texto1.remove(palabras)
    return texto1

def steamming(texto2):
    datas=[]
    stemmer=PorterStemmer()
    for x in range(len(texto2)): 
        datas.append(stemmer.stem(texto2[x]))
    return datas

########Implementacion de NPL en base a los Metodos sobre el web scrapping 
songori=limpiarData(txtsong)
eliminaStopWords(songori)
songori1=steamming(songori)

########Implementacion de NPL en base a los Metodos sobre el reconocimiento
songreco=limpiarData(txtreco)
eliminaStopWords(songreco)
songreco1=steamming(songreco)

###### Similitud 

def disJaccard(s1,s2):
  a=set(s1)
  b=set(s2)
  union=a.union(b)
  interseccion=a.intersection(b)
  if len(union)==0:
      if len(interseccion)==0:
          return 1
  similitud=len(interseccion)/len(union)
  return similitud

songorig=txtsong.lower()
songorig= re.sub('[^A-Za-z0-9]+',' ', songorig)
songorig= re.sub(r'\d','',songorig)


songreco=txtreco.lower()
songreco= re.sub('[^A-Za-z0-9]+',' ', songreco)
songreco= re.sub(r'\d','',songreco)

matrizSimilitud=np.zeros(shape=(len(songori),len(songreco1)))

for i in range(len(songori)):
    for j in range(len(songreco1)):
        d=disJaccard(songori[i],songreco1[j])
        matrizSimilitud[i,j]=d

print(matrizSimilitud)
eee=disJaccard(songorig,songreco)
print(eee)

#Topic Modeling
def lemmatize_stemming(text):
    nltk.download('wordnet')
    stemmer=SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
aux=preprocess(songorig)
aux=[aux]
dictionary = corpora.Dictionary(aux)
corpus = [dictionary.doc2bow(text) for text in aux]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)
ld = ldamodel.print_topics(num_topics=4, num_words=3)
print(ld)
#Grafica de Spectro 
muestreo, sonido = waves.read(file_name)
# canales: monofónico o estéreo
tamano = np.shape(sonido)
muestras = tamano[0]
m = len(tamano)
canales = 1  # monofónico
if (m>1):  # estéreo
    canales = tamano[1]
# experimento con un canal
if (canales>1):
    canal = 0
    uncanal = sonido[:,canal] 
else:
    uncanal = sonido
    
# rango de observación en segundos
inicia = 1.000
termina = 2.002
# observación en número de muestra
a = int(inicia*muestreo)
b = int(termina*muestreo)
parte = uncanal[a:b]

waves.write(file_name, muestreo, parte)
plt.plot(parte)
plt.show()