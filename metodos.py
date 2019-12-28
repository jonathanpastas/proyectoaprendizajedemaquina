from gensim import corpora, models
import gensim
from nltk import WordNetLemmatizer
from nltk import SnowballStemmer
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

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


def limpiarData(texto):
    txt = str(texto)
    txt1 = txt.lower()
    lmp1 = re.sub('[^A-Za-z0-9]+', ' ', txt1)
    lmp1 = re.sub(r'\d', '', lmp1)
    data = lmp1.split()

    return data


def eliminaStopWords(texto1):
    nltk.download('stopwords')
    steam = stopwords.words('english')
    # Eliminacion de StopWords
    for palabras in texto1:
        if palabras in steam:
            texto1.remove(palabras)
    return texto1


def steamming(texto2):
    datas = []
    stemmer = PorterStemmer()
    for x in range(len(texto2)):
        datas.append(stemmer.stem(texto2[x]))
    return datas


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


def similitudpreproceso (texto):
    song = texto.lower()
    song = re.sub('[^A-Za-z0-9]+', ' ', song)
    song = re.sub(r'\d', '', song)
    return song

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


def soloTitulos(lda,pos):
        auxtxt = str(lda[pos])
        auxt1 = re.sub('[^A-Za-z0-9]+', ' ', auxtxt)
        auxt1 = re.sub(r'\d', '', auxt1)
        return str(auxt1)


