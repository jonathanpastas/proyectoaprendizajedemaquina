# IMPORTAR MODULO DE FLASK , render_template, request
from flask import Flask, render_template, request
import io
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
# import ProyectoIB
from metodos import webscrapping, similitudpreproceso, disJaccard, preprocess, soloTitulos
from gensim import corpora, models
import gensim

# importar modulo de funciones

# CREAR UN OBJETO
# __name__ =nombre del modulo
app = Flask(__name__)


@app.route('/')
def formulario() -> 'html':
    return render_template('index.html', titulo='Selector de Canciones')


@app.route('/escoger', methods=['POST'])
def cambiar() -> 'html':
    cancion = request.form['cbCancion']
    if cancion == 'sacrifice':
        aut = 'Elton Jhon'
        tit = 'Sacrifice'
        path = 'D:\DESARROLLO PYTHON_SPYDER\musica\sacrifice1.wav'
        dir = 'https://www.letras.com/elton-john/20094/'

    if cancion == 'listen':
        aut = 'Roxette'
        tit = 'Listen to your health'
        path = 'D:\DESARROLLO PYTHON_SPYDER\musica\Listenfinal.wav'
        dir = 'https://www.letras.com/roxette/34460/'

    #######################################GOOGLE SPEECH RECOGNITION ###########################################
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "archivojsonclavesdegoogle"
    client = speech.SpeechClient()

    file_name = path

    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            audio_channel_count=2,
            language_code='en-US')

    response = client.recognize(config, audio)
    txtreco = ""
    for result in response.results:
        txtreco = txtreco + (result.alternatives[0].transcript)

    print(txtreco)
    #############################################################################################################

    ##########################################Proceso###########################################################
    web = webscrapping(dir)
    a1 = similitudpreproceso(txtreco)
    a2 = similitudpreproceso(web)
    dis = disJaccard(a2, a1)

    aux = preprocess(a2)
    aux = [aux]
    dictionary = corpora.Dictionary(aux)
    corpus = [dictionary.doc2bow(text) for text in aux]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    ld = ldamodel.print_topics(num_topics=3, num_words=3)
    topics1 = soloTitulos(ld, 0)
    topics2 = soloTitulos(ld, 1)
    topics3 = soloTitulos(ld, 2)
    # print(topics)
    #############################################################################################################
    return render_template('resultado.html', titulo='Resultados Obtenidos ', autor=aut, name=tit, transcripcion=a1,
                           webscra=a2, similitud=str((round(dis, 2) * 100)-20), t1=topics1, t2=topics2, t3=topics3)


# Ejecucion de la app
app.run(debug=True)
