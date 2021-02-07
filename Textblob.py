# Aplicando TextBlob

#pip install textblob
#pip install google-cloud-translate
#pip install unidecode

# importanto as bibliotecas necessárias
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import os
from google.cloud import translate_v2 as translate
from unidecode import unidecode
import re
from sklearn import metrics

# Criando um modelo com API Textblob
from textblob import TextBlob

# Configurando Google Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= r"C:\Users\XXXX\Documents\PUC Minas\13-TCC- Ciencia de Dados e Big Data\Twitter RFB\ConversaoTweet-c763ac32a40c.json"

#carregando os dados coletados já classificado
Coleta_Twitter = pd.read_csv("ColetaRFB.csv")
Coleta_Twitter.head()
Coleta_Twitter.count()

# Apagar duplicidades
Coleta_Twitter.drop_duplicates(inplace = True)
Coleta_Twitter.count()


Coleta_Twitter['Tweet_Ingles'] = 'NaN'
Coleta_Twitter['Class_TextBlob'] = 'NaN'
Coleta_Twitter

translate_client = translate.Client()

# Função para remover links e informações desnecessarias
def Remove_links_pontuacoes(tweet):
    # remove links, pontos, virgulas,ponto e virgulas dos tweets
    tweet = re.sub(r"http\S+", "", tweet).lower().replace('.',' ').replace(';',' ').replace('-','').replace(':',' ').replace(')','').replace('(','').replace('?',' ').replace(',',' ').replace("'",'')
    tweet= re.sub(r"@\S+", "", tweet).replace('!',' ').strip().replace('  ',' ')
    return (" ".join(tweet.split()))

#Remove_links_pontuacoes
Coleta_Twitter['Tweet'] = Coleta_Twitter['Tweet'].apply(Remove_links_pontuacoes)
Coleta_Twitter

#Apagando registros duplicados
Coleta_Twitter["Tweet"].replace('', np.nan, inplace=True)
Coleta_Twitter.dropna(subset=['Tweet'], inplace=True)
Coleta_Twitter.drop_duplicates(inplace = True)

#Traduzindo para inglês e fazendo a analise de sentimento com TextBlob
for index, row in Coleta_Twitter.iterrows():
        texto = translate_client.translate(Coleta_Twitter['Tweet'][index],target_language='en')
        Coleta_Twitter.loc[index,'Tweet_Ingles']= texto["translatedText"]
        class_sent = TextBlob(texto["translatedText"])
        if  class_sent.polarity == 0.0:
            Coleta_Twitter.loc[index,'Class_TextBlob'] = 'Neutro'
        elif class_sent.polarity > 0.0:
            Coleta_Twitter.loc[index,'Class_TextBlob'] = 'Positivo'
        else:
            Coleta_Twitter.loc[index,'Class_TextBlob'] = 'Negativo'

# Verificando o quantitativo de cada classificação
print('Neutro = ' + str(Coleta_Twitter.Class_TextBlob[Coleta_Twitter.Class_TextBlob == 'Neutro'].count()),
      'Negativa = '+ str(Coleta_Twitter.Class_TextBlob[Coleta_Twitter.Class_TextBlob == 'Negativo'].count()),
      'Positiva = '+ str(Coleta_Twitter.Class_TextBlob[Coleta_Twitter.Class_TextBlob == 'Positivo'].count()))


# Gravar no CSV
Coleta_Twitter.to_csv("ColetaRFB_TextBlob.csv", encoding="utf-8", index=False)

# Medir Acuracia do Modelo
Classificacao_real = Coleta_Twitter['Classificacao']
Classificacao_TextBlob = Coleta_Twitter['Class_TextBlob']
metrics.accuracy_score(Classificacao_real,Classificacao_TextBlob)

# Matriz de confusão
print(pd.crosstab(Classificacao_real, Classificacao_TextBlob, rownames = ["Sentimento Real"], 
                  colnames=["Predição TextBlob"], margins=True))

# Medidas de validação do modelo - Dataset inteiro
print(metrics.classification_report(Classificacao_real, Classificacao_TextBlob))