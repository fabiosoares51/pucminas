# Naive Bayes

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from unicodedata import normalize
from nltk.stem.rslp import RSLPStemmer

# Carregando os dados coletados
Twitter_Dataset = pd.read_csv("ColetaRFB.csv") 
Twitter_Dataset

# Colocar todos em letras minusculas
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.str.lower()
Twitter_Dataset

# Remover acentos
def Remove_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Remove_acentos)
Twitter_Dataset

# Função de remover as StopWords
def RemoverStopWords(frase):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    Newwords = ['rt', 'RT', 'voce', 'r$', 'receita','federal','pra','nao','sobre']
    stopwords.extend(Newwords)
    frase = str(frase)
    palavras = [i for i in frase.split() if not i in stopwords]
    return (" ".join(palavras))

# Função para remover links e informações desnecessarias
def Remove_links_pontuacoes(tweet):
    # remove links, pontos, virgulas,ponto e virgulas dos tweets
    tweet = re.sub(r"http\S+", "", tweet).lower().replace('.',' ').replace(';',' ').replace('-','').replace(':',' ').replace(')','').replace('(','').replace('?',' ').replace(',',' ').replace("'",'')
    tweet= re.sub(r"@\S+", "", tweet).replace('!',' ').strip().replace('  ',' ')
    return (" ".join(tweet.split()))

#Função de Stemming
def Lematizacao(frase):
    lemmatizer = RSLPStemmer()
    frase = str(frase)
    palavras = [i for i in frase.split()]
    for i in range(len(palavras)):
        palavras[i] = lemmatizer.stem(palavras[i])
    return (" ".join(palavras))

# Removendo StopWords
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(RemoverStopWords)

#Remove_links_pontuacoes
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Remove_links_pontuacoes)

#Aplica a Lematizacao
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Lematizacao)

# Removendo as duplicidades e Tweets em branco
Twitter_Dataset.drop_duplicates(inplace = True)
Twitter_Dataset["Tweet"].replace('', np.nan, inplace=True)
Twitter_Dataset.dropna(subset=['Tweet'], inplace=True)
Twitter_Dataset.count()

# Dividindo do Dataset em treino e teste
Twitter_Treino_Dataset, Twitter_Teste_Dataset = train_test_split(Twitter_Dataset)

# Quantidade de Tweets
print(Twitter_Teste_Dataset.count())
print(Twitter_Treino_Dataset.count())

# Verificando a base de treinamento
print('Neutro = ' + str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Neutro"].count()),
      'Negativo = '+ str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Negativo"].count()),
      'Positivo = '+ str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Positivo"].count()))

# Processo de tokenização
tweet_tokenizer = TweetTokenizer() 

# Separando as colunas de interesse
Twitter_treino = Twitter_Treino_Dataset['Tweet'].values
Classificacao_treino = Twitter_Treino_Dataset['Classificacao'].values

# Criando e Treinando o modelo
vectorizer = CountVectorizer(analyzer="word", tokenizer = tweet_tokenizer.tokenize )
tweets_treino_rep = vectorizer.fit_transform(Twitter_treino)
modelo_NB = MultinomialNB()
modelo_NB.fit(tweets_treino_rep,Classificacao_treino)

# Validação cruzada do modelo
Resultado = cross_val_predict(modelo_NB, tweets_treino_rep, Classificacao_treino, cv = 10)
Resultado

# Acurácia
metrics.accuracy_score(Classificacao_treino, Resultado)

# Medidas de validação do modelo - Dataset de treino
print(metrics.classification_report(Classificacao_treino, Resultado))

# Matriz de confusão Treino
print(pd.crosstab(Classificacao_treino, Resultado, rownames = ["Sentimento Real"], colnames=["Predição NB"], margins=True))

# Testando o modelo
Twitter_Val = Twitter_Teste_Dataset['Tweet'].values
Classificacao_Val = Twitter_Teste_Dataset['Classificacao'].values
Freq_Twitter_Val = vectorizer.transform(Twitter_Val)
Result_Val = modelo_NB.predict(Freq_Twitter_Val)
Result_Val

# Acurácia
metrics.accuracy_score(Classificacao_Val, Result_Val)

# Medidas de validação do modelo - Dataset de teste
print(metrics.classification_report(Classificacao_Val, Result_Val))

# Matriz de confusão - Dataset de Teste
print(pd.crosstab(Classificacao_Val, Result_Val, rownames = ["Sentimento Real"], colnames=["Predição NB"], margins=True))

# Gravando os resultados
Twitter_Teste_Dataset['NaiveBayes'] = Result_Val
Twitter_Treino_Dataset['NaiveBayes'] = Classificacao_treino
frames = [Twitter_Teste_Dataset, Twitter_Treino_Dataset]
Twitter_RFB = pd.concat(frames)
Twitter_RFB
Twitter_RFB.to_csv("Twitter_NB.csv")
