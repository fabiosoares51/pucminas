# Long short-term memory 

pip install tensorflow
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install wordcloud -q')

# Importando as bibliotecas necessarias
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import unidecode
from sklearn import metrics
from unicodedata import normalize
from nltk.stem.rslp import RSLPStemmer
from matplotlib import pyplot as plt

# Carregando os dados
#pd.set_option('mode.chained_assignment',None)
Twitter_Dataset = pd.read_csv("ColetaRFB.csv")
Twitter_Dataset

# Colocar todos em letras minusculas
Twitter_Dataset.loc[:,'Tweet'] = Twitter_Dataset.Tweet.str.lower()
Twitter_Dataset

# Remover acentos
def Remove_acentos(txt):
    txt = str(txt)
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

# Função de remover as StopWords
def Remove_StopWords(frase):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    Newwords = ['rt', 'RT', 'voce','r$','pra','nao','sobre']
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


# Removendo StopWords
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Remove_StopWords)

#Remove Acentos#Remove_#Remove_links_pontuacoes
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Remove_acentos)
Twitter_RFB = Twitter_Dataset.copy()

#Remove_links_pontuacoes
Twitter_Dataset['Tweet'] = Twitter_Dataset.Tweet.apply(Remove_links_pontuacoes)

# Verificar se tem Tweet em branco e deletar o registro, além de deletar as duplicações
Twitter_Dataset["Tweet"].replace('', np.nan, inplace=True)
Twitter_Dataset.dropna(subset=['Tweet'], inplace=True)
Twitter_Dataset.drop_duplicates(inplace = True)
Twitter_Dataset

# Verificando a base inteira
Neu = Twitter_Dataset.Classificacao[Twitter_Dataset.Classificacao == "Neutro"].count()
Neg = Twitter_Dataset.Classificacao[Twitter_Dataset.Classificacao == "Negativo"].count()
Pos = Twitter_Dataset.Classificacao[Twitter_Dataset.Classificacao == "Positivo"].count()

print('Neutro = ' + str(Neu),
      'Negativo = '+ str(Neg),
      'Positivo = '+ str(Pos))

# Plotando Grafico dos dados
sentimentos = ['Positivo', 'Negativo', 'Neutro']
valores = [Pos, Neg, Neu]
plt.bar(sentimentos, valores)
plt.show()

fig1, ax1 = plt.subplots()
cores = ['Green','Red','Yellow']
ax1.pie(valores, labels=sentimentos, autopct='%1.1f%%', colors=cores )
ax1.axis('equal')
ax1.set_title("Distribuição dos sentimentos no Dataset", weight="bold")
plt.show()

# Veriricando o Twitter com o maior número de palavras
Tamanho_frase = Twitter_Dataset['Tweet'].str.split().apply(len).max()
Twitter_Dataset['Tweet'][1289]

# Dividindo a base de treinamento e teste
Twitter_Treino_Dataset, Twitter_Teste_Dataset = train_test_split(Twitter_Dataset)
Twitter_Teste_Dataset.head()
Twitter_Treino_Dataset.head()

# Verificando a base de treinamento
print('Neutro = ' + str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Neutro"].count()),
      'Negativo = '+ str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Negativo"].count()),
      'Positivo = '+ str(Twitter_Treino_Dataset.Classificacao[Twitter_Treino_Dataset.Classificacao == "Positivo"].count()))

# Transformando numa lista
Tweet_Treino = Twitter_Treino_Dataset['Tweet'].values
Classificacao_Treino = Twitter_Treino_Dataset['Sentimento'].values
Tweet_Teste = Twitter_Teste_Dataset['Tweet'].values
Classificacao_Teste = Twitter_Teste_Dataset['Sentimento'].values

# Processo de Tokenização e Padding
Max_Palavras = 10000
tokenizer = Tokenizer(num_words=Max_Palavras, oov_token="<OOV>")
#Este metodo cria um vocabulário de indices baseados na frequencia das palavras
tokenizer.fit_on_texts(Tweet_Treino) 
tokenizer.fit_on_texts(Tweet_Teste) 
#Substitui cada palavra do texto pelo seu indice correspondente. 
Tweet_Treino_token = tokenizer.texts_to_sequences(Tweet_Treino)
Tweet_Teste_token = tokenizer.texts_to_sequences(Tweet_Teste)

# Deixa todas as sequencias com as mesmas dimensões
Tweet_Padded = pad_sequences(Tweet_Treino_token, padding='post', maxlen = Tamanho_frase)
Tweet_Padded_Teste = pad_sequences(Tweet_Teste_token, padding='post', maxlen = Tamanho_frase)
print(Tweet_Padded)
print(Tweet_Padded_Teste)

#Verificando o tamanho
len(Tweet_Padded[50]), len(Tweet_Padded[50])
len(Tweet_Padded_Teste[50]), len(Tweet_Padded_Teste[60])

print(Tweet_Padded[1100])

# Verificando as dimensões das arrays
Tweet_Padded_Teste.shape

#Criando e compilando o modelo

modelo_LSTM = tf.keras.models.Sequential()
modelo_LSTM.add(tf.keras.layers.Embedding(input_dim = Max_Palavras, output_dim = 128, input_length =Tamanho_frase))
modelo_LSTM.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
modelo_LSTM.add(tf.keras.layers.Dropout(0.2))
modelo_LSTM.add(tf.keras.layers.Dense(3, activation='softmax'))
modelo_LSTM.summary()                
modelo_LSTM.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#epoch - Epocas
modelo = modelo_LSTM.fit(Tweet_Padded, Classificacao_Treino, epochs= 15)

# Testando e avaliando o modelo
results = modelo_LSTM.evaluate(Tweet_Padded_Teste,Classificacao_Teste, verbose=1)
print(results)

# base de teste
Predicao_Teste = modelo_LSTM.predict(Tweet_Padded_Teste)
#base de treino
Predicao_Treino = modelo_LSTM.predict(Tweet_Padded)


Predicao_Test = np.argmax(Predicao_Teste, axis = 1)
Predicao_Train = np.argmax(Predicao_Treino, axis = 1)
print(Predicao_Test)
Classificacao_Teste


# Medidas de validação do modelo - Dataset de teste
#Sentimento = [0,1,2]
target_names = ['Negativo', 'Positivo', 'Neutro']
print(metrics.classification_report(Classificacao_Teste, Predicao_Test, target_names=target_names))

# Matriz de confusão
print(pd.crosstab(Classificacao_Teste, Predicao_Test, rownames = ["Sentimento Real"], 
                  colnames=["Predição LSTM"], margins=True))


# Gravar a classificacao do final do arquivo
Twitter_Teste_Dataset['LSTM'] = Predicao_Test
Twitter_Treino_Dataset['LSTM'] = Predicao_Train
frames = [Twitter_Teste_Dataset, Twitter_Treino_Dataset]
Twitter_RFB = pd.concat(frames)
Twitter_RFB

is_Neg = Twitter_RFB['Classificacao']=="Negativo"
Twitter_RFB_Neg = Twitter_RFB[is_Neg]
is_Pos = Twitter_RFB['Classificacao']=="Positivo"
Twitter_RFB_Pos = Twitter_RFB[is_Pos]

# Nuvem de palavras - Apresentação
Palavras = Twitter_RFB['Tweet']
Todos_Twitter = "".join(t for t in str(Palavras))
sw = nltk.corpus.stopwords.words('portuguese')
Newwords = ['rt', 'RT', 'voce','r$','pra','nao','sobre']
sw.extend(Newwords)
NuvemPalavras = WordCloud(background_color='black', width=1600,height=800,stopwords=sw ).generate(Todos_Twitter)
figura, ax = plt.subplots(figsize=(16,8))            
ax.imshow(NuvemPalavras, interpolation='bilinear')   
NuvemPalavras.to_file('NuvemPalavrasGeral.png',);

# Nuvem de palavras - Negativas
Palavras = Twitter_RFB_Neg['Tweet']
Todos_Twitter = "".join(t for t in str(Palavras))
NuvemPalavras = WordCloud(background_color='black', width=1600,height=800).generate(Todos_Twitter)
figura, ax = plt.subplots(figsize=(16,8))            
ax.imshow(NuvemPalavras, interpolation='bilinear')   
NuvemPalavras.to_file('NuvemPalavrasNegativos.png',);

# Nuvem de palavras - Positivas
Palavras = Twitter_RFB_Pos['Tweet']
Todos_Twitter = "".join(t for t in str(Palavras))
NuvemPalavras = WordCloud(background_color='black', width=1600,height=800).generate(Todos_Twitter)
figura, ax = plt.subplots(figsize=(16,8))            
ax.imshow(NuvemPalavras, interpolation='bilinear')   
NuvemPalavras.to_file('NuvemPalavrasPositivo.png',);

Twitter_RFB

# Capturar municipios e estados
UF = pd.read_csv("Estados_Municipios.csv")
UF = UF.rename(columns={'NOME DO MUNICÍPIO': 'Cidade'})
UF['Estado'] = 'NaN'

#Preenchendo o Nome dos Estados por extenso
            
def Estados_Extenso(uf):
    uf = str(uf).upper()

    Dic_Estados = {'AC': 'Acre','AL': 'Alagoas','AP': 'Amapá','AM': 'Amazonas','BA': 'Bahia','CE': 'Ceará','DF': 'Distrito Federal',
    'ES': 'Espírito Santo','GO': 'Goiás','MA': 'Maranhão','MT': 'Mato Grosso','MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais','PA': 'Pará','PB': 'Paraíba','PR': 'Paraná','PE': 'Pernambuco','PI': 'Piauí',
    'RJ': 'Rio de Janeiro','RN': 'Rio Grande do Norte','RS': 'Rio Grande do Sul','RO': 'Rondônia','RR': 'Roraima','SC': 'Santa Catarina','SP': 'São Paulo',
    'SE': 'Sergipe','TO': 'Tocantins'}

    return Dic_Estados[uf]

for index, row in UF.iterrows():
    UF.loc[index,'Estado'] = Estados_Extenso(row['UF'])


Twitter_RFB = Twitter_RFB.rename(columns={'User - Location': 'Cidade'})
Twitter_RFB

# Remover acentos e pontuações que permitam a comparação entre as strings
from unicodedata import normalize
def remover_acentos(txt):
    txt = str(txt)
    txt = txt.replace('/',',').replace('-',',').replace(' ,',',').replace(', ',',')
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

Twitter_RFB['Cidade'] = Twitter_RFB['Cidade'].apply(remover_acentos)
UF['Cidade'] =  UF['Cidade'].apply(remover_acentos)
UF['Estado'] =  UF['Estado'].apply(remover_acentos)

Twitter_RFB['UF'] = 'Não Informado'
Twitter_RFB

# Trabalhando os dados omissos - Preencher Cidades em branco com "Não Informado"
for index, row in Twitter_RFB.iterrows():
    if row['Cidade'] == "nan" :
        Twitter_RFB.loc[index,'Cidade'] = 'Não Informado'
Twitter_RFB

Twitter_RFB['Cidade'] = Twitter_RFB.Cidade.str.lower()
UF['Cidade'] = UF.Cidade.str.lower()
UF['Estado'] = UF.Estado.str.lower()

#Capturar UF do Tweet
for index, row in Twitter_RFB.iterrows():
    
    if row['Cidade'] == "não informado":
        print("em branco")
    else:
        Cidades = row['Cidade']
        Cidades = Cidades.split(',')
        for Cid in Cidades:
            for index2, row2 in UF.iterrows():    
                if Cid == row2['UF'].lower():
                    print("No dataset UF - "+ row2['UF'])
                    print("No dataset Twitter - "+ Twitter_RFB['Cidade'][index])
                    Twitter_RFB.loc[index,'UF'] = row2['UF']
                    break
                elif Cid == row2['Estado']:
                    print("No dataset UF - "+ row2['UF'])
                    print("No dataset Twitter - "+ Twitter_RFB['Cidade'][index])
                    Twitter_RFB.loc[index,'UF'] = row2['UF']
                    break
                elif Cid == row['Cidade']:
                    print("No dataset UF - "+ row2['UF'])
                    print("No dataset Twitter - "+ Twitter_RFB['Cidade'][index])
                    Twitter_RFB.loc[index,'UF'] = row2['UF']
                    break


Twitter_RFB
Twitter_RFB.to_csv("Twitter_LSTM_Final.csv")