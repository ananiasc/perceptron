import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

# import some data to play with
df_treino = pd.read_csv('dados_de_treino.csv')
X_treino = df_treino[['x', 'y']].values
y_treino = df_treino['cor'].map(lambda cor: 1 if cor == 'azul' else -1).values


df_teste = pd.read_csv('dados_de_teste.csv')
X_teste = df_teste[['x', 'y']].values
y_teste = df_teste['cor'].map(lambda cor: 1 if cor == 'azul' else -1).values


my_perceptron = Perceptron()
my_perceptron.fit(X_treino, y_treino)

ws = my_perceptron.getW()

previsoes = my_perceptron.predict(X_teste)

from sklearn.metrics import accuracy_score, confusion_matrix
acerto = accuracy_score(y_teste, previsoes)
matriz = confusion_matrix(y_teste, previsoes)