import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import Treinamento
from models import db

def get_ultimo_treinamento():
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def treinar_rede_neural():
    arquivos_dir = "arquivos"
    arquivos_csv = [f for f in os.listdir(arquivos_dir) if f.endswith(".csv")]
    arquivos_csv.sort(reverse=True)

    if not arquivos_csv:
        raise Exception("Nenhum arquivo CSV encontrado.")

    caminho_csv = os.path.join(arquivos_dir, arquivos_csv[0])
    dataset = pd.read_csv(caminho_csv)

    X = dataset.iloc[:, :-1].values
    y_raw = dataset.iloc[:, -1].values
    y = (y_raw == y_raw[0])  # Binário: classe igual à primeira = True

    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2)

    config = get_ultimo_treinamento()
    if not config:
        raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados")

    epocas = config.epocas
    neuronios = config.neuronios
    enlaces = config.enlaces

    rede_neural = tf.keras.models.Sequential()
    rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu', input_shape=(X.shape[1],)))

    for _ in range(enlaces - 1):
        rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu'))

    rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    historico = rede_neural.fit(X_treinamento, y_treinamento, epochs=epocas, validation_split=0.1)

    previsoes = rede_neural.predict(X_teste)
    previsoes = (previsoes > 0.5)

    acc = accuracy_score(y_teste, previsoes)
    cm = confusion_matrix(y_teste, previsoes)

    config.resultado = f"Accuracy: {acc:.4f}"
    db.session.commit()

    return acc, cm 
