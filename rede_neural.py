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

from routes import *

def hex_para_rgb_normalizado(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b]

def processar_dados(epocas, neuronios, enlaces):
    caminho_csv = "arquivos/tabela_cores.csv"  # Nome fixo aqui também

    if not os.path.exists(caminho_csv):
        raise FileNotFoundError("Arquivo de cores não encontrado. Verifique se o arquivo foi salvo corretamente.")

    dataset = pd.read_csv("arquivos/tabela_cores.csv")

    # Converte todas as colunas de cor para arrays de números
    X = []

    for _, row in dataset.iloc[:, :-1].iterrows():
        linha_convertida = []
        for cor in row:
            linha_convertida.extend(hex_para_rgb_normalizado(cor))
        X.append(linha_convertida)

    X = np.array(X)
    y_raw = dataset.iloc[:, -1].values
    y = (y_raw == y_raw[0])   # binário, pode adaptar depois

    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2)

    # Treina usando os dados passados
    acc, cm = treinar_rede_neural(X_treinamento, y_treinamento, epocas, neuronios, enlaces)

    novo_treinamento = Treinamento(
       epocas=epocas,
        neuronios=neuronios,
        enlaces=enlaces,
        resultado=f"{acc:.4f}",        # Salvar como string, se quiser salvar como número, mude o tipo no model
        usuario_id=1
    )
    db.session.add(novo_treinamento)
    db.session.commit()

    return acc, cm


def get_ultimo_treinamento():
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def treinar_rede_neural(X, y, epocas, neuronios, enlaces):
    # Configuração do modelo da rede neural
    rede_neural = tf.keras.models.Sequential()
    rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu', input_shape=(X.shape[1],)))
    
    # Adicionando camadas ocultas
    for _ in range(enlaces - 1):
        rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu'))

    # Camada de saída
    rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compilação do modelo
    rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinamento
    historico = rede_neural.fit(X, y, epochs=epocas, validation_split=0.1)

    # Fazendo previsões
    previsoes = rede_neural.predict(X)
    previsoes = (previsoes > 0.5)  # Classificando como 0 ou 1

    # Calculando a acurácia
    acc = accuracy_score(y, previsoes)

    # Calculando a matriz de confusão
    cm = confusion_matrix(y, previsoes)

    modelo_path = "modelo_cnn_salvo.h5"
    rede_neural.save(modelo_path)

    return acc, cm
