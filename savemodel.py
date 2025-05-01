import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from models import Treinamento, db

MODELO_SALVO_PATH = 'modelos_salvos/modelo_rede_neural.keras'
os.makedirs('modelos_salvos', exist_ok=True)

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
    y = LabelEncoder().fit_transform(y_raw) # Convertendo para binário (True/False)

    config = get_ultimo_treinamento()
    if not config:
        raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados")

    # Parâmetros do usuário
    epocas = config.epocas
    neuronios = config.neuronios
    enlaces = config.enlaces
    teste_size = getattr(config, 'teste_size', 0.2)  # Se o campo existir no banco, usa. Senão, 20% default

    # Divisão dos dados
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=teste_size)

    # Montagem da Rede Neural
    rede_neural = tf.keras.models.Sequential()
    rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu', input_shape=(X.shape[1],)))

    for _ in range(enlaces - 1):
        rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu'))

    rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compilação e Treinamento
    rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    historico = rede_neural.fit(X_treinamento, y_treinamento, epochs=epocas, validation_split=0.1, verbose=0)

    # Avaliação
    previsoes = rede_neural.predict(X_teste)
    previsoes = (previsoes > 0.5)

    acc = accuracy_score(y_teste, previsoes)
    cm = confusion_matrix(y_teste, previsoes)

    config.resultado = f"Accuracy: {acc:.4f}"
    db.session.commit()

    # Salva o modelo treinado
    rede_neural.save(MODELO_SALVO_PATH)

    return acc, cm

def classificar_nova_imagem(caminho_csv_imagem):
    # Carrega o modelo salvo
    if not os.path.exists(MODELO_SALVO_PATH):
        raise Exception("Modelo treinado não encontrado. Treine a rede neural primeiro!")

    modelo = tf.keras.models.load_model(MODELO_SALVO_PATH)

    # Lê o arquivo CSV gerado a partir da nova imagem
    dados = pd.read_csv(caminho_csv_imagem)
    X_novo = dados.values  # Sem labels

    previsoes = modelo.predict(X_novo)
    previsoes = (previsoes > 0.5)

    return previsoes