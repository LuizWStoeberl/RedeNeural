import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import  db

MODELO_SALVO_PATH = 'modelos_salvos/modelo_cnn.h5'
os.makedirs('modelos_salvos', exist_ok=True)

def get_ultimo_treinamento():
    from models import Treinamento
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def treinar_rede_neural_cnn():
    from models import Treinamento
    arquivos_dir = "arquivos"
    arquivos_csv = [f for f in os.listdir(arquivos_dir) if f.endswith(".csv")]
    arquivos_csv.sort(reverse=True)

    if not arquivos_csv:
        raise Exception("Nenhum arquivo CSV encontrado.")

    caminho_csv = os.path.join(arquivos_dir, arquivos_csv[0])
    dataset = pd.read_csv(caminho_csv)

    X = dataset.iloc[:, :-1].values
    y_raw = dataset.iloc[:, -1].values
    y = (y_raw == y_raw[0])  # Convertendo para binário (True/False)

    config = get_ultimo_treinamento()
    if not config:
        raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados")

    # Parâmetros do usuário
    epocas = config.epocas
    neuronios = config.neuronios
    camadas_convolucionais = config.camadas_convolucionais  # Número de camadas convolucionais
    tamanho_imagem = (150, 150)  # Definir o tamanho das imagens, se necessário
    teste_size = getattr(config, 'teste_size', 0.2)  # Se o campo existir no banco, usa. Senão, 20% default

    # Divisão dos dados
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=teste_size)

    # Usando ImageDataGenerator para pré-processamento de imagens
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'diretorio_de_imagens_treinamento',  # Substitua com o diretório real
        target_size=tamanho_imagem,
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        'diretorio_de_imagens_teste',  # Substitua com o diretório real
        target_size=tamanho_imagem,
        batch_size=32,
        class_mode='binary'
    )

    # Arquitetura da Rede Neural Convolucional
    modelo_cnn = tf.keras.models.Sequential()

    # Camadas convolucionais
    modelo_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    modelo_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    for _ in range(camadas_convolucionais - 1):
        modelo_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        modelo_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Camada densa final
    modelo_cnn.add(tf.keras.layers.Flatten())
    modelo_cnn.add(tf.keras.layers.Dense(128, activation='relu'))
    modelo_cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compilação e Treinamento
    modelo_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    historico = modelo_cnn.fit(
        train_generator,
        epochs=epocas,
        validation_data=validation_generator
    )

    # Avaliação
    acc = historico.history['accuracy'][-1]
    cm = confusion_matrix(y_teste, modelo_cnn.predict(X_teste) > 0.5)

    # Salva o modelo treinado
    modelo_cnn.save(MODELO_SALVO_PATH)

    return acc, cm

def classificar_nova_imagem_cnn(caminho_imagem):
    # Carrega o modelo salvo
    if not os.path.exists(MODELO_SALVO_PATH):
        raise Exception("Modelo treinado não encontrado. Treine a rede neural primeiro!")

    modelo_cnn = tf.keras.models.load_model(MODELO_SALVO_PATH)

    # Pré-processamento da imagem
    img = tf.keras.preprocessing.image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra para o batch

    previsao = modelo_cnn.predict(img_array)
    return previsao
