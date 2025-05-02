import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import db
import json
from datetime import datetime

# Caminho para salvar o modelo (arquivo .keras)
MODELO_SALVO_PATH = 'modelos_salvos/modelo_cnn.keras'
os.makedirs('modelos_salvos', exist_ok=True)

def get_ultimo_treinamento():
    from models import Treinamento
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def encontrar_ultimo_upload(base_path='arquivosRede2'):
    pastas = [p for p in os.listdir(base_path) if p.startswith('teste')]
    pastas.sort(reverse=True)
    if pastas:
        return os.path.join(base_path, pastas[0])
    else:
        raise Exception("Nenhuma pasta de upload encontrada.")

def carregar_modelo_cnn():
    """Carrega o modelo CNN salvo"""
    if os.path.exists(MODELO_SALVO_PATH):
        return load_model(MODELO_SALVO_PATH)
    else:
        raise FileNotFoundError("Modelo CNN não encontrado.")

def treinar_rede_neural_cnn():
    try:
        # 1. Localizar as pastas automaticamente
        pasta_base = encontrar_ultimo_upload()
        caminho_treinamento = os.path.join(pasta_base, 'treinamento2')
        caminho_teste = os.path.join(pasta_base, 'teste2')

        print("Carregando imagens de:", caminho_treinamento)
        print("Iniciando treinamento...")

        # 2. Configuração de pré-processamento
        tamanho_imagem = (150, 150)
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            caminho_treinamento,
            target_size=tamanho_imagem,
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = test_datagen.flow_from_directory(
            caminho_teste,
            target_size=tamanho_imagem,
            batch_size=32,
            class_mode='binary'
        )

        # 3. Recuperar configuração de treinamento do banco
        config = get_ultimo_treinamento()
        if not config:
            raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados.")

        epocas = config.epocas
        neuronios = config.neuronios
        camadas_convolucionais = config.enlaces

        # 4. Construção da CNN
        modelo_cnn = Sequential()

        # Primeira camada convolucional
        modelo_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        modelo_cnn.add(MaxPooling2D((2, 2)))

        # Demais camadas convolucionais
        for _ in range(camadas_convolucionais - 1):
            modelo_cnn.add(Conv2D(64, (3, 3), activation='relu'))
            modelo_cnn.add(MaxPooling2D((2, 2)))

        # Camadas finais
        modelo_cnn.add(Flatten())
        modelo_cnn.add(Dense(neuronios, activation='relu'))
        modelo_cnn.add(Dense(1, activation='sigmoid'))

        # 5. Compilar e treinar
        modelo_cnn.compile(optimizer='adam', 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])

        historico = modelo_cnn.fit(
            train_generator,
            epochs=epocas,
            validation_data=validation_generator
        )

        with open(f'modelos_salvos/classes_{config.id}.json', 'w') as f:
            json.dump(train_generator.class_indices, f)

        # 6. Avaliação
        acc = historico.history['accuracy'][-1]
        val_acc = historico.history['val_accuracy'][-1]

        # Previsões para matriz de confusão
        validation_generator = test_datagen.flow_from_directory(
            caminho_teste,
            target_size=tamanho_imagem,
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )

        y_true = validation_generator.classes
        y_pred = modelo_cnn.predict(validation_generator)
        y_pred_classes = (y_pred > 0.5).astype("int32")
        cm = confusion_matrix(y_true, y_pred_classes)

       # 7. Salvar modelo com timestamp no nome
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_path = f'modelos_salvos/cnn_model_{timestamp}.h5'
        modelo_cnn.save(modelo_path)

        return {
            'acuracia': acc,
            'val_acuracia': val_acc,
            'matriz_confusao': cm.tolist(),
            'epocas': epocas,
            'neuronios': neuronios,
            'camadas': camadas_convolucionais
        }

    except Exception as e:
        print(f"Erro durante o treinamento CNN: {str(e)}")
        raise e