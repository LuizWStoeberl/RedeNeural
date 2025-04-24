import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from models import Treinamento, db

def get_ultimo_treinamento():
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def treinar_rede_cnn():
    imagens_dir = "imagens"
    if not os.path.exists(imagens_dir):
        raise Exception("Diretório de imagens não encontrado")

    config = get_ultimo_treinamento()
    if not config:
        raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados")

    epocas = config.epocas
    neuronios = config.neuronios
    enlaces = config.enlaces

    gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)
    base_treinamento = gerador_treinamento.flow_from_directory(
        os.path.join(imagens_dir, 'training_set'),
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

    gerador_teste = ImageDataGenerator(rescale=1./255)
    base_teste = gerador_teste.flow_from_directory(
        os.path.join(imagens_dir, 'test_set'),
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical',
        shuffle=False)

    rede_neural = Sequential()
    rede_neural.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
    rede_neural.add(Conv2D(32, (3, 3), activation='relu'))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
    rede_neural.add(Flatten())

    for _ in range(enlaces):
        rede_neural.add(Dense(units=neuronios, activation='relu'))

    num_classes = base_treinamento.num_classes
    rede_neural.add(Dense(units=num_classes, activation='softmax'))

    rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rede_neural.fit(base_treinamento, epochs=epocas, validation_data=base_teste)

    previsoes = rede_neural.predict(base_teste)
    previsoes_classes = np.argmax(previsoes, axis=1)
    verdadeiros = base_teste.classes

    acc = accuracy_score(verdadeiros, previsoes_classes)
    cm = confusion_matrix(verdadeiros, previsoes_classes)

    config.resultado = f"CNN Accuracy: {acc:.4f}"
    db.session.commit()

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.savefig("static/matriz_cnn.png")

    return acc, cm
