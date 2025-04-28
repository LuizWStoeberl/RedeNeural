import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def treinar_rede_neural(caminho_csv, epocas=100, neuronios=64, camadas_ocultas=3):
    """Treina uma rede neural com os parâmetros especificados"""
    
    # Carregar dados
    df = pd.read_csv(caminho_csv)
    X = df.drop('classe', axis=1).values
    y = df['classe'].values - 1  # Converter para 0 e 1
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Construir modelo
    model = Sequential()
    model.add(Dense(neuronios, activation='relu', input_shape=(X_train.shape[1],)))
    
    for _ in range(camadas_ocultas - 1):
        model.add(Dense(neuronios, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Treinar
    history = model.fit(
        X_train, y_train,
        epochs=epocas,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Avaliar
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    return {
        'acuracia': acuracia,
        'matriz_confusao': matriz_confusao.tolist(),
        'epocas': epocas,
        'neuronios': neuronios,
        'camadas': camadas_ocultas,
        'history': history.history
    }