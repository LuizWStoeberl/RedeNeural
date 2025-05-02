import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from datetime import datetime
import os

class RedeNeural1:
    def __init__(self):
        self.model = None
        # Define as funções de treino uma única vez
        self.train_step = None
        self.test_step = None

    @tf.function(reduce_retracing=True)  # ← Adicione esta linha
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.model.loss(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _criar_modelo(self, input_shape, neuronios, camadas):
        model = Sequential()

        # Remover camadas convolucionais, já que o modelo agora trabalha com dados tabulares
        model.add(Dense(neuronios, input_dim=input_shape, activation='relu'))  # Primeira camada densa

        # Camadas ocultas densas
        for _ in range(camadas):
            model.add(Dense(neuronios, activation='relu'))

        # Camada de saída
        model.add(Dense(1, activation='sigmoid'))  # Saída binária, ajuste conforme necessário

        # Compilação do modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def treinar(self, X, y, epocas, neuronios, camadas):
        # 1. Garantir que os dados são numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # 2. Dividir os dados apenas uma vez
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Reiniciar o modelo completamente
        self.model = None
        tf.keras.backend.clear_session()  # Limpa a sessão do TF
        
        # 4. Criar novo modelo
        self.model = self._criar_modelo(X.shape[1], neuronios, camadas)
        
        # 5. Callback para evitar treinamento prolongado
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        # 6. Treinar apenas uma vez
        history = self.model.fit(
            X_train, y_train,
            epochs=epocas,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        # 7. Salvar o modelo com timestamp no nome e garantir que a pasta exista
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_modelo = os.path.join('modelos_salvos', f'modeloopp_{timestamp}')
        os.makedirs(pasta_modelo, exist_ok=True)

        modelo_path = os.path.join(pasta_modelo, 'modelo.h5')
        self.model.save(modelo_path)

        # Salvar classes (exemplo fixo)
        class_indices = {0: 'Classe 0', 1: 'Classe 1'}
        with open(os.path.join(pasta_modelo, 'classes.json'), 'w') as f:
            json.dump(class_indices, f)

        # Salvar classes (exemplo fixo)
        with open(f'modelos_salvos/classes_{timestamp}.json', 'w') as f:
            json.dump(class_indices, f)

        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Retorne como dicionário (ou tupla se preferir)
        return {
            'acuracia': acc,
            'matriz_confusao': cm.tolist(),  # Converte numpy array para lista
            'history': history.history
        }
    
        


