import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class RedeNeural1:
    def __init__(self):
        self.model = None
        self.NUM_ATRIBUTOS = 3  # 3 atributos por classe, fixo

    def criar_modelo(self, input_shape, neuronios, camadas):
        """Cria o modelo da rede neural"""
        model = Sequential()
        model.add(Dense(neuronios, activation='relu', input_shape=input_shape))
        for _ in range(camadas - 1):
            model.add(Dense(neuronios, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def treinar(self, X, y, epocas, neuronios, camadas):
        """Treina a rede com os dados e parâmetros"""
        try:
            # Dividir em treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Criar modelo
            self.model = Sequential()
            self.model.add(Dense(neuronios, activation='relu', input_shape=(X_train.shape[1],)))
            for _ in range(camadas - 1):
                self.model.add(Dense(neuronios, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))

            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Treinar modelo
            history = self.model.fit(
                X_train, y_train,
                epochs=epocas,
                validation_data=(X_test, y_test),
                verbose=1
            )

            # Avaliar
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            acuracia = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            return {
                'acuracia': acuracia,
                'matriz_confusao': cm,
                'history': history.history
            }

        except Exception as e:
            print(f"Erro durante treinamento: {str(e)}")
            raise
