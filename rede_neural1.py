import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

    def treinar(self, X, y, epocas, neuronios, camadas):
        # ... (código existente)
        
        # Cria o modelo uma única vez
        if self.model is None:
            self.model = self._criar_modelo(X.shape[1:], neuronios, camadas)
            self.train_step = self._train_step  # Atribui a função otimizada

        for epoch in range(epocas):
            loss = self.train_step(X_train, y_train)  # Usa a função pré-compilada
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
    

def treinar(self, X, y, epocas, neuronios, camadas):
    self.model.save('modelos_salvos/modelo_rede1.keras')  # Salva o modelo
    return { ... }

