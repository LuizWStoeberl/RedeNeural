import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class RedeNeural1:
    def __init__(self):
        self.model = None
        self.NUM_ATRIBUTOS = 3  # Definindo fixamente 3 atributos por classe

    def processar_imagens(self, imagens_treinamento, atributos1, atributos2, caminho_csv):
        """Processa imagens e gera dataset CSV com 3 atributos para cada classe"""
        # Validação dos atributos
        if len(atributos1) != self.NUM_ATRIBUTOS or len(atributos2) != self.NUM_ATRIBUTOS:
            raise ValueError(f"Deve fornecer exatamente {self.NUM_ATRIBUTOS} atributos para cada classe")

        resultados = []
        for imagem_path in imagens_treinamento:
            contagem_classe1, contagem_classe2 = self._processar_imagem(imagem_path, atributos1, atributos2)
            linha = contagem_classe1 + contagem_classe2 + [1 if sum(contagem_classe1) > sum(contagem_classe2) else 2]
            resultados.append(linha)
        
        # Nomes das colunas fixos para 3 atributos
        colunas = [
            'atributo1_classe1', 'atributo2_classe1', 'atributo3_classe1',
            'atributo1_classe2', 'atributo2_classe2', 'atributo3_classe2',
            'classe'
        ]
        
        df = pd.DataFrame(resultados, columns=colunas)
        os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
        df.to_csv(caminho_csv, index=False)
        return caminho_csv

    # ... (restante do código permanece igual)

    def _comparar_cores(self, cor_pixel, atributo, tolerancia):
        """Compara cores com tolerância"""
        r, g, b = cor_pixel
        r_attr, g_attr, b_attr = atributo
        return (abs(r - r_attr) <= tolerancia and
                abs(g - g_attr) <= tolerancia and
                abs(b - b_attr) <= tolerancia)

    def criar_modelo(self, input_shape, neuronios, camadas):
        """Cria modelo da rede neural"""
        model = Sequential()
        model.add(Dense(neuronios, activation='relu', input_shape=input_shape))
        
        for _ in range(camadas - 1):
            model.add(Dense(neuronios, activation='relu'))
            
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def treinar(self, X, y, epocas, neuronios, camadas):
        """Treina o modelo com os parâmetros especificados"""
        try:
            # 1. Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 2. Criar modelo
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
            
            # 3. Treinar
            history = self.model.fit(
                X_train, y_train,
                epochs=epocas,
                validation_data=(X_test, y_test),
                verbose=1  # Agora mostra progresso
            )
            
            # 4. Avaliar
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