import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import processar_imagem

from models import Treinamento
from models import db

from routes import *

def hex_para_rgb_normalizado(hex_color):
    """Converte cores HEX para RGB normalizado entre 0 e 1."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b]

def comparar_cores(cor_pixel, atributo, tolerancia=15):
    """Compara se a cor do pixel é similar ao atributo dentro de uma tolerância de +-15 unidades RGB."""
    r, g, b = cor_pixel
    r_atributo, g_atributo, b_atributo = atributo
    return (abs(r - r_atributo) <= tolerancia / 255.0 and
            abs(g - g_atributo) <= tolerancia / 255.0 and
            abs(b - b_atributo) <= tolerancia / 255.0)

def processar_imagens():
    if not ultimo_upload_caminho:
        flash('Nenhum upload foi feito ainda.')
        return redirect(url_for('principal.index'))

    # Carregar os atributos escolhidos pelo usuário
    atributos1 = request.form.getlist('atributos1[]')
    atributos2 = request.form.getlist('atributos2[]')

    # Caminhos das pastas de treinamento e teste
    caminho_treinamento_classe1 = os.path.join(ultimo_upload_caminho, "classe1", "treinamento")
    caminho_treinamento_classe2 = os.path.join(ultimo_upload_caminho, "classe2", "treinamento")
    caminho_teste_classe1 = os.path.join(ultimo_upload_caminho, "classe1", "teste")
    caminho_teste_classe2 = os.path.join(ultimo_upload_caminho, "classe2", "teste")

    # Verificar se os diretórios existem
    if not os.path.exists(caminho_treinamento_classe1):
        flash(f"Diretório {caminho_treinamento_classe1} não encontrado.")
        return redirect(url_for('principal.index'))

    if not os.path.exists(caminho_treinamento_classe2):
        flash(f"Diretório {caminho_treinamento_classe2} não encontrado.")
        return redirect(url_for('principal.index'))

    if not os.path.exists(caminho_teste_classe1):
        flash(f"Diretório {caminho_teste_classe1} não encontrado.")
        return redirect(url_for('principal.index'))

    if not os.path.exists(caminho_teste_classe2):
        flash(f"Diretório {caminho_teste_classe2} não encontrado.")
        return redirect(url_for('principal.index'))

    # Processar imagens de treinamento e teste
    imagens_treinamento_classe1 = [os.path.join(caminho_treinamento_classe1, f) for f in os.listdir(caminho_treinamento_classe1)]
    imagens_treinamento_classe2 = [os.path.join(caminho_treinamento_classe2, f) for f in os.listdir(caminho_treinamento_classe2)]
    imagens_teste_classe1 = [os.path.join(caminho_teste_classe1, f) for f in os.listdir(caminho_teste_classe1)]
    imagens_teste_classe2 = [os.path.join(caminho_teste_classe2, f) for f in os.listdir(caminho_teste_classe2)]

    # Unir todas as imagens de treinamento e teste
    imagens_treinamento = imagens_treinamento_classe1 + imagens_treinamento_classe2
    imagens_teste = imagens_teste_classe1 + imagens_teste_classe2

    # Agora, processe as imagens aqui
    salvar_resultados_csv(imagens_treinamento, atributos1, atributos2)
    return "Imagens processadas com sucesso!"

def salvar_resultados_csv(imagens_treinamento, atributos1, atributos2, caminho_csv="caminho_csv"):
    """Processa todas as imagens de treinamento e salva os resultados no CSV."""
    resultados = []
    
    for imagem_path in imagens_treinamento:
        contagem_classe1, contagem_classe2 = processar_imagem(imagem_path, atributos1, atributos2)
        soma_classe1 = sum(contagem_classe1)
        soma_classe2 = sum(contagem_classe2)
        
        # Verifica a classe da imagem com base na soma dos atributos
        classe = 1 if soma_classe1 > soma_classe2 else 2
        resultados.append(contagem_classe1 + contagem_classe2 + [classe])
    
    # Salva os resultados em um DataFrame e depois em um CSV
    colunas = ['atributo1_classe1', 'atributo2_classe1', 'atributo3_classe1', 'atributo1_classe2', 'atributo2_classe2', 'atributo3_classe2', 'classe']
    df_resultados = pd.DataFrame(resultados, columns=colunas)
    df_resultados.to_csv(caminho_csv, index=False)

def treinar_rede_neural(X, y, epocas, neuronios, enlaces):
    """Treina a rede neural com base nos dados de entrada e retorna a acurácia e a matriz de confusão."""
    rede_neural = tf.keras.models.Sequential()
    rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu', input_shape=(X.shape[1],)))
    
    # Camadas intermediárias
    for _ in range(enlaces - 1):
        rede_neural.add(tf.keras.layers.Dense(units=neuronios, activation='relu'))

    # Camada de saída
    rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compilação e treinamento da rede
    rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    historico = rede_neural.fit(X, y, epochs=epocas, validation_split=0.1)

    # Predições e avaliação
    previsoes = rede_neural.predict(X)
    previsoes = (previsoes > 0.5)  # Transformando para binário

    acc = accuracy_score(y, previsoes)
    cm = confusion_matrix(y, previsoes)

    # Salvando o modelo treinado
    modelo_path = "modelo_cnn_salvo.h5"
    rede_neural.save(modelo_path)

    return acc, cm

def processar_dados(epocas, neuronios, enlaces):
    """Processa os dados de entrada e treina a rede neural, retornando a acurácia e a matriz de confusão."""
    caminho_csv = "caminho_csv"

    if not os.path.exists(caminho_csv):
        raise FileNotFoundError("Arquivo de cores não encontrado. Verifique se o arquivo foi salvo corretamente.")

    # Carrega as cores do CSV
    dataset = pd.read_csv(caminho_csv)
    atributos1 = [hex_para_rgb_normalizado(cor) for cor in dataset.iloc[0, :-1]]  # Primeira linha de cores para classe 1
    atributos2 = [hex_para_rgb_normalizado(cor) for cor in dataset.iloc[1, :-1]]  # Segunda linha de cores para classe 2

    # Usando o último caminho de upload
    if not ultimo_upload_caminho:
        raise ValueError("Nenhum upload foi feito ainda!")

    caminho_imagens_treinamento = os.path.join(ultimo_upload_caminho, "classe1", "treinamento")  # Caminho dinâmico
    imagens_treinamento = [os.path.join(caminho_imagens_treinamento, f) for f in os.listdir(caminho_imagens_treinamento)]

    # Processa as imagens e salva os resultados
    salvar_resultados_csv(imagens_treinamento, atributos1, atributos2)

    # Agora treina a rede neural com os dados extraídos
    X = pd.read_csv(caminho_csv).values[:, :-1]
    y = pd.read_csv(caminho_csv).values[:, -1]
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2)

    acc, cm = treinar_rede_neural(X_treinamento, y_treinamento, epocas, neuronios, enlaces)

    novo_treinamento = Treinamento(
        epocas=epocas,
        neuronios=neuronios,
        enlaces=enlaces,
        resultado=f"{acc:.4f}",
        usuario_id=1
    )
    db.session.add(novo_treinamento)
    db.session.commit()

    return acc, cm
