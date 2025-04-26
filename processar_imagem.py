import os
import json
import numpy as np
import pandas as pd
from PIL import Image

def carregar_intervalos_cores():
    intervalos = {}
    for arquivo in os.listdir('cores_definidas'):
        if arquivo.endswith('.json'):
            classe = arquivo.replace('.json', '')
            with open(os.path.join('cores_definidas', arquivo), 'r') as f:
                intervalos[classe] = json.load(f)
    return intervalos

def pixel_dentro_intervalo(pixel, cor_definida):
    r, g, b = pixel
    tolerancia = cor_definida['tolerancia']
    return (abs(r - cor_definida['r']) <= tolerancia and
            abs(g - cor_definida['g']) <= tolerancia and
            abs(b - cor_definida['b']) <= tolerancia)

def contar_pixels(imagem, cores_definidas):
    contagem = [0] * len(cores_definidas)
    pixels = np.array(imagem)

    for linha in pixels:
        for pixel in linha:
            for idx, cor in enumerate(cores_definidas):
                if pixel_dentro_intervalo(pixel, cor):
                    contagem[idx] += 1
                    break  # Se um pixel encaixa em um intervalo, para de testar

    return contagem

def processar_todas_imagens(pasta_uploads):
    intervalos = carregar_intervalos_cores()

    dados = []
    colunas = []

    # Gerar nomes de coluna com base na quantidade de cores
    for classe, cores in intervalos.items():
        for i in range(len(cores)):
            colunas.append(f"{classe}_cor{i+1}")
    colunas.append('Classe')

    for classe_pasta in os.listdir(pasta_uploads):
        caminho_classe = os.path.join(pasta_uploads, classe_pasta)
        if not os.path.isdir(caminho_classe):
            continue

        cores_definidas = intervalos.get(classe_pasta)
        if not cores_definidas:
            continue  # Não encontrou definição de cores para esta classe

        for img_nome in os.listdir(caminho_classe):
            img_caminho = os.path.join(caminho_classe, img_nome)
            try:
                img = Image.open(img_caminho).convert('RGB')
                contagem = contar_pixels(img, cores_definidas)
                linha = contagem + [classe_pasta]
                dados.append(linha)
            except Exception as e:
                print(f"Erro ao processar imagem {img_nome}: {str(e)}")

    df = pd.DataFrame(dados, columns=colunas)
    os.makedirs('csvs_gerados', exist_ok=True)
    df.to_csv('csvs_gerados/dados_imagens.csv', index=False)
    return 'csvs_gerados/dados_imagens.csv'
