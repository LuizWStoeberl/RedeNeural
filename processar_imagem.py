import os
import json
import csv
import numpy as np
import pandas as pd
from PIL import Image
from models import IntervaloCor

CORES_FOLDER = 'cores_definidas'
os.makedirs(CORES_FOLDER, exist_ok=True)

def hex_para_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b

def identificar_classe_pixel(r, g, b, intervalos):
    for intervalo in intervalos:
        if (intervalo.r_min <= r <= intervalo.r_max and
            intervalo.g_min <= g <= intervalo.g_max and
            intervalo.b_min <= b <= intervalo.b_max):
            return intervalo.classe
    return None

def converter_imagens_para_csv(pasta_imagens='arquivoUsuario', saida_csv='arquivos/dados.csv'):
    intervalos = IntervaloCor.query.all()
    dados = []

    for subdir, dirs, files in os.walk(pasta_imagens):
        for file in files:
            caminho_imagem = os.path.join(subdir, file)
            if caminho_imagem.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = Image.open(caminho_imagem).convert('RGB')
                img_array = np.array(img)
                for linha in img_array:
                    for pixel in linha:
                        r, g, b = pixel
                        classe = identificar_classe_pixel(r, g, b, intervalos)
                        if classe:
                            dados.append([r, g, b, classe])

    with open(saida_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['R', 'G', 'B', 'Classe'])
        writer.writerows(dados)

    return saida_csv

def carregar_intervalos_cores():
    intervalos = {}
    for arquivo in os.listdir(CORES_FOLDER):
        if arquivo.endswith('.json'):
            classe = arquivo.replace('.json', '')
            with open(os.path.join(CORES_FOLDER, arquivo), 'r') as f:
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
                    break
    return contagem

def processar_todas_imagens(pasta_uploads):
    intervalos = carregar_intervalos_cores()
    dados = []
    colunas = []

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
            continue

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