from PIL import Image
import numpy as np
import os
import csv
from models import IntervaloCor

def identificar_classe_pixel(r, g, b, intervalos):
    for intervalo in intervalos:
        if (intervalo.r_min <= r <= intervalo.r_max and
            intervalo.g_min <= g <= intervalo.g_max and
            intervalo.b_min <= b <= intervalo.b_max):
            return intervalo.classe
    return None  # Pixel que não pertence a nenhum intervalo

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
                        if classe:  # Só salva se identificar a classe
                            dados.append([r, g, b, classe])

    # Salvar no CSV
    with open(saida_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['R', 'G', 'B', 'Classe'])  # Cabeçalho
        writer.writerows(dados)

    return saida_csv
