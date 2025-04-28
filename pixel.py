import os
import numpy as np
import pandas as pd
from PIL import Image

def hex_para_rgb_normalizado(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r / 255.0, g / 255.0, b / 255.0]

def comparar_cores(cor_pixel, atributo, tolerancia=15):
    r, g, b = cor_pixel
    r_attr, g_attr, b_attr = atributo
    return (abs(r - r_attr) <= tolerancia / 255.0 and
            abs(g - g_attr) <= tolerancia / 255.0 and
            abs(b - b_attr) <= tolerancia / 255.0)

def processar_imagem(imagem_path, atributos1, atributos2):
    try:
        img = Image.open(imagem_path).convert('RGB')
        pixels = np.array(img)
        contagem_classe1 = [0] * len(atributos1)
        contagem_classe2 = [0] * len(atributos2)

        for linha in pixels:
            for pixel in linha:
                r, g, b = [v / 255.0 for v in pixel]
                for i, attr in enumerate(atributos1):
                    if comparar_cores((r, g, b), attr):
                        contagem_classe1[i] += 1
                        break
                for j, attr in enumerate(atributos2):
                    if comparar_cores((r, g, b), attr):
                        contagem_classe2[j] += 1
                        break

        return contagem_classe1, contagem_classe2
    except Exception as e:
        print(f"Erro ao processar {imagem_path}: {e}")
        return [0]*len(atributos1), [0]*len(atributos2)

def converter_imagens_para_csv(imagens_treinamento, atributos1, atributos2, caminho_csv="arquivos/tabela_cores.csv"):
    resultados = []
    for imagem_path in imagens_treinamento:
        contagem_classe1, contagem_classe2 = processar_imagem(imagem_path, atributos1, atributos2)
        soma1 = sum(contagem_classe1)
        soma2 = sum(contagem_classe2)
        classe = 1 if soma1 > soma2 else 2
        resultados.append(contagem_classe1 + contagem_classe2 + [classe])
    
    colunas = [f'atributo{i+1}_classe1' for i in range(3)] + \
              [f'atributo{i+1}_classe2' for i in range(3)] + ['classe']
    df_resultados = pd.DataFrame(resultados, columns=colunas)
    os.makedirs('arquivos', exist_ok=True)
    df_resultados.to_csv(caminho_csv, index=False)
    return caminho_csv