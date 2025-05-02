from flask import Blueprint,request, redirect, url_for, render_template, flash, session, Flask, jsonify
import random
import shutil
import time
import jsonify
from models import *
from datetime import datetime
import os
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from rede_neural1 import RedeNeural1
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from cnn_model import *
import json 
from tensorflow.keras.models import load_model
from pathlib import Path

# Configurações
bp = Blueprint("routes", __name__)


MODEL_PREFIXES = {
    'OPP': 'modeloopp_',
    'PADRAO': 'modelopp_'  # Ou outro prefixo para o segundo tipo
}

ARQUIVOSREDE1_DIR = 'arquivosRede1'
os.makedirs(ARQUIVOSREDE1_DIR, exist_ok=True)

UPLOAD_FOLDER = 'arquivosRede2'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

diretorio_modelos = 'modelos_salvos'
for arquivo in os.listdir(diretorio_modelos):
    if arquivo.endswith('.h5') or arquivo.endswith('.keras'):
        caminho_modelo = os.path.join(diretorio_modelos, arquivo)
        modelo = load_model(caminho_modelo)
        break

# Helpers

def criar_pasta_segura(base_dir, prefixo):
    """Cria pasta com nome único de forma segura"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_pasta = f"{prefixo}_{timestamp}"
    caminho = os.path.join(base_dir, nome_pasta)
    os.makedirs(caminho, exist_ok=True)
    return caminho

def salvar_arquivos(arquivos, pasta_destino):
    os.makedirs(pasta_destino, exist_ok=True)
    for arquivo in arquivos:
        if arquivo.filename.strip():
            filename = secure_filename(arquivo.filename)
            destino = os.path.join(pasta_destino, filename)
            print(f"Salvando: {destino}")
            arquivo.save(destino)


def hex_para_rgb_normalizado(hex_color):
    """Converte cor HEX para RGB normalizado (valores entre 0 e 1)"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Formato inválido de cor: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def obter_atributos_rgb_normalizados(request_form):
    """Extrai e converte os atributos de cor do formulário HTML"""
    atributos1 = request_form.getlist('atributos1[]')
    atributos2 = request_form.getlist('atributos2[]')

    if len(atributos1) != 3 or len(atributos2) != 3:
        raise ValueError("Cada classe deve conter exatamente 3 cores.")

    try:
        atributos1_rgb = [hex_para_rgb_normalizado(cor) for cor in atributos1]
        atributos2_rgb = [hex_para_rgb_normalizado(cor) for cor in atributos2]
    except Exception as e:
        print(f"Erro ao converter cores: {e}")

    return atributos1_rgb, atributos2_rgb

def comparar_cores(cor_pixel, atributo, tolerancia=50):
    r, g, b = cor_pixel
    r_attr, g_attr, b_attr = atributo
    distancia = ((r - r_attr) ** 2 + (g - g_attr) ** 2 + (b - b_attr) ** 2) ** 0.5
    return distancia <= tolerancia / 255.0


def distribuir_arquivos_treinamento_teste(pasta_origem):
    """Distribui arquivos entre treinamento e teste (80/20)"""
    pasta_treinamento = os.path.join(pasta_origem, "treinamento")
    pasta_teste = os.path.join(pasta_origem, "teste")
    
    for classe in ["Classe1", "Classe2"]:
        pasta_classe = os.path.join(pasta_origem, classe)
        if not os.path.exists(pasta_classe):
            continue
            
        arquivos = [
            f for f in os.listdir(pasta_classe)
            if os.path.isfile(os.path.join(pasta_classe, f)) and
            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        
        random.shuffle(arquivos)
        div = int(0.8 * len(arquivos))
        
        # Mover para treinamento
        os.makedirs(os.path.join(pasta_treinamento, classe), exist_ok=True)
        for f in arquivos[:div]:
            shutil.move(
                os.path.join(pasta_classe, f),
                os.path.join(pasta_treinamento, classe, f)
            )
        
        # Mover para teste
        os.makedirs(os.path.join(pasta_teste, classe), exist_ok=True)
        for f in arquivos[div:]:
            shutil.move(
                os.path.join(pasta_classe, f),
                os.path.join(pasta_teste, classe, f)
            )

def processar_imagem(imagem_path, atributos1, atributos2):
    """Processa uma imagem, contando os pixels que correspondem aos atributos das classes"""
    try:
        img = Image.open(imagem_path).convert('RGB')
        pixels = np.array(img)

        contagem_classe1 = [0] * len(atributos1)
        contagem_classe2 = [0] * len(atributos2)

        for linha in pixels:
            for pixel in linha:
                r, g, b = [v / 255.0 for v in pixel]
                correspondeu = False

                # Verificar atributos da Classe 1
                for i, attr in enumerate(atributos1):
                    if comparar_cores((r, g, b), attr):
                        contagem_classe1[i] += 1
                        correspondeu = True
                        break  # já classificou como classe 1

                # Se não correspondeu à Classe 1, verifica Classe 2
                if not correspondeu:
                    for j, attr in enumerate(atributos2):
                        if comparar_cores((r, g, b), attr):
                            contagem_classe2[j] += 1
                            break  # já classificou como classe 2

        return contagem_classe1, contagem_classe2

    except Exception as e:
        print(f"Erro ao processar {imagem_path}: {e}")
        return [0] * len(atributos1), [0] * len(atributos2)

    except Exception as e:
        print(f"Erro ao processar {imagem_path}: {e}")
        return [0] * len(atributos1), [0] * len(atributos2)

def converter_imagens_para_csv(imagens_treinamento, atributos1, atributos2, caminho_csv):
    """Converte imagens de treinamento para CSV, tratando corretamente os atributos conforme a classe da imagem."""
    resultados = []

    for imagem_path in imagens_treinamento:
        # Detecta a classe real com base no caminho
        if "Classe1" in imagem_path:
            atributos_classe1 = atributos1
            atributos_classe2 = atributos2
            classe_real = "Classe1"
        else:
            atributos_classe1 = atributos2
            atributos_classe2 = atributos1
            classe_real = "Classe2"

        # Processa a imagem com os atributos organizados corretamente
        contagem_classe1, contagem_classe2 = processar_imagem(imagem_path, atributos_classe1, atributos_classe2)
        soma1 = sum(contagem_classe1)
        soma2 = sum(contagem_classe2)

        # Atribuir a classe com base na soma maior
        if soma1 > soma2:
            classe = "Classe1"  # Classe com mais pixels da cor
        elif soma2 > soma1:
            classe = "Classe2"
        else:
            classe = classe_real  # Se as somas forem iguais, mantemos a classe real

        resultados.append(contagem_classe1 + contagem_classe2 + [classe])

    # Geração dos nomes de coluna (genéricos, ou você pode substituí-los por nomes semânticos)
    colunas = [f'atributo{i+1}_classe1' for i in range(len(atributos1))] + \
              [f'atributo{i+1}_classe2' for i in range(len(atributos2))] + ['classe']

    df_resultados = pd.DataFrame(resultados, columns=colunas)

    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    df_resultados.to_csv(caminho_csv, index=False)

    return caminho_csv


def classificar_imagem(modelo_id, imagem):
    modelo_path = f'modelos_salvos/modelo_{modelo_id}.h5'
    classes_path = f'modelos_salvos/classes_{modelo_id}.json'
    
    # Carregar o modelo
    modelo = tf.keras.models.load_model(modelo_path)
    
    # Carregar o mapeamento de classes
    with open(classes_path, 'r') as f:
        class_indices = json.load(f)
    
    # Reverter o dicionário de classes
    index_to_class = {v: k for k, v in class_indices.items()}
    
    # Processar a imagem para extrair as características de cor (não a imagem bruta)
    atributos1_rgb = session.get('atributos1')  # Carregar as cores da sessão
    atributos2_rgb = session.get('atributos2')
    
    contagem_classe1, contagem_classe2 = processar_imagem(imagem, atributos1_rgb, atributos2_rgb)
    
    # Combinar as contagens das classes para gerar o vetor de características
    caracteristicas = contagem_classe1 + contagem_classe2
    caracteristicas = np.expand_dims(caracteristicas, axis=0)  # Ajuste para o formato correto (1, 6)
    
    # Fazer a predição
    pred = modelo.predict(caracteristicas)[0][0]
    classe_idx = int(pred > 0.5)  # Ajuste para classificação binária
    classe_predita = index_to_class[classe_idx]
    
    return classe_predita

def upload_pasta_atributos():
    try:
        # Aqui você faz o processamento do upload como antes...
        # Criação do diretório de upload, etc.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_upload = os.path.join(ARQUIVOSREDE1_DIR, f"upload_{timestamp}")
        os.makedirs(pasta_upload, exist_ok=True)
        os.makedirs(pasta_upload, exist_ok=True)
        print(f"Diretório de upload criado: {pasta_upload}")

        arquivos_classe1 = request.files.getlist('arquivos1[]')
        arquivos_classe2 = request.files.getlist('arquivos2[]')
        atributos1_rgb, atributos2_rgb = obter_atributos_rgb_normalizados(request.form)

        # Salvar arquivos nas respectivas pastas
        salvar_arquivos(arquivos_classe1, os.path.join(pasta_upload, "Classe1"))
        salvar_arquivos(arquivos_classe2, os.path.join(pasta_upload, "Classe2"))

        # Agora sim faz sentido salvar a pasta na sessão
        session['ultimo_upload'] = pasta_upload
        session['atributos1'] = atributos1_rgb
        session['atributos2'] = atributos2_rgb

        # Continuar com o processamento...
        flash('Arquivos processados com sucesso! Agora defina os parâmetros da rede.', 'success')
        return redirect(url_for('routes.variaveisRede1'))

    except Exception as e:
        flash(f'Erro no processamento: {str(e)}', 'error')
        return redirect(url_for('routes.variaveisRede1'))

def obter_imagens_ultimo_upload():
    # Recuperar o caminho do último upload da sessão
    pasta_upload = session.get('ultimo_upload')

    if pasta_upload:
        pasta_treinamento = os.path.join(pasta_upload, "treinamento")
        
        imagens_treinamento = []
        for classe in ["Classe1", "Classe2"]:
            pasta_classe = os.path.join(pasta_treinamento, classe)
            
            # Verificar se a pasta existe antes de tentar acessar
            if os.path.exists(pasta_classe):
                imagens_treinamento.extend([
                    os.path.join(pasta_classe, f)
                    for f in os.listdir(pasta_classe)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ])

        return imagens_treinamento
    else:
        raise FileNotFoundError("Nenhum upload encontrado!")   

def carregar_modelo_mais_recente():
    """
    Função para carregar o modelo mais recente da pasta 'modelos_salvos'
    e retornar tanto o modelo quanto as classes associadas.
    """
    # Localização da pasta onde os modelos estão salvos
    pasta_modelos = 'modelos_salvos'

    # Listar todos os diretórios dentro de 'modelos_salvos' para pegar os modelos mais recentes
    diretorios = sorted(
        [d for d in os.listdir(pasta_modelos) if os.path.isdir(os.path.join(pasta_modelos, d))],
        reverse=True  # Para garantir que o mais recente venha primeiro
    )

    if not diretorios:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta 'modelos_salvos'.")

    # Pegando o diretório do modelo mais recente
    modelo_dir = diretorios[0]

    # Carregar o modelo treinado
    modelo_path = os.path.join(pasta_modelos, modelo_dir, 'modelo.h5')
    modelo = tf.keras.models.load_model(modelo_path)

    # Verificar se o modelo foi carregado corretamente
    if not modelo:
        raise Exception("Erro ao carregar o modelo.")

    # Carregar as classes do modelo
    classes_path = os.path.join(pasta_modelos, modelo_dir, 'classes.json')
    with open(classes_path, 'r') as f:
        classes = json.load(f)

    return modelo, classes


def listar_modelos(tipo_rede):
    pasta_modelos = 'modelos_salvos'
    tipo_modelo_pasta = f"model{tipo_rede}"  # modelopp ou modelocnn
    modelos = []

    # Percorrer as subpastas da pasta principal de modelos_salvos
    for root, dirs, files in os.walk(pasta_modelos):
        for dir_name in dirs:
            if tipo_modelo_pasta in dir_name:  # Verifica se é do tipo correto
                modelos.append(dir_name)


def extrair_atributos(imagem_array, atributos1, atributos2):
    """
    Extrai os atributos das classes comparando os pixels da imagem com os atributos fornecidos.
    
    :param imagem_array: A imagem processada em formato numpy (normalizada).
    :param atributos1: Lista com os atributos da Classe1 (RGB normalizado).
    :param atributos2: Lista com os atributos da Classe2 (RGB normalizado).
    :return: Vetor com as contagens dos atributos das duas classes.
    """
    contagem_classe1 = [0] * len(atributos1)
    contagem_classe2 = [0] * len(atributos2)

    # Percorrer os pixels da imagem
    for linha in imagem_array:
        for pixel in linha:
            r, g, b = pixel

            # Verificar atributos da Classe1
            for i, attr in enumerate(atributos1):
                if comparar_cores((r, g, b), attr):  # Função comparar_cores já normaliza as cores
                    contagem_classe1[i] += 1
                    break  # Já contou para a Classe1

            # Se não for Classe1, verificar para a Classe2
            for j, attr in enumerate(atributos2):
                if comparar_cores((r, g, b), attr):
                    contagem_classe2[j] += 1
                    break  # Já contou para a Classe2

    # Retorna o vetor com as contagens de atributos das duas classes
    return contagem_classe1 + contagem_classe2

def carregar_modelo(modelo_nome):
    try:
        # Determinar tipo de modelo
        if modelo_nome.startswith('modeloopp_'):
            tipo = 'OPP'
        elif modelo_nome.startswith('modelopp_'):
            tipo = 'PADRAO'
        else:
            raise ValueError("Prefixo do modelo não reconhecido")

        # Caminhos dos arquivos
        base_path = os.path.join('modelos_salvos', modelo_nome)
        modelo_path = os.path.join(base_path, 'modelo.h5')
        classes_path = os.path.join(base_path, 'classes.json')

        # Verificar existência
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Modelo não encontrado: {modelo_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes não encontradas: {classes_path}")

        # Carregar componentes
        modelo = tf.keras.models.load_model(modelo_path)
        with open(classes_path, 'r') as f:
            classes = json.load(f)

        # Carregar atributos adicionais para OPP
        if tipo == 'OPP':
            atributos_path = os.path.join(base_path, 'atributos.json')
            if os.path.exists(atributos_path):
                with open(atributos_path, 'r') as f:
                    atributos = json.load(f)
                    session['atributos1'] = atributos.get('atributos1', [])
                    session['atributos2'] = atributos.get('atributos2', [])

        return modelo, classes

    except Exception as e:
        print(f"Erro ao carregar modelo {modelo_nome}: {str(e)}")
        raise


# Rotas simples
@bp.route("/")
def home():
    return render_template("home.html")

@bp.route("/rede1")
def rede1():
    return render_template("rede1.html")

@bp.route("/rede2")
def rede2():
    return render_template("rede2.html")

@bp.route("/variaveisRede1.html")
def variaveisRede1():
    return render_template("variaveisRede1.html")

@bp.route("/variaveisRede2.html")
def variaveisRede2():
    return render_template("variaveisRede2.html")

@bp.route("/resultadoRede.html")
def resultadoRede():
    return render_template("resultadoRede.html")

@bp.route("/escolherModelo.html")
def escolherModelo():
    return render_template("escolherModelo.html")

@bp.route('/selecionarModelos', methods=['GET'])
def selecionar_modelos_opp():
    # Seleciona os modelos para a rede "opp"
    modelos_rede_opp = carregar_modelo("opp")  # Para Rede 1 (modelopp)
    # Passa os modelos para o template
    return render_template('selecionarModelos.html', modelos_rede=modelos_rede_opp)  # Corrigido

@bp.route('/selecionarModelos2', methods=['GET'])
def selecionar_modelos_cnn():
    # Seleciona os modelos para a rede "cnn"
    modelos_rede_cnn = carregar_modelo("cnn")  # Para Rede 2 (modelocnn)
    # Passa os modelos para o template
    return render_template('selecionarModelos2.html', modelos_rede=modelos_rede_cnn)  # Corrigido


@bp.route("/usarModelo.html")
def usarModelo():
    return render_template("usarModelo.html")

# Rotas de Upload e Processamento
@bp.route('/upload_pasta_atributos', methods=['POST'])
def upload_pasta_atributos():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_upload = os.path.join(ARQUIVOSREDE1_DIR, f"upload_{timestamp}")
        os.makedirs(pasta_upload, exist_ok=True)  # Criação única da pasta

        arquivos_classe1 = request.files.getlist('arquivos1[]')
        arquivos_classe2 = request.files.getlist('arquivos2[]')

        # Obtenção dos atributos RGB
        atributos1_rgb, atributos2_rgb = obter_atributos_rgb_normalizados(request.form)
        
        # Armazenando as informações na sessão
        session['atributos1'] = atributos1_rgb
        session['atributos2'] = atributos2_rgb
        session['ultimo_upload'] = pasta_upload

        # Salvar arquivos nas respectivas pastas
        salvar_arquivos(arquivos_classe1, os.path.join(pasta_upload, "Classe1"))
        salvar_arquivos(arquivos_classe2, os.path.join(pasta_upload, "Classe2"))

        # Distribuir arquivos entre treinamento e teste
        distribuir_arquivos_treinamento_teste(pasta_upload)

        # Obter imagens do último upload
        imagens_treinamento = obter_imagens_ultimo_upload()
        caminho_csv = os.path.join(ARQUIVOSREDE1_DIR, f"dados_{timestamp}.csv")
        caminho_csv = converter_imagens_para_csv(imagens_treinamento, atributos1_rgb, atributos2_rgb, caminho_csv)

        # Salvando caminho CSV e pasta de teste na sessão
        session['caminho_csv'] = caminho_csv
        session['pasta_teste'] = os.path.join(pasta_upload, "teste")

        # Exibir mensagem de sucesso
        flash('Arquivos processados com sucesso! Agora defina os parâmetros da rede.', 'success')
        return redirect(url_for('routes.variaveisRede1'))

    except Exception as e:
        # Exibir mensagem de erro caso ocorra
        flash(f'Erro no processamento: {str(e)}', 'error')
        return redirect(url_for('routes.variaveisRede1'))

# Rota de Treinamento
@bp.route('/treinar_rede', methods=['POST'])
def treinar_rede():
    try:
        # 1. Validar parâmetros
        epocas = int(request.form.get('epocas', 0))
        neuronios = int(request.form.get('neuronios', 0))
        camadas = int(request.form.get('camadas', 0))
        if not all([epocas > 0, neuronios > 0, camadas > 0]):
            flash("Parâmetros inválidos!", "error")
            return redirect(url_for('routes.variaveisRede1'))

        # 2. Validar CSV
        caminho_csv = session.get('caminho_csv')
        if not caminho_csv:
            flash("CSV não encontrado na sessão!", "error")
            return redirect(url_for('routes.variaveisRede1'))
        
        df = pd.read_csv(caminho_csv)
        if 'classe' not in df.columns:
            flash("CSV não contém a coluna 'classe'.", "error")
            return redirect(url_for('routes.variaveisRede1'))

        # 3. Treinar (usando RedeNeural1)
        from rede_neural1 import RedeNeural1
        X = df.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df['classe'])  # Converte classes para 0/1
        
        rede = RedeNeural1()
        resultado = rede.treinar(X, y, epocas, neuronios, camadas)
        #acc, cm = treinar_rede()
        
        acc = resultado['acuracia']  # ← Correto! Acessa a chave 'acuracia'
        cm = resultado['matriz_confusao']
    
        # 4. Salvar modelo e resultados
        rede.model.save('modelos_salvos/modelo_rede1.keras')  # Garante que o modelo é salvo
        novo_treinamento = Treinamento(
            epocas=epocas,
            neuronios=neuronios,
            enlaces=camadas,
            resultado=f"Acurácia: {acc:.4f}",
            #matriz_confusao=str(resultado['matriz_confusao']),
            #data_treinamento=datetime.now()
        )
        db.session.add(novo_treinamento)
        db.session.commit()

        # 5. Exibir resultados
        return render_template('resultadoRede.html',
                            acuracia=f"{resultado['acuracia']:.2%}",
                            epocas=epocas,
                            neuronios=neuronios,
                            camadas=camadas)

    except Exception as e:
        print(f"ERRO: {str(e)}")
        flash(f"Falha no treinamento: {str(e)}", "error")
        return redirect(url_for('routes.variaveisRede1'))



# Rotas para Rede Neural 2 
@bp.route('/upload', methods=['POST'])
def upload():
    num_pastas = int(request.form['num_pastas'])
    timestamp = str(int(time.time()))
    
    # Pasta principal do upload do usuário
    pasta_principal = os.path.join(UPLOAD_FOLDER, f'teste{timestamp}')
    pasta_treinamento = os.path.join(pasta_principal, 'treinamento2')
    pasta_teste = os.path.join(pasta_principal, 'teste2')

    os.makedirs(pasta_treinamento, exist_ok=True)
    os.makedirs(pasta_teste, exist_ok=True)

    for i in range(1, num_pastas + 1):
        classe_nome = request.form[f'classe{i}']

        # Cria subpastas por classe dentro de treinamento2/ e teste2/
        caminho_treinamento_classe = os.path.join(pasta_treinamento, classe_nome)
        caminho_teste_classe = os.path.join(pasta_teste, classe_nome)
        os.makedirs(caminho_treinamento_classe, exist_ok=True)
        os.makedirs(caminho_teste_classe, exist_ok=True)

        arquivos = request.files.getlist(f'upload{i}')
        random.shuffle(arquivos)
        num_treinamento = int(len(arquivos) * 0.8)
        arquivos_treinamento = arquivos[:num_treinamento]
        arquivos_teste = arquivos[num_treinamento:]

        for arquivo in arquivos_treinamento:
            caminho_arquivo = os.path.join(caminho_treinamento_classe, arquivo.filename)
            arquivo.save(caminho_arquivo)

        for arquivo in arquivos_teste:
            caminho_arquivo = os.path.join(caminho_teste_classe, arquivo.filename)
            arquivo.save(caminho_arquivo)

    return redirect(url_for('routes.variaveisRede2'))

@bp.route('/enviar2', methods=['POST'])
def enviar2():
    epocas = request.form.get('epocas')
    neuronios = request.form.get('neuronios')
    camadas = request.form.get('camadas')

    if not epocas or not neuronios or not camadas:
        flash('Por favor, preencha todos os campos.', 'erro')
        return redirect(url_for('routes.erro'))

    try:
        epocas = int(epocas)
        neuronios = int(neuronios)
        camadas = int(camadas)
        
        # Salvar configuração no banco antes de treinar
        novo_treinamento = Treinamento(
            epocas=epocas,
            neuronios=neuronios,
            enlaces=camadas,
            #data_treinamento=datetime.now()
        )
        db.session.add(novo_treinamento)
        db.session.commit()
        
        
        resultado = treinar_rede_neural_cnn()
        
        # Atualizar resultado no banco
        novo_treinamento.resultado = f"Acurácia: {resultado['acuracia']:.4f} | Validação: {resultado['val_acuracia']:.4f}"
        novo_treinamento.matriz_confusao = str(resultado['matriz_confusao'])
        db.session.commit()
        
        # Após salvar o modelo, você já tem o valor de acurácia:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_path = f'modelos_salvos/{timestamp}/modelo.h5'
        
        # Agora, crie o diretório para armazenar o modelo e a acurácia
        os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
        

        # Agora salve a acurácia em um arquivo info.json dentro da mesma pasta
        info = {
            'acuracia': resultado['acuracia'],
            'val_acuracia': resultado['val_acuracia'],  # Adicionando a acurácia de validação também
            'epocas': epocas,
            'neuronios': neuronios,
            'camadas': camadas
        }

        with open(f'modelos_salvos/{timestamp}/info.json', 'w') as f:
            json.dump(info, f)

        return render_template('resultadoRede.html',
                            acuracia=resultado['acuracia'],
                            val_acuracia=resultado['val_acuracia'],
                            matriz_confusao=resultado['matriz_confusao'],
                            epocas=epocas,
                            neuronios=neuronios,
                            camadas=camadas)

            
    except Exception as e:
        flash(f'Erro ao treinar a CNN: {str(e)}', 'erro')
        return redirect(url_for('routes.variaveisRede2'))

@bp.route('/classificar_imagem', methods=['POST'])
def classificar_imagem():
    try:
        arquivo = request.files['imagem']
        caminho_temporario = os.path.join('uploads_imagens', secure_filename(arquivo.filename))
        os.makedirs('uploads_imagens', exist_ok=True)
        arquivo.save(caminho_temporario)
        
        # Carregar modelo CNN
        from cnn_model import carregar_modelo_cnn
        modelo = carregar_modelo_cnn()
        
        # Pré-processar imagem
        img = tf.keras.preprocessing.image.load_img(
            caminho_temporario, 
            target_size=(150, 150)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Fazer previsão
        predicao = modelo.predict(img_array)
        classe_predita = int(predicao[0][0] > 0.5)
        
        # Mapear classe para nome (ajuste conforme suas classes)
        CLASSES = {0: "Classe1", 1: "Classe2"}
        classe_nome = CLASSES.get(classe_predita, "Desconhecido")
        
        return jsonify({
            "mensagem": f"Imagem classificada como: {classe_nome}",
            "classe_predita": classe_predita,
            "probabilidade": float(predicao[0][0])
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 400


@bp.route('/selecionarModelos', methods=['GET', 'POST'])
def selecionar_modelo():
    modelos = []
    modelos_dir = 'modelos_salvos'
    
    try:
        # Verificar se o diretório de modelos existe
        if not os.path.exists(modelos_dir):
            flash("Diretório de modelos não encontrado", "error")
            return render_template('selecionarModelos.html', modelos=modelos)

        # Listar todos os modelos válidos
        for item in os.listdir(modelos_dir):
            item_path = os.path.join(modelos_dir, item)
            
            # Verificar se é um diretório e contém os arquivos necessários
            if os.path.isdir(item_path):
                has_model = any(f.endswith(('.h5', '.keras')) for f in os.listdir(item_path))
                has_classes = any(f.endswith('.json') and 'classes' in f.lower() for f in os.listdir(item_path))
                
                if has_model and has_classes:
                    # Determinar o tipo do modelo
                    model_type = 'OPP' if item.startswith('modeloopp_') else 'CNN' if item.startswith('modelopp_') else 'OUTRO'
                    
                    model_info = {
                        'nome': item,
                        'tipo': model_type,
                        'timestamp': item.split('_')[-1],
                        'data_criacao': datetime.fromtimestamp(
                            os.path.getctime(item_path)
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        'caminho': item_path
                    }
                    modelos.append(model_info)

        # Ordenar por timestamp
        modelos.sort(key=lambda x: x['timestamp'], reverse=True)

        if request.method == 'POST':
            modelo_selecionado = request.form.get('modelo_nome')
            selected_model = next((m for m in modelos if m['nome'] == modelo_selecionado), None)
            
            if selected_model:
                session['modelo_selecionado'] = selected_model['nome']
                session['modelo_tipo'] = selected_model['tipo']
                session['modelo_path'] = selected_model['caminho']
                flash(f"Modelo {selected_model['nome']} selecionado com sucesso!", "success")
                return redirect(url_for('routes.selecionar_modelo'))
            else:
                flash("Modelo selecionado inválido", "error")

        return render_template('selecionarModelos.html', 
                            modelos=modelos,
                            tipos_modelos=set(m['tipo'] for m in modelos))

    except Exception as e:
        flash(f"Erro ao listar modelos: {str(e)}", "error")
        return render_template('selecionarModelos.html', modelos=[])


@bp.route('/selecionarRede2', methods=['GET', 'POST'])
def selecionar_modelo_rede2():
    diretorio_modelos = 'modelos_salvos'
    modelos = []

    for nome_pasta in os.listdir(diretorio_modelos):
        if not nome_pasta.startswith("modelocnn_"):
            continue  # ignora os modelos que não são da rede 2

        caminho_pasta = os.path.join(diretorio_modelos, nome_pasta)
        if os.path.isdir(caminho_pasta):
            modelo_path = os.path.join(caminho_pasta, 'modelo.h5')
            if os.path.exists(modelo_path):
                modelos.append({'nome': nome_pasta})

    modelo_selecionado = None
    if request.method == 'POST':
        modelo_selecionado = request.form.get('modelo_nome')
        print(f"Modelo selecionado: {modelo_selecionado}")  # Debug: verifique se o nome do modelo está correto
        session['modelo_selecionado'] = modelo_selecionado
        return redirect(url_for('routes.selecionar_modelo_rede2'))  # Redireciona para a mesma página para atualizar a seleção

    # Quando o método for GET, renderiza a página com o modelo
    return render_template('selecionarModelos2.html', modelos=modelos, modelo_selecionado=session.get('modelo_selecionado'))


@bp.route('/classificar_rede1', methods=['POST'])
def classificar_rede1():
    # Validação inicial
    if 'imagem' not in request.files:
        return render_template('erro.html', mensagem="Arquivo de imagem não enviado."), 400
    
    arquivo = request.files['imagem']
    if arquivo.filename == '':
        return render_template('erro.html', mensagem="Nenhuma imagem selecionada."), 400

    modelo_selecionado = session.get('modelo_selecionado')
    if not modelo_selecionado or not modelo_selecionado.startswith('modeloopp_'):
        return render_template('erro.html', mensagem="Modelo OPP não selecionado ou inválido."), 400

    try:
        # Carregar modelo e classes
        modelo, classes = carregar_modelo(modelo_selecionado)
        
        # Verificar atributos necessários
        atributos1 = session.get('atributos1')
        atributos2 = session.get('atributos2')
        if not atributos1 or not atributos2:
            raise Exception("Atributos de cores não encontrados na sessão.")

        # Processamento específico para Rede 1 (OPP)
        temp_path = os.path.join('temp_uploads', secure_filename(arquivo.filename))
        arquivo.save(temp_path)
        
        # Extrair atributos específicos para OPP
        contagem_classe1, contagem_classe2 = processar_imagem(temp_path, atributos1, atributos2)
        atributos_extraidos = np.array(contagem_classe1 + contagem_classe2).reshape(1, -1)
        
        # Classificação
        pred = modelo.predict(atributos_extraidos)[0][0]
        classe_predita = 1 if pred > 0.5 else 0
        nome_classe = classes.get(str(classe_predita), f'Classe {classe_predita}')
        
        # Resultados adicionais para OPP
        porcentagem_classe1 = (contagem_classe1 / sum(contagem_classe1 + contagem_classe2)) * 100
        porcentagem_classe2 = 100 - porcentagem_classe1

        # Limpeza
        os.remove(temp_path)

        return render_template('resultado_classificacao_rede1.html',
                            nome_classe=nome_classe,
                            classe_predita=classe_predita,
                            porcentagem_classe1=f"{porcentagem_classe1:.2f}%",
                            porcentagem_classe2=f"{porcentagem_classe2:.2f}%",
                            atributos_utilizados=atributos1+atributos2)

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return render_template('erro.html', mensagem=f"Erro na classificação OPP: {str(e)}"), 500

@bp.route('/classificar_rede2', methods=['POST'])
def classificar_rede2():
    # Validação inicial
    if 'imagem' not in request.files:
        return render_template('erro.html', mensagem="Arquivo de imagem não enviado."), 400
    
    arquivo = request.files['imagem']
    if arquivo.filename == '':
        return render_template('erro.html', mensagem="Nenhuma imagem selecionada."), 400

    modelo_selecionado = session.get('modelo_selecionado')
    if not modelo_selecionado or not modelo_selecionado.startswith('modelopp_'):
        return render_template('erro.html', mensagem="Modelo padrão não selecionado ou inválido."), 400

    try:
        # Carregar modelo e classes
        modelo, classes = carregar_modelo(modelo_selecionado)
        
        # Processamento de imagem padrão
        img = Image.open(arquivo).convert('RGB')
        img = img.resize((150, 150))  # Ajustar conforme necessidade do modelo
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Classificação
        pred = modelo.predict(img_array)[0][0]
        classe_predita = 1 if pred > 0.5 else 0
        nome_classe = classes.get(str(classe_predita), f'Classe {classe_predita}')
        
        # Probabilidade de confiança
        probabilidade = pred if classe_predita == 1 else (1 - pred)
        
        return render_template('resultado_classificacao_rede2.html',
                            nome_classe=nome_classe,
                            classe_predita=classe_predita,
                            confianca=f"{probabilidade*100:.2f}%",
                            modelo_utilizado=modelo_selecionado)

    except Exception as e:
        return render_template('erro.html', mensagem=f"Erro na classificação padrão: {str(e)}"), 500
    
