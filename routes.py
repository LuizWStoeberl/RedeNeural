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


# Configurações
bp = Blueprint("routes", __name__)

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

        print(f"Imagem: {imagem_path}, Classe: {classe_real}, Contagem Classe 1: {contagem_classe1}, Contagem Classe 2: {contagem_classe2}")
        print(f"Soma Classe 1: {soma1}, Soma Classe 2: {soma2}")

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
    # Carregar o modelo e o mapeamento de classes
    modelo_path = f'modelos_salvos/modelo_{modelo_id}.h5'
    classes_path = f'modelos_salvos/classes_{modelo_id}.json'
    
    # Carregar o modelo
    modelo = tf.keras.models.load_model(modelo_path)
    
    # Carregar o mapeamento de classes
    with open(classes_path, 'r') as f:
        class_indices = json.load(f)
    
    # Reverter o dicionário de classes
    index_to_class = {v: k for k, v in class_indices.items()}
    
    # Processar a imagem (ajustar o tamanho da imagem de acordo com a rede)
    img = tf.keras.preprocessing.image.load_img(imagem, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Fazer a predição
    pred = modelo.predict(img_array)[0][0]
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


@bp.route("/selecionarModelos.html")
def selecionarModelos():
    return render_template("selecionarModelos.html")

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



# Rotas para Rede Neural 2 (mantidas inalteradas para brevidade)
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


# ROTAS BANCO DE DADOS



@bp.route('/selecionarModelos', methods=['GET', 'POST'])
def selecionar_modelo():
    diretorio_modelos = 'modelos_salvos'
    modelos = []

    # Carrega os arquivos .h5 no diretório
    for nome_arquivo in os.listdir(diretorio_modelos):
        if nome_arquivo.endswith('.h5'):
            timestamp = nome_arquivo.split('_')[-1].replace('.h5', '')
            modelos.append({
                'nome': nome_arquivo,
                'timestamp': timestamp,
                'acuracia': "90%"  # Coloque a acurácia correta se tiver salva
            })

    modelo_selecionado = request.form.get('modelo_nome') if request.method == 'POST' else None

    return render_template('selecionarModelos.html', modelos=modelos, modelo_selecionado=modelo_selecionado)


@bp.route('/classificar', methods=['GET', 'POST'])
def classificar_imagem_usuario():
    # Verifica se os campos obrigatórios foram enviados
    if 'modelo_nome' not in request.form:
        return jsonify({"erro": "Campo 'modelo_nome' faltando"}), 400
    
    if 'imagem' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    # Obtém os dados do formulário
    modelo_nome = request.form['modelo_nome']
    imagem = request.files['imagem']

    # Verifica se o arquivo tem um nome válido
    if imagem.filename == '':
        return jsonify({"erro": "Nome de arquivo inválido"}), 400

    try:
        # Caminhos dos arquivos (modelo e classes)
        caminho_modelo = os.path.join('modelos_salvos', modelo_nome)
        nome_base = modelo_nome.replace('.h5', '').replace('.keras', '')
        timestamp = nome_base.split('_')[-1]
        caminho_classes = os.path.join('modelos_salvos', f'classes_{timestamp}.json')

        # Verifica se os arquivos existem
        if not os.path.exists(caminho_modelo):
            return jsonify({"erro": f"Modelo não encontrado: {modelo_nome}"}), 404
        
        if not os.path.exists(caminho_classes):
            return jsonify({"erro": f"Arquivo de classes não encontrado para o modelo: {modelo_nome}"}), 404

        # Salva a imagem temporariamente
        os.makedirs('uploads_temp', exist_ok=True)
        caminho_temp = os.path.join('uploads_temp', secure_filename(imagem.filename))
        imagem.save(caminho_temp)

        # Carrega o modelo e as classes
        modelo = load_model(caminho_modelo)  # Aqui, certifique-se de que 'load_model' está correto
        with open(caminho_classes, 'r') as f:
            class_indices = json.load(f)

        # Pré-processa a imagem e faz a predição
        img = tf.keras.preprocessing.image.load_img(caminho_temp, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = modelo.predict(img_array)[0][0]
        classe_idx = int(pred > 0.5)  # Adapte para multiclasse se necessário
        classe_predita = class_indices.get(str(classe_idx), "classe_desconhecida")

        # Remove o arquivo temporário
        os.remove(caminho_temp)

        # Retorna APENAS a classe prevista (em formato texto ou JSON)
        return jsonify({"classe": classe_predita})  # Ou return classe_predita (texto puro)
    
    except Exception as e:
        return jsonify({"erro": f"Erro durante a classificação: {str(e)}"}), 500