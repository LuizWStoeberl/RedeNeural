from flask import Blueprint,request, redirect, url_for, render_template, flash, session
import random
import shutil
import time
from models import db, Treinamento
from datetime import datetime
from processar_imagem import processar_todas_imagens
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from rede_neural1 import RedeNeural1

# Configurações
bp = Blueprint("routes", __name__)

ARQUIVOSREDE1_DIR = 'arquivosRede1'
os.makedirs(ARQUIVOSREDE1_DIR, exist_ok=True)

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

def comparar_cores(cor_pixel, atributo, tolerancia=15):
    r, g, b = cor_pixel
    r_attr, g_attr, b_attr = atributo
    return (abs(r - r_attr) <= tolerancia / 255.0 and
            abs(g - g_attr) <= tolerancia / 255.0 and
            abs(b - b_attr) <= tolerancia / 255.0)

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

        # Percorre cada pixel da imagem e verifica se ele corresponde a alguma cor dos atributos
        for linha in pixels:
            for pixel in linha:
                r, g, b = [v / 255.0 for v in pixel]
                # Verificar se o pixel corresponde a algum dos atributos de Classe 1
                for i, attr in enumerate(atributos1):
                    if comparar_cores((r, g, b), attr):
                        contagem_classe1[i] += 1
                        break
                # Verificar se o pixel corresponde a algum dos atributos de Classe 2
                for j, attr in enumerate(atributos2):
                    if comparar_cores((r, g, b), attr):
                        contagem_classe2[j] += 1
                        break

        return contagem_classe1, contagem_classe2
    except Exception as e:
        print(f"Erro ao processar {imagem_path}: {e}")
        return [0]*len(atributos1), [0]*len(atributos2)

def converter_imagens_para_csv(imagens_treinamento, atributos1, atributos2, caminho_csv="arquivos/tabela_cores.csv"):
    """Converte as imagens de treinamento para um CSV com suas características (contagem de pixels)."""
    resultados = []
    for imagem_path in imagens_treinamento:
        contagem_classe1, contagem_classe2 = processar_imagem(imagem_path, atributos1, atributos2)
        soma1 = sum(contagem_classe1)
        soma2 = sum(contagem_classe2)
        # Definir a classe como a que tiver mais pixels da cor correspondente
        classe = 1 if soma1 > soma2 else 2
        # Adicionar as contagens e a classe à lista de resultados
        resultados.append(contagem_classe1 + contagem_classe2 + [classe])
    
    # Definir os nomes das colunas para o CSV
    colunas = [f'atributo{i+1}_classe1' for i in range(3)] + \
              [f'atributo{i+1}_classe2' for i in range(3)] + ['classe']
    
    # Criar o DataFrame e salvar no CSV
    df_resultados = pd.DataFrame(resultados, columns=colunas)
    os.makedirs('arquivos', exist_ok=True)  # Garantir que o diretório existe
    df_resultados.to_csv(caminho_csv, index=False)  # Salvar o arquivo CSV
    return caminho_csv

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

@bp.route("/rede2.html")
def rede2():
    return render_template("rede2.html")

@bp.route("/variaveisRede1.html")
def variaveisRede1():
    return render_template("variaveisRede1.html")

@bp.route("/variaveisRede2.html")
def variaveisRede2():
    return render_template("variaveisRede2.html")

@bp.route("/resultadoRede1.html")
def resultadoRede1():
    return render_template("resultadoRede1.html")

@bp.route("/resultadoRede2.html")
def resultadoRede2():
    return render_template("resultadoRede2.html")

# Rotas de Upload e Processamento
@bp.route('/upload_pasta_atributos', methods=['POST'])
def upload_pasta_atributos():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_upload = os.path.join(ARQUIVOSREDE1_DIR, f"upload_{timestamp}")
    
    atributos1_rgb = session.get('atributos1')  # Atributos da classe 1
    atributos2_rgb = session.get('atributos2')  # Atributos da classe 2

    atributos1_rgb, atributos2_rgb = obter_atributos_rgb_normalizados(request.form)
    session['atributos1'] = atributos1_rgb
    session['atributos2'] = atributos2_rgb

     # Imprimir os valores para depuração
    print(pasta_upload)
    print(atributos1_rgb)
    print(atributos1_rgb)
    if not pasta_upload or not atributos1_rgb or not atributos1_rgb:
        flash('Por favor, faça o upload das imagens e defina os atributos de cor!', 'error')
        return redirect(url_for('routes.upload'))
    try:
        # 1. Validar uploads e atributos
        arquivos_classe1 = request.files.getlist('arquivos1[]')
        arquivos_classe2 = request.files.getlist('arquivos2[]')

        # Verificar se arquivos foram enviados
        if not arquivos_classe1 or not arquivos_classe2:
            flash('Envie arquivos para ambas as classes!', 'error')
            return redirect(url_for('routes.rede1'))

        # 2. Criar estrutura de diretórios
        pasta_upload = session.get('ultimo_upload')  # Caminho da última pasta de upload
        os.makedirs(pasta_upload, exist_ok=True)

        # 3. Salvar arquivos de forma segura com função reutilizável
        salvar_arquivos(arquivos_classe1, os.path.join(pasta_upload, "Classe1"))
        salvar_arquivos(arquivos_classe2, os.path.join(pasta_upload, "Classe2"))

        

        # 5. Distribuir arquivos entre treinamento e teste com função reutilizável
        distribuir_arquivos_treinamento_teste(pasta_upload)

        # Processar imagens de treinamento e gerar CSV
        imagens_treinamento = []
        for classe in ["Classe1", "Classe2"]:
            pasta_classe = os.path.join(pasta_upload, classe)
            imagens_treinamento.extend([
                os.path.join(pasta_classe, f)
                for f in os.listdir(pasta_classe)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])

        # Gerar o CSV com as características das imagens
        caminho_csv = os.path.join(ARQUIVOSREDE1_DIR, f"dados_{timestamp}.csv")
        caminho_csv = converter_imagens_para_csv(imagens_treinamento, atributos1_rgb, atributos2_rgb, caminho_csv)
        
        # 7. Armazenar informações na sessão
        session['caminho_csv'] = caminho_csv
        session['pasta_teste'] = os.path.join(pasta_upload, "teste")

        flash('Arquivos processados com sucesso! Agora defina os parâmetros da rede.', 'success')
        return redirect(url_for('routes.variaveisRede1'))

    except Exception as e:
        # Limpeza em caso de erro
        #if 'pasta_upload' in locals() and os.path.exists(pasta_upload):
         #   shutil.rmtree(pasta_upload)

        flash(f'Erro no processamento: {str(e)}', 'error')
        return redirect(url_for('routes.variaveisRede1'))


# Rota de Treinamento
@bp.route('/treinar_rede', methods=['POST'])
def treinar_rede():
    try:
        # 1. Validar e obter parâmetros do formulário
        epocas = int(request.form.get('epocas', 0))
        neuronios = int(request.form.get('neuronios', 0))
        camadas = int(request.form.get('camadas', 0))
        
        if not all([epocas > 0, neuronios > 0, camadas > 0]):
            flash('Todos os parâmetros devem ser números positivos!', 'error')
            return redirect(url_for('routes.variaveisRede1'))

        # 2. Obter caminho do CSV da sessão
        caminho_csv = session.get('caminho_csv')
        if not caminho_csv or not os.path.exists(caminho_csv):
            flash('Dados de treinamento não encontrados. Por favor, faça o upload novamente.', 'error')
            return redirect(url_for('routes.rede1'))

        # 3. Carregar e validar dataset
        try:
            df = pd.read_csv(caminho_csv)
            
            # Verificar se temos as 6 colunas de atributos (3 para cada classe) + classe
            colunas_esperadas = [
                'atributo1_classe1', 'atributo2_classe1', 'atributo3_classe1',
                'atributo1_classe2', 'atributo2_classe2', 'atributo3_classe2',
                'classe'
            ]
            
            if not all(col in df.columns for col in colunas_esperadas):
                flash('Estrutura de dados inválida. Por favor, refaça o upload das imagens.', 'error')
                return redirect(url_for('routes.rede1'))
                
            X = df[colunas_esperadas[:-1]].values  # Todas as colunas exceto a última
            y = df['classe'].values - 1  # Converter classes (1,2) para (0,1)

        except Exception as e:
            flash(f'Erro ao ler dados de treinamento: {str(e)}', 'error')
            return redirect(url_for('routes.rede1'))

        # 4. Treinar a rede neural
        try:
            rede = RedeNeural1()
            resultado = rede.treinar(
                X=X,
                y=y,
                epocas=epocas,
                neuronios=neuronios,
                camadas=camadas
            )

            # 5. Salvar resultados no banco de dados
            novo_treinamento = Treinamento(
                epocas=epocas,
                neuronios=neuronios,
                camadas=camadas,
                acuracia=resultado['acuracia'],
                matriz_confusao=str(resultado['matriz_confusao']),
                data_treinamento=datetime.now()
            )
            db.session.add(novo_treinamento)
            db.session.commit()

            # 6. Preparar dados para exibição
            matriz_confusao = [
                ["Classe Prevista 0", "Classe Prevista 1"],
                ["Classe Real 0", resultado['matriz_confusao'][0][0]],
                ["Classe Real 1", resultado['matriz_confusao'][1][1]]
            ]

            return render_template('resultadoRede1.html',
                                acuracia=f"{resultado['acuracia']:.2%}",
                                matriz_confusao=matriz_confusao,
                                epocas=epocas,
                                neuronios=neuronios,
                                camadas=camadas)

        except Exception as e:
            db.session.rollback()
            flash(f'Erro durante o treinamento: {str(e)}', 'error')
            return redirect(url_for('routes.resultadoRede1'))

    except ValueError:
        flash('Por favor, insira valores numéricos válidos!', 'error')
        return redirect(url_for('routes.resultadoRede1'))
        
    except Exception as e:
        flash(f'Erro inesperado: {str(e)}', 'error')
        return redirect(url_for('routes.resultadoRede1'))



# Rotas para Rede Neural 2 (mantidas inalteradas para brevidade)
@bp.route('/upload', methods=['POST'])
def upload():
    num_pastas = int(request.form['num_pastas'])
    timestamp = str(int(time.time()))
    pasta_principal = os.path.join(UPLOAD_FOLDER, f'teste{timestamp}')
    os.makedirs(pasta_principal, exist_ok=True)

    for i in range(1, num_pastas + 1):
        pasta_nome = request.form[f'classe{i}']
        pasta_destino = os.path.join(pasta_principal, pasta_nome)
        os.makedirs(pasta_destino, exist_ok=True)

        pasta_treinamento = os.path.join(pasta_destino, 'treinamento')
        pasta_teste = os.path.join(pasta_destino, 'teste')
        os.makedirs(pasta_treinamento, exist_ok=True)
        os.makedirs(pasta_teste, exist_ok=True)

        arquivos = request.files.getlist(f'upload{i}')
        random.shuffle(arquivos)
        num_treinamento = int(len(arquivos) * 0.8)
        arquivos_treinamento = arquivos[:num_treinamento]
        arquivos_teste = arquivos[num_treinamento:]

        for arquivo in arquivos_treinamento:
            caminho_arquivo = os.path.join(pasta_treinamento, arquivo.filename)
            arquivo.save(caminho_arquivo)

        for arquivo in arquivos_teste:
            caminho_arquivo = os.path.join(pasta_teste, arquivo.filename)
            arquivo.save(caminho_arquivo)

    return redirect(url_for('routes.variaveisRede2'))

@bp.route('/enviar2', methods=['POST'])
def enviar2():
    epocas = request.form.get('epocas')
    neuronios = request.form.get('neuronios')
    camadas = request.form.get('camadas')

    if not epocas or not neuronios or not camadas:
        flash('Por favor, preencha todos os campos.', 'erro')
        return redirect(url_for('routes.variaveisRede2'))

    try:
        epocas = int(epocas)
        neuronios = int(neuronios)
        camadas = int(camadas)
        acc, cm = treinar_rede_neural_cnn()  # Assume função de treinamento CNN
        novo_treinamento = Treinamento(
            epocas=epocas,
            neuronios=neuronios,
            enlaces=camadas,
            resultado=f"Acurácia: {acc:.4f}"
        )
        db.session.add(novo_treinamento)
        db.session.commit()
        return render_template('resultadoRede2.html',
                              acuracia=acc,
                              matriz_confusao=cm.tolist(),
                              epocas=epocas,
                              neuronios=neuronios,
                              camadas=camadas)
    except Exception as e:
        flash(f'Erro ao treinar a CNN: {str(e)}', 'erro')
        return redirect(url_for('routes.variaveisRede2'))





#POR ENQUANTO DEIXAR QUIETO



@bp.route('/salvar', methods=['POST'])
def salvar():
    dados = request.json
    linha_count = dados.get("linhaCount", 0)
    quantidade_por_linha = dados.get("quantidadePorLinha", 0)
    cores = dados.get("cores", [])
    labels = dados.get("labels", []) 

    nome_arquivo = "tabela_cores.csv" 
    caminho_arquivo = os.path.join(ARQUIVOSREDE1_DIR, nome_arquivo)

    with open(caminho_arquivo, 'w') as f:
        header = [f"Cor{i+1}" for i in range(quantidade_por_linha)] + ["Classe"]
        f.write(','.join(header) + '\n')
 
        for linha_cores, classe in zip(cores, labels):
            linha_formatada = ','.join(linha_cores) + f",{classe}\n"
            f.write(linha_formatada)

    return {'message': 'Arquivo salvo com sucesso!'}


@bp.route('/formulario_classes', methods=['GET'])
def formulario_classes():
    return render_template("upload_classes.html")

def encontrar_ultima_pasta_teste(base_dir):
    pastas_teste = [p for p in os.listdir(base_dir) if p.startswith("Teste") and os.path.isdir(os.path.join(base_dir, p))]
    if not pastas_teste:
        return None
    pastas_teste.sort(key=lambda x: int(x.replace("Teste", "")))
    return os.path.join(base_dir, pastas_teste[-1])

numero_de_classes = 0

def criar_nova_pasta_teste(base_dir):
    i = 1
    while True:
        nome_pasta = f"Teste{i}"
        caminho_completo = os.path.join(base_dir, nome_pasta)
        if not os.path.exists(caminho_completo):
            os.makedirs(caminho_completo)
            return caminho_completo
        i += 1



@bp.route("/treinar", methods=["POST"])
def treinar():
    try:
        # 1. Converter imagens em CSV
        caminho_csv = converter_imagens_para_csv()

        # 2. Treinar a rede neural com base no CSV gerado
        acc, cm = treinar_rede_neural(caminho_csv)

        return jsonify({
            "mensagem": "Treinamento concluído com sucesso!",
            "acuracia": acc,
            "matriz_confusao": cm.tolist()
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 400
    
@bp.route("/treinar_cnn", methods=["POST"])
def treinar_cnn():
    try:
        from cnn_model import treinar_cnn
        acc, cm = treinar_cnn()
        return jsonify({"mensagem": "CNN treinada com sucesso", "acuracia": acc})
    except Exception as e:
        return jsonify({"erro": str(e)}), 400    


@bp.route('/processar-imagens', methods=['POST'])
def processar_imagens_route():
    try:
        caminho_csv = processar_todas_imagens('uploads_imagens')  # pasta onde o usuário envia as imagens
        return jsonify({"mensagem": "Imagens processadas com sucesso!", "csv_gerado": caminho_csv})
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

@bp.route('/treinar-cnn', methods=['POST'])
def treinar_cnn_route():
    try:
        resultado = treinar_cnn('uploads_imagens')  # Pasta base das imagens organizadas em subpastas (por classe)
        return jsonify({"mensagem": "Treinamento concluído!", "resultado": resultado})
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

@bp.route('/classificar-imagem', methods=['POST'])
def classificar_imagem():
    try:
        arquivo = request.files['imagem']
        caminho_temporario = os.path.join('Uploads_imagens', arquivo.filename)
        os.makedirs('Uploads_imagens', exist_ok=True)
        arquivo.save(caminho_temporario)
        modelo = load_model('modelos_salvos/modelo_cnn.h5')  # Ajustado para usar modelo_cnn.h5
        img = image.load_img(caminho_temporario, target_size=(150, 150))  # Ajustado para 150x150
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predicao = modelo.predict(img_array)
        classe_predita = int(predicao[0][0] > 0.5)  # Para classificação binária
        personagem = CLASSES_PERSONAGENS.get(classe_predita, "Desconhecido")
        return jsonify({
            "mensagem": f"Personagem identificado: {personagem}",
            "classe_predita": classe_predita
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 400
    
@bp.route('/classificar-imagem-densa', methods=['POST'])
def classificar_imagem_densa():
    try:
        from pixel import converter_imagens_para_csv
        arquivo = request.files['imagem']
        caminho_temporario = os.path.join('Uploads_imagens', arquivo.filename)
        os.makedirs('Uploads_imagens', exist_ok=True)
        arquivo.save(caminho_temporario)
        # Converte a imagem para CSV
        pasta_temp = 'temp_imagem'
        os.makedirs(pasta_temp, exist_ok=True)
        shutil.move(caminho_temporario, os.path.join(pasta_temp, arquivo.filename))
        caminho_csv = converter_imagens_para_csv(pasta_temp)
        # Classifica usando a rede densa
        from savemodel import classificar_nova_imagem
        previsoes = classificar_nova_imagem(caminho_csv)
        classe_predita = int(previsoes[0])  # Assume saída binária
        personagem = CLASSES_PERSONAGENS.get(classe_predita, "Desconhecido")
        # Remove pasta temporária
        shutil.rmtree(pasta_temp)
        return jsonify({
            "mensagem": f"Personagem identificado: {personagem}",
            "classe_predita": classe_predita
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 400
    
@bp.route('/definir_intervalos_cor', methods=['POST'])
def definir_intervalos_cor():
    dados = request.json  # Deve vir uma lista de intervalos
    for intervalo in dados:
        cor = IntervaloCor(
            classe=intervalo['classe'],
            r_min=intervalo['r_min'],
            r_max=intervalo['r_max'],
            g_min=intervalo['g_min'],
            g_max=intervalo['g_max'],
            b_min=intervalo['b_min'],
            b_max=intervalo['b_max'],
        )
        db.session.add(cor)
    db.session.commit()
    return jsonify({'mensagem': 'Intervalos de cor salvos com sucesso!'})

@bp.route('/modelos', methods=['GET'])
def listar_modelos():
    modelos = ModeloTreinado.query.all()
    lista = []
    for modelo in modelos:
        lista.append({
            'id': modelo.id,
            'nome': modelo.nome_modelo,
            'tipo': modelo.tipo_modelo,
            'data': modelo.data_treinamento.strftime('%Y-%m-%d %H:%M:%S'),
            'resultado': modelo.resultado
        })
    return jsonify(lista)

@bp.route('/selecionar_modelo', methods=['POST'])
def selecionar_modelo():
    dados = request.json
    id_modelo = dados.get('id_modelo')

    modelo = ModeloTreinado.query.get(id_modelo)
    if not modelo:
        return jsonify({'erro': 'Modelo não encontrado'}), 404

    return jsonify({
        'caminho_modelo': modelo.caminho_modelo,
        'nome_modelo': modelo.nome_modelo
    })
