from flask import Blueprint, request, redirect, url_for, render_template, flash, session
import random 
import shutil
import time
from models import db, Treinamento
from datetime import datetime
from processar_imagem import processar_todas_imagens
from cnn_model import treinar_rede_neural_cnn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

bp = Blueprint("routes", __name__)

#Criando as pastas e toda lógica necessaria para elas funcionarem

ARQUIVOSREDE1_DIR = 'arquivosRede1'
if not os.path.exists(ARQUIVOSREDE1_DIR):
    os.makedirs(ARQUIVOSREDE1_DIR)
def criar_nova_pasta_teste(base_dir):
    i = 1
    while True:
        nome_pasta = f"Teste{i}"
        caminho_completo = os.path.join(base_dir, nome_pasta)
        if not os.path.exists(caminho_completo):
            os.makedirs(caminho_completo)
            return caminho_completo
        i += 1



UPLOAD_FOLDER = 'arquivosRede2'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def criar_nova_pasta_teste(base_dir):
    i = 1
    while True:
        nome_pasta = f"Teste{i}"
        caminho_completo = os.path.join(base_dir, nome_pasta)
        if not os.path.exists(caminho_completo):
            os.makedirs(caminho_completo)
            return caminho_completo
        i += 1


#Aqui começam as rotas

@bp.route("/")
def home():
   return render_template("home.html")

@bp.route("/rede1")
def rede1():
    return render_template("rede1.html")

@bp.route("/templates/rede2.html")
def rede2():
   return render_template("rede2.html")

@bp.route("/templates/variaveisRede1.html")
def variaveisRede1():
   return render_template("variaveisRede1.html")

@bp.route("/templates/variaveisRede2.html")
def variaveisRede2():
   return render_template("variaveisRede2.html")

@bp.route("/templates/resultadoRede1.html")
def resultadoRede1():
    return render_template("resultadoRede1.html")

@bp.route("/templates/resultadoRede2.html")
def resultadoRede2():
    return render_template("resultadoRede2.html")



#AQUI COMEÇAM AS ROTAS MAIS COMPLEXAS


# --> REDE 1 <--

ultimo_upload_caminho = None  

@bp.route('/upload_pasta_atributos', methods=['POST'])
def upload_pasta_atributos():
    global ultimo_upload_caminho

    # Obter os arquivos enviados
    arquivos_classe1 = request.files.getlist('arquivos1[]')
    arquivos_classe2 = request.files.getlist('arquivos2[]')

    # Validação básica
    if not arquivos_classe1 or not arquivos_classe2:
        flash('Envie arquivos para ambas as classes!', 'erro')
        return redirect(url_for('principal.index'))

    # Pasta de destino (cria se não existir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_upload = os.path.join("arquivosRede1", f"upload_{timestamp}")
    os.makedirs(pasta_upload, exist_ok=True)

    # Função para sanitizar nomes de arquivo
    def sanitizar_nome(nome):
        # Remove barras e caracteres problemáticos
        return nome.replace('/', '_').replace('\\', '_')

    # Salvar arquivos nas pastas das classes
    def salvar_arquivos(arquivos, classe):
        pasta_classe = os.path.join(pasta_upload, classe)
        os.makedirs(pasta_classe, exist_ok=True)
        for arquivo in arquivos:
            if arquivo.filename.strip():
                nome_sanitizado = sanitizar_nome(arquivo.filename)
                caminho_completo = os.path.join(pasta_classe, nome_sanitizado)
                arquivo.save(caminho_completo)

    salvar_arquivos(arquivos_classe1, "Classe1")
    salvar_arquivos(arquivos_classe2, "Classe2")

    # Distribuir entre treinamento/teste
    distribuir_arquivos_em_treinamento_e_teste(pasta_upload)

    # Atualizar variável global
    ultimo_upload_caminho = pasta_upload
    flash('Arquivos distribuídos com sucesso!', 'sucesso')
    return redirect(url_for('routes.variaveisRede1'))

def distribuir_arquivos_em_treinamento_e_teste(pasta_principal):
    # Pastas de origem
    pasta_classe1 = os.path.join(pasta_principal, "Classe1")
    pasta_classe2 = os.path.join(pasta_principal, "Classe2")

    # Pastas de destino (treinamento/teste)
    pasta_treinamento = os.path.join(pasta_principal, "treinamento")
    pasta_teste = os.path.join(pasta_principal, "teste")

    # Processa cada classe
    for classe in ["Classe1", "Classe2"]:
        pasta_origem = os.path.join(pasta_principal, classe)
        
        # Filtra apenas imagens
        extensoes = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        arquivos = [f for f in os.listdir(pasta_origem) 
                   if os.path.isfile(os.path.join(pasta_origem, f)) and 
                   f.lower().endswith(extensoes)]
        
        if not arquivos:
            continue

        # Embaralha e divide 80/20
        random.shuffle(arquivos)
        div = int(0.8 * len(arquivos))

        # Move para treinamento (80%)
        os.makedirs(os.path.join(pasta_treinamento, classe), exist_ok=True)
        for arquivo in arquivos[:div]:
            shutil.move(
                os.path.join(pasta_origem, arquivo),
                os.path.join(pasta_treinamento, classe, arquivo)
            )

        # Move para teste (20%)
        os.makedirs(os.path.join(pasta_teste, classe), exist_ok=True)
        for arquivo in arquivos[div:]:
            shutil.move(
                os.path.join(pasta_origem, arquivo),
                os.path.join(pasta_teste, classe, arquivo)
            )

    print(f"Distribuição concluída em: {pasta_principal}")


@bp.route('/enviar', methods=['POST'])
def enviar():
    # Capturando os dados do formulário
    epocas = request.form.get('epocas')
    neuronios = request.form.get('neuronios')
    enlaces = request.form.get('enlaces')

    # Validação simples para garantir que os campos não estão vazios
    if not epocas or not neuronios or not enlaces:
        flash('Por favor, preencha todos os campos.')
        return redirect(url_for('routes.rede_neural1'))
    
    try:
        epocas = int(epocas)
        neuronios = int(neuronios)
        enlaces = int(enlaces)
        # Chama a função de treinamento
        acc, cm = treinar_rede_neural()
        # Salva os parâmetros no banco
        novo_treinamento = Treinamento(
            epocas=epocas,
            neuronios=neuronios,
            enlaces=enlaces,
            resultado=f"Accuracy: {acc:.4f}"
        )
        db.session.add(novo_treinamento)
        db.session.commit()
        # Renderiza o template com os resultados
        return render_template('resultadoRede1.html', 
                             acuracia=acc, 
                             matriz_confusao=cm.tolist(),
                             epocas=epocas,
                             neuronios=neuronios,
                             enlaces=enlaces)
    except Exception as e:
        flash(f'Erro ao treinar a rede: {str(e)}')
        return redirect(url_for('routes.variaveisRede1'))

    # Armazenar os dados na sessão
 #   session['dados_rede_neural'] = {
  #      'epocas': epocas,
    #    'neuronios': neuronios,
   #     'enlaces': enlaces
   # }

   # flash('Dados armazenados com sucesso!')

    # Redireciona para a próxima etapa
   # return redirect(url_for('routes.resultadoRede1')) 



# --> REDE 2 <--

@bp.route('/upload', methods=['POST'])
def upload():
    num_pastas = int(request.form['num_pastas'])
    timestamp = str(int(time.time()))  # Usar o timestamp para nomear a pasta 'testeN'
    pasta_principal = os.path.join(UPLOAD_FOLDER, f'teste{timestamp}')
    os.makedirs(pasta_principal, exist_ok=True)

    for i in range(1, num_pastas + 1):
        pasta_nome = request.form[f'classe{i}']
        pasta_destino = os.path.join(pasta_principal, pasta_nome)
        os.makedirs(pasta_destino, exist_ok=True)

        # Criando as pastas 'treinamento' e 'teste' dentro de cada classe
        pasta_treinamento = os.path.join(pasta_destino, 'treinamento')
        pasta_teste = os.path.join(pasta_destino, 'teste')
        os.makedirs(pasta_treinamento, exist_ok=True)
        os.makedirs(pasta_teste, exist_ok=True)

        # Salvar arquivos e dividi-los entre treinamento e teste
        arquivos = request.files.getlist(f'upload{i}')
        
        # Dividir os arquivos em 80% para treinamento e 20% para teste
        random.shuffle(arquivos)  # Embaralha a lista de arquivos

        num_treinamento = int(len(arquivos) * 0.8)  # 80% para treinamento
        arquivos_treinamento = arquivos[:num_treinamento]
        arquivos_teste = arquivos[num_treinamento:]

        # Salvar arquivos na pasta 'treinamento'
        for arquivo in arquivos_treinamento:
            caminho_arquivo = os.path.join(pasta_treinamento, arquivo.filename)
            arquivo.save(caminho_arquivo)

        # Salvar arquivos na pasta 'teste'
        for arquivo in arquivos_teste:
            caminho_arquivo = os.path.join(pasta_teste, arquivo.filename)
            arquivo.save(caminho_arquivo)

    return redirect(url_for('routes.variaveisRede2'))


@bp.route('/enviar2', methods=['POST'])
def enviar2():
    # Capturando os dados do formulário
    epocas = request.form.get('epocas')
    neuronios = request.form.get('neuronios')
    camadas = request.form.get('camadas')

    # Validação simples para garantir que os campos não estão vazios
    if not epocas or not neuronios or not camadas:
        flash('Por favor, preencha todos os campos.')
        return redirect(url_for('routes.rede_neural1'))
    
    try:
        epocas = int(epocas)
        neuronios = int(neuronios)
        camadas = int(camadas)
        # Chama a função de treinamento CNN
        acc, cm = treinar_rede_neural_cnn()
        # Salva os parâmetros no banco
        novo_treinamento = Treinamento(
            epocas=epocas,
            neuronios=neuronios,
            enlaces=camadas,  # Usando enlaces para camadas convolucionais
            resultado=f"Accuracy: {acc:.4f}"
        )
        db.session.add(novo_treinamento)
        db.session.commit()
        # Renderiza o template com os resultados
        return render_template('resultadoRede2.html', 
                             acuracia=acc, 
                             matriz_confusao=cm.tolist(),
                             epocas=epocas,
                             neuronios=neuronios,
                             camadas=camadas)
    except Exception as e:
        flash(f'Erro ao treinar a CNN: {str(e)}')
        return redirect(url_for('routes.variaveisRede2'))

    # Armazenar os dados na sessão
   # session['dados_rede_neural'] = {
   #     'epocas': epocas,
    #    'neuronios': neuronios,
    #    'camadas': camadas
   # }

   # flash('Dados armazenados com sucesso!')

    # Redireciona para a próxima etapa
    #return redirect(url_for('routes.resultadoRede2'))





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
