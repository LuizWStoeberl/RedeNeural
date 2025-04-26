from flask import Blueprint, render_template, request, jsonify
from models import db, Treinamento
from datetime import datetime
from processar_imagem import processar_todas_imagens
from cnn_model import treinar_cnn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

bp = Blueprint("routes", __name__)

ARQUIVOS_DIR = 'arquivos'
if not os.path.exists(ARQUIVOS_DIR):
    os.makedirs(ARQUIVOS_DIR)


UPLOAD_FOLDER = "arquivoUsuario"
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


@bp.route("/")
def home():
   return render_template("home.html")

@bp.route("/templates/rede1.html")
def rede1():
   return render_template("rede1.html")

@bp.route("/templates/rede2.html")
def rede2():
   return render_template("rede2.html")

@bp.route("/templates/variaveisRede1.html")
def variaveisRede1():
   return render_template("variaveisRede1.html")

@bp.route('/enviar', methods=['POST'])
def processar():
    epocas = int(request.form['epocas'])
    neuronios = int(request.form['neuronios'])
    enlaces = int(request.form['enlaces'])

    from rede_neural import processar_dados
    acc, cm = processar_dados(epocas, neuronios, enlaces)

    return jsonify({"mensagem": "Treinamento concluído", "acuracia": acc, "matriz_confusao": cm.tolist()})

@bp.route('/salvar', methods=['POST'])
def salvar():
    dados = request.json
    linha_count = dados.get("linhaCount", 0)
    quantidade_por_linha = dados.get("quantidadePorLinha", 0)
    cores = dados.get("cores", [])
    labels = dados.get("labels", []) 

    nome_arquivo = "tabela_cores.csv"  # Nome fixo aqui!
    caminho_arquivo = os.path.join(ARQUIVOS_DIR, nome_arquivo)

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

@bp.route('/upload_classes', methods=['POST'])
def upload_classes():
    nomes_classes = request.form.getlist('classes[]')
    arquivos_lista = request.files.getlist('arquivos[]')

    ultima_pasta_teste = encontrar_ultima_pasta_teste(UPLOAD_FOLDER)
    if not ultima_pasta_teste:
        ultima_pasta_teste = criar_nova_pasta_teste(UPLOAD_FOLDER)

    # Como cada input de arquivos está em sequência, emparelhamos classe e arquivos por índice
    from werkzeug.datastructures import MultiDict
    arquivos_form = MultiDict(request.files)

    for idx, nome_classe in enumerate(nomes_classes):
        arquivos_para_classe = arquivos_form.getlist('arquivos[]')[idx::len(nomes_classes)]

        destino_classe = os.path.join(ultima_pasta_teste, nome_classe)
        os.makedirs(destino_classe, exist_ok=True)

        for arquivo in arquivos_para_classe:
            caminho_arquivo = os.path.join(destino_classe, arquivo.filename)
            os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
            arquivo.save(caminho_arquivo)

    global numero_de_classes
    numero_de_classes = len(nomes_classes)
    
    return render_template("variaveisRede2.html")


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
        caminho_temporario = os.path.join('uploads_imagens', arquivo.filename)
        arquivo.save(caminho_temporario)

        modelo = load_model('modelo_cnn_salvo.h5')

        img = image.load_img(caminho_temporario, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predicao = modelo.predict(img_array)
        classe_predita = np.argmax(predicao)

        return jsonify({"classe_predita": int(classe_predita)})
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