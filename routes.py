from flask import Blueprint, render_template, request, jsonify
from models import db, Treinamento
from datetime import datetime
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

    nome_arquivo = "tabela_cores.csv" 
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
        from rede_neural import treinar_rede_neural
        acc, cm = treinar_rede_neural()
        return jsonify({"mensagem": "Treinamento concluido", "acuracia": acc})
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
    
@bp.route('/listar_treinamentos', methods=['GET'])
def listar_treinamentos():
    treinamentos = Treinamento.query.order_by(Treinamento.id.desc()).all()
    return jsonify([t.to_dict() for t in treinamentos])

@bp.route("/teste_salvar", methods=["GET"])
def teste_salvar():
    novo = Treinamento(epocas=5, neuronios=32, enlaces=2)
    db.session.add(novo)
    db.session.commit()
    return jsonify({"mensagem": "Salvo com sucesso!", "id": novo.id})

@bp.route("/criar_usuario", methods=["GET"])
def criar_usuario():
    from models import Usuario, db

    novo_usuario = Usuario(nome="Usuário Teste", email="teste@exemplo.com")
    db.session.add(novo_usuario)
    db.session.commit()

    return jsonify({"mensagem": "Usuário criado com sucesso!", "id": novo_usuario.id})
