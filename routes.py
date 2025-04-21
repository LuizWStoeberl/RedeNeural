from flask import Blueprint, render_template, request, jsonify
from models import db, Treinamento
from datetime import datetime
import os

from teste import processar_dados

bp = Blueprint("routes", __name__)

ARQUIVOS_DIR = 'arquivos'
if not os.path.exists(ARQUIVOS_DIR):
    os.makedirs(ARQUIVOS_DIR)

UPLOAD_FOLDER = "arquivoUsuario"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
   epoca = request.form['epoca']
   neuronios = request.form['neuronios']
   enlaces = request.form['enlaces']

   processar_dados(epoca, neuronios, enlaces)   

   return "Deu certo!"

@bp.route("/templates/variaveisRede2.html")
def variaveisRede2():
   return render_template("variaveisRede2.html")

@bp.route('/salvar', methods=['POST'])
def salvar():
    dados = request.json
    linha_count = dados.get("linhaCount", 0)
    quantidade_por_linha = dados.get("quantidadePorLinha", 0)
    cores = dados.get("cores", [])
    labels = dados.get("labels", []) 

    nome_arquivo = f"tabela_cores_{linha_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    caminho_arquivo = os.path.join(ARQUIVOS_DIR, nome_arquivo)

    with open(caminho_arquivo, 'w') as f:
        header = [f"Cor{i+1}" for i in range(quantidade_por_linha)] + ["Classe"]
        f.write(','.join(header) + '\n')
 
        for linha_cores, classe in zip(cores, labels):
            linha_formatada = ','.join(linha_cores) + f",{classe}\n"
            f.write(linha_formatada)

    return {'message': 'Arquivo salvo com sucesso!'}


@bp.route('/upload', methods=['POST'])
def upload():
    arquivos = request.files.getlist('arquivos')

    for arquivo in arquivos:
        caminho = os.path.join(UPLOAD_FOLDER, arquivo.filename)


        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        arquivo.save(caminho)

    return 'Arquivos enviados com sucesso!'
