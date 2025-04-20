from flask import Blueprint, render_template, request, jsonify
from RedeNeural.models import db, Treinamento
from datetime import datetime
import os

bp = Blueprint("routes", __name__)

ARQUIVOS_DIR = 'arquivos'
if not os.path.exists(ARQUIVOS_DIR):
    os.makedirs(ARQUIVOS_DIR)

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