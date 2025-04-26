import os
import json
from flask import Blueprint, request, jsonify

bp_cores = Blueprint('cores', __name__)

CORES_FOLDER = 'cores_definidas'
os.makedirs(CORES_FOLDER, exist_ok=True)

@bp_cores.route('/definir_cores', methods=['POST'])
def definir_cores():
    dados = request.json

    classe = dados.get('classe')
    cores = dados.get('cores')  # Lista de dicionários [{"r":255, "g":100, "b":50, "tolerancia":10}, ...]

    if not classe or not cores or len(cores) < 3:
        return jsonify({"erro": "Defina pelo menos três intervalos de cores para cada classe."}), 400

    caminho_arquivo = os.path.join(CORES_FOLDER, f"{classe}.json")

    with open(caminho_arquivo, 'w') as f:
        json.dump(cores, f)

    return jsonify({"mensagem": f"Intervalos de cor para a classe {classe} salvos com sucesso!"})
