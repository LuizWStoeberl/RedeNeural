import os
import json
from flask import Blueprint, request, jsonify
from models import db, IntervaloCor

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

bp_cores = Blueprint('cores', __name__)

def hex_para_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b

@bp_cores.route('/definir_cores_hex', methods=['POST'])
def definir_cores_hex():
    dados = request.json

    for intervalo in dados:
        classe = intervalo['classe']
        cor_min = intervalo['cor_min']  # Ex: "#FFAABB"
        cor_max = intervalo['cor_max']  # Ex: "#FFEEDD"

        r_min, g_min, b_min = hex_para_rgb(cor_min)
        r_max, g_max, b_max = hex_para_rgb(cor_max)

        cor_obj = IntervaloCor(
            classe=classe,
            r_min=r_min, r_max=r_max,
            g_min=g_min, g_max=g_max,
            b_min=b_min, b_max=b_max
        )
        db.session.add(cor_obj)

    db.session.commit()
    return jsonify({'mensagem': 'Cores HEX convertidas e salvas com sucesso!'})


