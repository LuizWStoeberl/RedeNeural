import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

bp_uploads = Blueprint('uploads', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@bp_uploads.route('/upload_pastas', methods=['POST'])
def upload_pastas():
    if 'arquivos' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400

    arquivos = request.files.getlist('arquivos')

    if len(arquivos) < 2:
        return jsonify({"erro": "Envie pelo menos duas pastas (classes)"}), 400

    for arquivo in arquivos:
        filename = secure_filename(arquivo.filename)

        # Cada arquivo vem no formato "classe/nome_imagem"
        caminho_classe = os.path.join(UPLOAD_FOLDER, os.path.dirname(filename))
        os.makedirs(caminho_classe, exist_ok=True)

        caminho_completo = os.path.join(UPLOAD_FOLDER, filename)
        arquivo.save(caminho_completo)

    return jsonify({"mensagem": "Pastas e arquivos enviados com sucesso!"})