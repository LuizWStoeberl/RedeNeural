from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

db = SQLAlchemy()

class Treinamento(db.Model):
    __tablename__ = 'treinamentos'
    id = db.Column(db.Integer, primary_key=True)
    epocas = db.Column(db.Integer, nullable=False)
    neuronios = db.Column(db.Integer, nullable=False)
    enlaces = db.Column(db.Integer, nullable=False)
    resultado = db.Column(db.String(100))

class IntervaloCor(db.Model):
    __tablename__ = 'intervalos_cor'
    id = db.Column(db.Integer, primary_key=True)
    classe = db.Column(db.String(100), nullable=False)
    r_min = db.Column(db.Integer, nullable=False)
    r_max = db.Column(db.Integer, nullable=False)
    g_min = db.Column(db.Integer, nullable=False)
    g_max = db.Column(db.Integer, nullable=False)
    b_min = db.Column(db.Integer, nullable=False)
    b_max = db.Column(db.Integer, nullable=False)

class ModeloTreinado(db.Model):
    __tablename__ = 'modelos_treinados'
    id = db.Column(db.Integer, primary_key=True)
    nome_modelo = db.Column(db.String(200), nullable=False)
    caminho_modelo = db.Column(db.String(300), nullable=False)
    data_treinamento = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    tipo_modelo = db.Column(db.String(50), nullable=False)
    resultado = db.Column(db.String(100))

    @staticmethod
    def salvar_modelo(nome_arquivo, tipo_modelo, resultado):
        novo_modelo = ModeloTreinado(
            nome_modelo=nome_arquivo,
            caminho_modelo=os.path.join('modelos_salvos', nome_arquivo),
            data_treinamento=datetime.utcnow(),
            tipo_modelo=tipo_modelo,
            resultado=resultado
        )
        db.session.add(novo_modelo)
        db.session.commit()

class ClassePersonagem(db.Model):
    __tablename__ = 'classes_personagens'
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False, unique=True)
    data_criacao = db.Column(db.DateTime, default=datetime.utcnow)

    # Relação com os modelos treinados (opcional)
    modelos = db.relationship('ModeloTreinado', backref='classe', lazy=True)
