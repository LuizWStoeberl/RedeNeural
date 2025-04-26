from flask_sqlalchemy import SQLAlchemy
from models import db

db = SQLAlchemy()


class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key= True)
    nome = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    treinamentos = db.relationship("Treinamento", backref= "usuario", lazy=True)

class Treinamento(db.Model):
     id = db.Column(db.Integer, primary_key= True)
     epocas = db.Column(db.Integer, nullable=False)
     neuronios = db.Column(db.Integer, nullable=False)
     enlaces = db.Column(db.Integer, nullable=False)
     resultado = db.Column(db.String(200))
     usuario_id = db.Column(db.Integer, db.ForeignKey("usuario.id"), nullable=True)

     def to_dict(self):
        return {
            "id": self.id,
            "epocas": self.epocas,
            "neuronios": self.neuronios,
            "enlaces": self.enlaces,
            "resultado": self.resultado,
            "usuario_id": self.usuario_id
        }

class IntervaloCor(db.Model):
    __tablename__ = 'intervalos_cor'
    id = db.Column(db.Integer, primary_key=True)
    classe = db.Column(db.String(100), nullable=False)  # Ex: "Bart", "Homer"
    r_min = db.Column(db.Integer, nullable=False)
    r_max = db.Column(db.Integer, nullable=False)
    g_min = db.Column(db.Integer, nullable=False)
    g_max = db.Column(db.Integer, nullable=False)
    b_min = db.Column(db.Integer, nullable=False)
    b_max = db.Column(db.Integer, nullable=False)