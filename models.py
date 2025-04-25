from flask_sqlalchemy import SQLAlchemy

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