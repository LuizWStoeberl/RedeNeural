import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
<<<<<<< HEAD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import db

MODELO_SALVO_PATH = 'modelos_salvos/modelo_cnn.h5'
os.makedirs('modelos_salvos', exist_ok=True)

def get_ultimo_treinamento():
    from models import Treinamento
    return Treinamento.query.order_by(Treinamento.id.desc()).first()

def encontrar_ultimo_upload(base_path='arquivosRede2'):
    pastas = [p for p in os.listdir(base_path) if p.startswith('teste')]
    pastas.sort(reverse=True)  # Mais recente primeiro
    if pastas:
        return os.path.join(base_path, pastas[0])
    else:
        raise Exception("Nenhuma pasta de upload encontrada.")

def treinar_rede_neural_cnn():
    # 1. Localizar as pastas automaticamente
    pasta_base = encontrar_ultimo_upload()
    caminho_treinamento = os.path.join(pasta_base, 'treinamento')
    caminho_teste = os.path.join(pasta_base, 'teste')

    # 2. Configuração de pré-processamento
    tamanho_imagem = (150, 150)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        caminho_treinamento,
        target_size=tamanho_imagem,
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        caminho_teste,
        target_size=tamanho_imagem,
        batch_size=32,
        class_mode='binary'
    )

    # 3. Recuperar configuração de treinamento do banco
    config = get_ultimo_treinamento()
    if not config:
        raise Exception("Nenhuma configuração de treinamento encontrada no banco de dados.")

    epocas = config.epocas
    neuronios = config.neuronios
    camadas_convolucionais = config.enlaces  # Aqui está usando o campo 'enlaces' como número de camadas convolucionais

    # 4. Construção da CNN
    modelo_cnn = tf.keras.models.Sequential()

    # Primeira camada convolucional
    modelo_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    modelo_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Demais camadas convolucionais
    for _ in range(camadas_convolucionais - 1):
        modelo_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        modelo_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Camadas finais
    modelo_cnn.add(tf.keras.layers.Flatten())
    modelo_cnn.add(tf.keras.layers.Dense(neuronios, activation='relu'))
    modelo_cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # 5. Compilar e treinar
    modelo_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    historico = modelo_cnn.fit(
        train_generator,
        epochs=epocas,
        validation_data=validation_generator
    )

    # 6. Avaliação
    acc = historico.history['accuracy'][-1]

    # (Confusion matrix opcional se quiser gerar depois via predict)
    # Previsões no conjunto de validação
    y_true = validation_generator.classes
    y_pred = modelo_cnn.predict(validation_generator)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    cm = confusion_matrix(y_true, y_pred_classes)

    # 7. Salvar modelo
    modelo_cnn.save(MODELO_SALVO_PATH)

    return acc, cm
=======

def treinar_rede_neural(caminho_csv, epocas=100, neuronios=64, camadas_ocultas=3):
    """Treina uma rede neural com os parâmetros especificados"""
    
    # Carregar dados
    df = pd.read_csv(caminho_csv)
    X = df.drop('classe', axis=1).values
    y = df['classe'].values - 1  # Converter para 0 e 1
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Construir modelo
    model = Sequential()
    model.add(Dense(neuronios, activation='relu', input_shape=(X_train.shape[1],)))
    
    for _ in range(camadas_ocultas - 1):
        model.add(Dense(neuronios, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Treinar
    history = model.fit(
        X_train, y_train,
        epochs=epocas,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Avaliar
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    return {
        'acuracia': acuracia,
        'matriz_confusao': matriz_confusao.tolist(),
        'epocas': epocas,
        'neuronios': neuronios,
        'camadas': camadas_ocultas,
        'history': history.history
    }
>>>>>>> 583778416cd6d0b10c6af7fe3d86fb3ee3aac280
