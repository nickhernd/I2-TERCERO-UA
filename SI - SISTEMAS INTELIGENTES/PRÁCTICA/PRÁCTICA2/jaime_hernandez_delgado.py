"""
Autor: Jaime Hermández Delgado
DNI: 48776654W
Práctica 2: Visión artificial y aprendizaje
Sistemas Inteligentes - Universidad de Alicante
Curso 2024/2025
"""

# Configuración inicial para suprimir advertencias
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Importaciones necesarias
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
from PIL import Image

# ================== FUNCIONES BÁSICAS ==================
"""
Carga y preprocesa el dataset CIFAR10.
Returns:
    tuple: (X_train, Y_train, X_test, Y_test) normalizados y procesados
"""
def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    # Normalizar valores de píxeles al rango [0,1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convertir etiquetas a codificación one-hot
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    return X_train, Y_train, X_test, Y_test


#Visualiza accuracy y loss en una única gráfica con dos ejes Y.
def visualizar_metricas_combinadas(historia, titulo="Métricas de Entrenamiento"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Graficar accuracy
    ax1.plot(historia.history['accuracy'], 'b-', label='Train')
    ax1.plot(historia.history['val_accuracy'], 'g-', label='Validation')
    ax1.set_title(f'{titulo} - Accuracy')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()

    # Graficar loss
    ax2.plot(historia.history['loss'], 'r-', label='Train')
    ax2.plot(historia.history['val_loss'], 'orange', label='Validation')
    ax2.set_title(f'{titulo} - Loss')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def visualizar_matriz_confusion(Y_true, Y_pred, clases):
    if len(Y_true.shape) > 1:
        Y_true = np.argmax(Y_true, axis=1)
    if len(Y_pred.shape) > 1:
        Y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()

# ================== TAREA A: MLP BÁSICO ==================

def crear_mlp_basico():
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(32, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

# ================== TAREA B: AJUSTE DE EPOCHS ==================

def experimentar_epochs(X_train, Y_train, X_test, Y_test, lista_epochs=[5, 10, 20, 50, 100]):
    resultados = []

    for epochs in lista_epochs:
        print(f"\nEntrenando con {epochs} epochs...")
        modelo = crear_mlp_basico()

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'epochs': epochs,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia, f'Entrenamiento con {epochs} epochs')

    return resultados

# ================== TAREA C: AJUSTE DE BATCH SIZE ==================

def experimentar_batch_sizes(X_train, Y_train, X_test, Y_test,
                           batch_sizes=[32, 64, 128, 256]):
    resultados = []

    for batch_size in batch_sizes:
        print(f"\nEntrenando con batch_size={batch_size}...")
        modelo = crear_mlp_basico()

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,  # Usando el mejor valor de epochs
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'batch_size': batch_size,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia,
                                     f'Entrenamiento con batch_size={batch_size}')

    return resultados

# ================== TAREA D: FUNCIONES DE ACTIVACIÓN ==================

def crear_mlp_con_activacion(activacion):
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(32, activation=activacion),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_activaciones(X_train, Y_train, X_test, Y_test,
                            activaciones=['sigmoid', 'relu', 'tanh']):
    resultados = []

    for activacion in activaciones:
        print(f"\nEntrenando con activación {activacion}...")
        modelo = crear_mlp_con_activacion(activacion)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=128,  # Usando el mejor batch_size
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'activacion': activacion,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia,
                                     f'Entrenamiento con activación {activacion}')

    return resultados

# ================== TAREA E: NÚMERO DE NEURONAS ==================

def crear_mlp_neuronas(num_neuronas):
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(num_neuronas, activation='relu'),  # Usando la mejor activación
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_neuronas(X_train, Y_train, X_test, Y_test,
                         lista_neuronas=[16, 32, 64, 128, 256]):
    resultados = []

    for neuronas in lista_neuronas:
        print(f"\nEntrenando con {neuronas} neuronas...")
        modelo = crear_mlp_neuronas(neuronas)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'neuronas': neuronas,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia,
                                     f'Entrenamiento con {neuronas} neuronas')

    return resultados

# ================== TAREA F: MLP MULTICAPA ==================

def crear_mlp_multicapa(arquitectura):
    modelo = Sequential([Flatten(input_shape=(32, 32, 3))])

    # Añadir capas ocultas
    for neuronas in arquitectura:
        modelo.add(Dense(neuronas, activation='relu'))

    # Capa de salida
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_arquitecturas_mlp(X_train, Y_train, X_test, Y_test):
    arquitecturas = [
        [32],
        [64],
        [32, 32],
        [64, 32],
        [32, 64],
        [64, 64],
        [128, 64, 32]
    ]

    resultados = []

    for arq in arquitecturas:
        nombre_arq = ' -> '.join(map(str, arq))
        print(f"\nEntrenando arquitectura: {nombre_arq}")

        modelo = crear_mlp_multicapa(arq)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'arquitectura': nombre_arq,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia,
                                     f'Arquitectura: {nombre_arq}')

    return resultados

# ================== TAREAS G, H, I: CNN ==================

def crear_cnn_basica(kernel_size=(3,3), usar_maxpool=True):
    modelo = Sequential([
        Conv2D(16, kernel_size, activation='relu', padding='same',
               input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)) if usar_maxpool else Dropout(0.25),

        Conv2D(32, kernel_size, activation='relu', padding='same'),
        MaxPooling2D((2, 2)) if usar_maxpool else Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_kernel_sizes(X_train, Y_train, X_test, Y_test,
                            kernel_sizes=[(3,3), (5,5), (7,7)]):
    resultados = []

    for kernel_size in kernel_sizes:
        print(f"\nEntrenando CNN con kernel_size={kernel_size}")
        modelo = crear_cnn_basica(kernel_size)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'kernel_size': kernel_size,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia,
                                     f'CNN con kernel {kernel_size[0]}x{kernel_size[1]}')

    return resultados

"""
Crea una CNN optimizada con la arquitectura especificada.
Args:
    arquitectura: Lista con el número de filtros para cada capa conv
"""
def crear_cnn_optimizada(arquitectura):
    modelo = Sequential()

    # Primera capa convolucional
    modelo.add(Conv2D(arquitectura[0], (3, 3), activation='relu',
                     input_shape=(32, 32, 3)))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D((2, 2)))

    # Capas convolucionales adicionales
    for filtros in arquitectura[1:]:
        modelo.add(Conv2D(filtros, (3, 3), activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))

    # Capas fully connected
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_arquitecturas_cnn(X_train, Y_train, X_test, Y_test):
    arquitecturas = [
        [32],
        [32, 64],
        [32, 64, 128],
        [64, 128, 256],
        [32, 64, 128, 256]
    ]

    resultados = []

    for arq in arquitecturas:
        nombre_arq = ' -> '.join(map(str, arq))
        print(f"\nEntrenando CNN con arquitectura: {nombre_arq}")

        modelo = crear_cnn_optimizada(arq)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'arquitectura': nombre_arq,
            'historia': historia,
            'test_acc': test_acc,
            'tiempo': tiempo_total
        })

        visualizar_metricas_combinadas(historia, f'CNN: {nombre_arq}')

    return resultados

# ================== TAREAS J, K, L: DATASET PROPIO ==================

def cargar_dataset_propio(directorio):
    clases = ['avion', 'automovil', 'pajaro', 'gato', 'ciervo',
              'perro', 'rana', 'caballo', 'barco', 'camion']

    X = []
    Y = []

    for idx, clase in enumerate(clases):
        dir_clase = os.path.join(directorio, clase)
        if not os.path.exists(dir_clase):
            print(f"No se encontró el directorio para {clase}")
            continue

        for img_name in os.listdir(dir_clase):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img = os.path.join(dir_clase, img_name)
                try:
                    # Cargar y preprocesar imagen
                    img = Image.open(ruta_img)
                    img = img.resize((32, 32))
                    img = img.convert('RGB')
                    img_array = np.array(img) / 255.0

                    X.append(img_array)
                    Y.append(idx)
                except Exception as e:
                    print(f"Error cargando {ruta_img}: {str(e)}")

    return np.array(X), keras.utils.to_categorical(np.array(Y))

def crear_modelo_mejorado():
    modelo = Sequential([
        # Data augmentation
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),

        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Capas fully connected
        Flatten(),
        Dense(128, activation='relu',
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def evaluar_dataset_propio(modelo, X, Y, clases):
    # Evaluar modelo
    test_loss, test_acc = modelo.evaluate(X, Y, verbose=0)
    print(f"\nAccuracy en dataset propio: {test_acc:.4f}")
    print(f"Loss en dataset propio: {test_loss:.4f}")

    # Obtener predicciones
    Y_pred = modelo.predict(X)

    # Visualizar matriz de confusión
    visualizar_matriz_confusion(Y, Y_pred, clases)

    return test_acc, test_loss, Y_pred

# ================== EJECUCIÓN PRINCIPAL ==================

if __name__ == "__main__":
    # Cargar datos CIFAR10
    print("Cargando dataset CIFAR10...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    # Lista de clases para visualización
    clases = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
              'perro', 'rana', 'caballo', 'barco', 'camión']

    # Menú de selección de tarea
    print("\nTareas disponibles:")
    print("A - MLP básico")
    print("B - Ajuste de epochs")
    print("C - Ajuste de batch size")
    print("D - Funciones de activación")
    print("E - Número de neuronas")
    print("F - MLP multicapa")
    print("G - CNN básica")
    print("H - Ajuste de kernel size")
    print("I - CNN optimizada")
    print("J - Dataset propio")
    print("K - Evaluación en dataset propio")
    print("L - Mejoras para generalización")

    tarea = input("\nSeleccionar tarea (A-L): ").upper()

    # Ejecutar tarea seleccionada
    if tarea == 'A':
        modelo = crear_mlp_basico()
        historia = modelo.fit(X_train, Y_train, epochs=50,
                            validation_split=0.2, batch_size=32)
        visualizar_metricas_combinadas(historia)

    elif tarea == 'B':
        resultados = experimentar_epochs(X_train, Y_train, X_test, Y_test)

    elif tarea == 'C':
        resultados = experimentar_batch_sizes(X_train, Y_train, X_test, Y_test)

    elif tarea == 'D':
        resultados = experimentar_activaciones(X_train, Y_train, X_test, Y_test)

    elif tarea == 'E':
        resultados = experimentar_neuronas(X_train, Y_train, X_test, Y_test)

    elif tarea == 'F':
        resultados = experimentar_arquitecturas_mlp(X_train, Y_train, X_test, Y_test)

    elif tarea == 'G':
        modelo = crear_cnn_basica()
        historia = modelo.fit(X_train, Y_train, epochs=50,
                            validation_split=0.2, batch_size=128)
        visualizar_metricas_combinadas(historia)

    elif tarea == 'H':
        resultados = experimentar_kernel_sizes(X_train, Y_train, X_test, Y_test)

    elif tarea == 'I':
        resultados = experimentar_arquitecturas_cnn(X_train, Y_train, X_test, Y_test)

    elif tarea in ['J', 'K', 'L']:
        # Cargar dataset propio
        print("\nCargando dataset propio...")
        X_propio, Y_propio = cargar_dataset_propio('dataset_propio')

        if tarea == 'J':
            print(f"\nDataset propio cargado:")
            print(f"X shape: {X_propio.shape}")
            print(f"Y shape: {Y_propio.shape}")

        elif tarea == 'K':
            # Evaluar varios modelos en el dataset propio
            modelos = {
                'MLP básico': crear_mlp_basico(),
                'CNN básica': crear_cnn_basica(),
                'CNN optimizada': crear_cnn_optimizada([32, 64, 128])
            }

            for nombre, modelo in modelos.items():
                print(f"\nEvaluando {nombre}...")
                modelo.fit(X_train, Y_train, epochs=50,
                         validation_split=0.2, batch_size=128)
                evaluar_dataset_propio(modelo, X_propio, Y_propio, clases)

        elif tarea == 'L':
            # Entrenar modelo mejorado
            print("\nEntrenando modelo mejorado...")
            modelo = crear_modelo_mejorado()
            historia = modelo.fit(X_train, Y_train, epochs=50,
                                validation_split=0.2, batch_size=128)
            evaluar_dataset_propio(modelo, X_propio, Y_propio, clases)
            visualizar_metricas_combinadas(historia, "Modelo Mejorado")

    else:
        print("Tarea no válida")
