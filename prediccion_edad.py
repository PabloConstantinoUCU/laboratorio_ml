"""Fase 1: entrenamiento y guardado del modelo de estimación de edad.

Implementa los pasos 1 a 5 definidos en ``laboratorio_ml-ai.md`` y guarda
el modelo entrenado en ``age_model.keras`` para poder usarlo sin el dataset.

Las imágenes se leen del disco por lotes (no se cargan todas en memoria),
para poder entrenar con miles de fotos sin agotar la RAM.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence, to_categorical  # to_categorical: paso 3

DATA_DIR = "UTKFace"
IMG_SIZE = (128, 128)
CLASES = ["child", "young", "adult", "senior"]
MODEL_PATH = "age_model.keras"
EPOCHS = 5
BATCH_SIZE = 32


class FaceDataGenerator(Sequence):
    """Carga y preprocesa imágenes solo cuando el modelo las necesita."""

    def __init__(self, paths, labels, indexes, batch_size=32, shuffle=False):
        self.paths = paths
        self.labels = labels
        self.indexes = np.array(indexes, dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        batch_idx = self.indexes[start : start + self.batch_size]

        X = np.zeros((len(batch_idx), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        y_batch = np.zeros((len(batch_idx), len(CLASES)), dtype=np.float32)

        for i, idx in enumerate(batch_idx):
            img = cv2.imdecode(np.fromfile(self.paths[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            y_batch[i] = self.labels[idx]

        return X, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# --- 1. Cargar el dataset -------------------------------------------------
image_paths, ages = [], []

for filename in sorted(os.listdir(DATA_DIR)):
    try:
        age = int(filename.split("_")[0])
    except (ValueError, IndexError):
        continue

    path = os.path.join(DATA_DIR, filename)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        image_paths.append(path)  # solo guardamos la ruta, no la imagen
        ages.append(age)

df = pd.DataFrame({"age": ages})


# --- 2. Crear categorías de edad -----------------------------------------
def categorize_age(age):
    """Umbrales definidos en la descripción del laboratorio."""
    if age <= 12:
        return "child"
    elif age <= 25:
        return "young"
    elif age <= 59:
        return "adult"
    else:
        return "senior"


df["age_group"] = df["age"].apply(categorize_age)


# --- 3. Preparar datos para entrenamiento --------------------------------
y = (
    pd.get_dummies(df["age_group"])
    .reindex(columns=CLASES, fill_value=0)
    .values.astype("float32")
)

indices = np.arange(len(image_paths))
try:
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )
except ValueError:
    # Con pocas imágenes (p. ej. 10) no hay muestras suficientes por clase.
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

train_gen = FaceDataGenerator(image_paths, y, train_idx, BATCH_SIZE, shuffle=True)
val_gen = FaceDataGenerator(image_paths, y, test_idx, BATCH_SIZE, shuffle=False)


# --- 4. Modelo base (transfer learning) ----------------------------------
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False  # Congelar capas base

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(4, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def on_epoch_begin(epoch, logs=None):
    print(f"Iniciando época {epoch + 1}/{EPOCHS}...", flush=True)


def on_epoch_end(epoch, logs=None):
    print(
        f"Época {epoch + 1}/{EPOCHS} terminada | "
        f"loss={logs.get('loss', float('nan')):.4f} | "
        f"accuracy={logs.get('accuracy', float('nan')):.4f} | "
        f"val_accuracy={logs.get('val_accuracy', float('nan')):.4f}",
        flush=True,
    )


progress_callback = LambdaCallback(
    on_train_begin=lambda logs: print(f"Entrenando {EPOCHS} épocas...", flush=True),
    on_epoch_begin=on_epoch_begin,
    on_epoch_end=on_epoch_end,
    on_train_end=lambda logs: print("Entrenamiento finalizado.", flush=True),
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[progress_callback],
    verbose=0,
)

# Guardar el modelo entrenado: aquí quedan los pesos y la arquitectura.
model.save(MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")


# --- 5. Evaluación y visualización ---------------------------------------
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.title("Evolución de la precisión")
plt.show()

loss, accuracy = model.evaluate(val_gen, verbose=0)
print(f"Pérdida en test: {loss:.4f} | Precisión en test: {accuracy:.4f}")

y_true = y[test_idx].argmax(axis=1)
y_pred = model.predict(val_gen, verbose=0).argmax(axis=1)

print(
    classification_report(
        y_true, y_pred, target_names=CLASES, zero_division=0
    )
)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES).plot()
plt.title("Matriz de confusión")
plt.show()

# Ejemplo de predicción (paso 5 del laboratorio)
preview_gen = FaceDataGenerator(
    image_paths, y, test_idx[:5], batch_size=5, shuffle=False
)
pred = model.predict(preview_gen, verbose=0)
for i, probabilities in enumerate(pred):
    clase = CLASES[int(np.argmax(probabilities))]
    print(f"Imagen {i + 1}: {clase} ({np.max(probabilities):.2%})")
