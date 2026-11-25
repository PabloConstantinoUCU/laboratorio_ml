"""Predicción por lote de los rostros en ``RostrosConocidos``.

Uso:
    python predecir_edad.py

Los archivos deben llamarse ``[edad]_[nombre].jpg``, por ejemplo ``25_Juan.jpg``.
Se genera el reporte ``reporte_predicciones.txt`` con la edad real, el rango
real y el rango predicho para cada imagen.
"""

import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = (128, 128)
CLASES = ["child", "young", "adult", "senior"]
MODEL_PATH = "age_model.keras"
ROSTROS_DIR = "RostrosConocidos"
REPORTE_PATH = "reporte_predicciones.txt"
EXTENSIONES = (".jpg", ".jpeg", ".png")


def categorize_age(age):
    """Mismos umbrales usados al entrenar el modelo."""
    if age <= 12:
        return "child"
    elif age <= 25:
        return "young"
    elif age <= 59:
        return "adult"
    else:
        return "senior"


def predict_age_group(model, image_path):
    """Predice el rango etario de una imagen de rostro."""
    # imdecode + fromfile tolera mejor rutas con acentos y distintos formatos.
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    # Mismo preprocesamiento usado durante el entrenamiento.
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
    probabilities = model.predict(np.expand_dims(img / 255.0, axis=0), verbose=0)[0]
    return CLASES[int(np.argmax(probabilities))], float(np.max(probabilities))


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH}. Colócalo en la misma carpeta que este script."
        )

    if not os.path.isdir(ROSTROS_DIR):
        raise FileNotFoundError(f"No se encontró la carpeta: {ROSTROS_DIR}")

    model = load_model(MODEL_PATH)

    lineas = [
        "Reporte de predicciones - RostrosConocidos",
        f"Modelo: {MODEL_PATH}",
        "=" * 60,
    ]
    aciertos = 0
    procesadas = 0
    omitidas = 0
    con_error = 0

    for filename in sorted(os.listdir(ROSTROS_DIR)):
        if not filename.lower().endswith(EXTENSIONES):
            continue

        partes = filename.split("_", 1)
        try:
            edad_real = int(partes[0])
        except (ValueError, IndexError):
            omitidas += 1
            print(f"Omitido (nombre no válido): {filename}")
            continue

        nombre = os.path.splitext(partes[1])[0] if len(partes) > 1 else ""
        rango_real = categorize_age(edad_real)

        try:
            rango_predicho, probabilidad = predict_age_group(
                model, os.path.join(ROSTROS_DIR, filename)
            )
        except (FileNotFoundError, cv2.error, ValueError) as exc:
            con_error += 1
            print(f"Omitido (no se pudo leer): {filename} - {exc}")
            lineas.extend(
                [
                    f"Archivo: {filename}",
                    f"Nombre: {nombre}",
                    f"Edad real: {edad_real}",
                    f"Rango real: {rango_real}",
                    "Rango predicho: ERROR",
                    "Error: no se pudo leer la imagen (formato no soportado o dañado)",
                    "-" * 60,
                ]
            )
            continue

        acierto = rango_predicho == rango_real
        aciertos += int(acierto)
        procesadas += 1

        lineas.extend(
            [
                f"Archivo: {filename}",
                f"Nombre: {nombre}",
                f"Edad real: {edad_real}",
                f"Rango real: {rango_real}",
                f"Rango predicho: {rango_predicho}",
                f"Probabilidad: {probabilidad:.2%}",
                f"Resultado: {'ACIERTO' if acierto else 'FALLO'}",
                "-" * 60,
            ]
        )
        print(
            f"{filename} | real={rango_real} | predicho={rango_predicho} "
            f"| {probabilidad:.2%} | {'ACIERTO' if acierto else 'FALLO'}"
        )

    precision = aciertos / procesadas if procesadas else 0.0
    lineas.extend(
        [
            f"Total de imágenes procesadas: {procesadas}",
            f"Aciertos: {aciertos}",
            f"Precisión: {precision:.2%}",
            f"Archivos omitidos (nombre no válido): {omitidas}",
            f"Archivos con error de lectura: {con_error}",
        ]
    )

    with open(REPORTE_PATH, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")

    print(f"\nReporte guardado en: {REPORTE_PATH}")
    print(f"Aciertos: {aciertos}/{procesadas} ({precision:.2%})")


if __name__ == "__main__":
    main()
