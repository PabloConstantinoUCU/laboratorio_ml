# Estimación de edad a partir de imágenes faciales

Laboratorio de Machine Learning / AI. El proyecto entrena un modelo que predice el **rango etario** de una persona a partir de una imagen de su rostro, usando transfer learning con MobileNetV2 y el dataset [UTKFace](https://susanqq.github.io/UTKFace/).

## Rangos de edad

| Clase  | Edad     |
|--------|----------|
| child  | 0–12     |
| young  | 13–25    |
| adult  | 26–59    |
| senior | 60 o más |

## Contenido del repositorio

| Archivo | Descripción |
|---------|-------------|
| `prediccion_edad.py` | Entrena el modelo y lo evalúa con un conjunto de validación. |
| `predecir_edad.py` | Usa el modelo entrenado para predecir imágenes nuevas y genera un reporte. |
| `age_model.keras` | Modelo entrenado (arquitectura y pesos), generado por `prediccion_edad.py`. |

## Archivos generados

Al ejecutar los scripts se generan los siguientes archivos y materiales, que no están incluidos en el repositorio:

| Archivo | Descripción |
|---------|-------------|
| `matriz_confusion.png` | Al terminar el entrenamiento se muestra la matriz de confusión sobre el conjunto de validación. |
| `evolucion.png` | Al terminar el entrenamiento se muestra la gráfica de la evolución de la precisión (entrenamiento vs. validación). |
| `reporte_predicciones.txt` | Luego de ejecutar `predecir_edad.py` se guarda un archivo de texto con un reporte de las predicciones sobre las imágenes de prueba. |

## Requisitos

- Python 3.10 o superior
- Librerías: `tensorflow`, `keras`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

## Dataset

El dataset no se incluye en el repositorio. Descargá UTKFace desde https://susanqq.github.io/UTKFace/ y colocá las imágenes en una carpeta `UTKFace/` junto a los scripts. Cada archivo tiene el formato `edad_genero_raza_fecha.jpg`, y la edad real se toma del inicio del nombre.

## Uso

**1. Entrenar el modelo**

```bash
python prediccion_edad.py
```

Entrena durante 5 épocas con una partición 80 % entrenamiento / 20 % validación, guarda `age_model.keras` y muestra la matriz de confusión, el gráfico de evolución de la precisión y las métricas por clase.

**2. Predecir imágenes nuevas**

Colocá las imágenes en una carpeta `RostrosConocidos/`. Cada archivo debe llamarse `[edad]_[nombre].jpg`, por ejemplo `25_Juan.jpg`, para poder comparar con la edad real. Luego ejecutá:

```bash
python predecir_edad.py
```

Se genera `reporte_predicciones.txt` con la edad real, el rango real, el rango predicho, la probabilidad y el resultado (ACIERTO o FALLO) de cada imagen, además de la precisión total.

## Resultados

Sobre el conjunto de validación (1204 imágenes) el modelo alcanzó una precisión global de **65,7 %**. El detalle por clase, la matriz de confusión y las conclusiones están en el informe del laboratorio.