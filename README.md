# Football Vision

Un repositorio de visión por computador para detección de jugadores, pelota y keypoints en deportes.

## Estructura del proyecto

- `notebooks/` - Carpeta que contiene los notebooks de entrenamiento e inferencia.
  - `notebooks/YOLO.ipynb` - Notebook principal para entrenar, validar y ejecutar inferencia con YOLOv8.
  - `notebooks/Codigo_Pelota.ipynb` - Notebook enfocado en detección de pelota.
  - `notebooks/Player_identification.ipynb` - Notebook de identificación de jugadores.
  - `notebooks/KeypointsCoco.ipynb` - Notebook para keypoints con COCO.
- `datasets/` - Conjuntos de datos personalizados para entrenamiento y validación.
- `models/` - Pesos de modelos y resultados de entrenamiento.
- `yolo/` - Experimentos de YOLO y pesos asociados.
- `train2/`, `train4/` - Registros y pesos de entrenamiento.
- `videos/` - Videos de entrenamiento y prueba.
- `frames/` - Capturas y frames generados.
- `requirements.txt` - Dependencias de Python.

## Qué contiene este repositorio

- Código y notebooks para entrenar y evaluar modelos de detección de jugadores y keypoints.
- Ejemplos de inferencia en vídeo con OpenCV y YOLOv8.
- Configuraciones de dataset en formato YAML para Ultralytics YOLO.
- Documentación básica para reproducir el proyecto.

## Requisitos

- Python 3.10+ recomendado.
- `pip install -r requirements.txt`
- `ultralytics` para YOLOv8.
- `opencv-python`, `torch`, `numpy`, y demás paquetes listados en `requirements.txt`.

## Uso básico

1. Clonar el repositorio:

```bash
git clone <tu-repositorio>
cd Football
```

2. Crear e instalar el entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Abrir y ejecutar los notebooks en la carpeta `notebooks/`:

- `notebooks/YOLO.ipynb`
- `notebooks/Player_identification.ipynb`
- `notebooks/Codigo_Pelota.ipynb`
- `notebooks/KeypointsCoco.ipynb`

4. Entrenar un modelo YOLOv8:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='./datasets/players/data.yaml', epochs=50)
```

5. Validar el modelo:

```python
metrics = model.val()
print(metrics)
```

6. Exportar el modelo:

```python
model.save('yolov8n_exported.pt')
```

## Datos y modelos

> Para mantener el repositorio ligero, los modelos entrenados y los datasets pesados se excluyen mediante `.gitignore`.

- Descarga o genera tus propios datasets en `datasets/`.
- Guarda los pesos entrenados en `models/` o `yolo/`.
- No incluyas archivos `.pt`, `.pth`, ni carpetas de datos grandes.

## Ejecución de vídeo

Ejemplo de inferencia con vídeo en `YOLO.ipynb`:

- Abrir el notebook.
- Cargar el modelo deseado (`train4/weights/best.pt`, `yolov8n_exported.pt`, etc.).
- Ejecutar la celda de inferencia y guardar el vídeo de salida.

## Enlaces útiles

- Dataset de basketball en Roboflow: https://universe.roboflow.com/zy-vevvi/basketball-g7knr
- Dataset de keypoints basketball court: https://universe.roboflow.com/project/basketball_250_pt2-f2r5y/dataset/1
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

## Git remoto y push

```bash
git remote add origin https://github.com/Fyrthuz/Football-Vision.git
git branch -M main
git push -u origin main
```

@file:output3.mp4

## Notas

- Si agregas datos o modelos locales, asegúrate de que no estén trackeados en Git antes de hacer push.
- <span style="color:red; font-weight:bold;">ADVERTENCIA: Tras la reestructuración de carpetas, algunos imports en los notebooks pueden fallar. Revisa y ajusta las rutas de los imports antes de ejecutar el código.</span>
- Recomendado: usa `git status` para revisar cambios antes de subir.
