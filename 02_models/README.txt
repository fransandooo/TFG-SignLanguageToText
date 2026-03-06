================================================================================
                            MODELOS ENTRENADOS
                  Sign Language Recognition System
================================================================================

Esta carpeta contiene los modelos de detección de manos entrenados para el TFG.

================================================================================
                           MODELOS DISPONIBLES
================================================================================

1. yolov8n.pt
   Arquitectura:    YOLOv8 Nano
   Tarea:           Detección de manos (hand detection)
   Tamaño:          ~6.3 MB

   Entrenado en:
   - EgoHands Dataset
   - Kaggle Hand Dataset (armannikkhah/hand-dataset)

   Configuración de entrenamiento:
   - Resolución: 416x416 píxeles
   - Épocas: Ver configuración en 05_experiments/
   - Optimizador: SGD (por defecto YOLO)

   Uso:
   ```python
   from ultralytics import YOLO
   model = YOLO('02_models/yolov8n.pt')
   results = model('imagen.jpg', conf=0.5)
   ```

   Notas:
   - Modelo base del proyecto
   - Usado en experimentos iniciales
   - Precisión: Ver métricas en 05_experiments/hand_bbox_baseline/

--------------------------------------------------------------------------------

2. yolo11n.pt
   Arquitectura:    YOLO11 Nano
   Tarea:           Detección de manos (hand detection)
   Tamaño:          ~5.4 MB

   Entrenado en:
   - Dataset mixto (EgoHands + Kaggle)

   Configuración de entrenamiento:
   - Resolución: 416x416 píxeles
   - Épocas: Ver configuración en 05_experiments/

   Uso:
   ```python
   from ultralytics import YOLO
   model = YOLO('02_models/yolo11n.pt')
   results = model('imagen.jpg', conf=0.5)
   ```

   Notas:
   - Versión actualizada con arquitectura YOLO11
   - Mejoras en velocidad de inferencia
   - Modelo utilizado en el servidor (04_server/main.py)

================================================================================
                          COMPARACIÓN DE MODELOS
================================================================================

Modelo       | Tamaño | Velocidad | Precisión | Uso Recomendado
-------------|--------|-----------|-----------|----------------------------------
yolov8n.pt   | 6.3 MB | Rápido    | Alta      | Experimentos, baseline
yolo11n.pt   | 5.4 MB | Muy rápido| Alta      | Producción, app móvil, servidor

================================================================================
                     CÓMO USAR LOS MODELOS ENTRENADOS
================================================================================

1. INFERENCIA BÁSICA
   ```python
   from ultralytics import YOLO

   # Cargar modelo
   model = YOLO('ruta/al/modelo.pt')

   # Predecir en una imagen
   results = model('imagen.jpg', conf=0.5)

   # Procesar resultados
   for result in results:
       boxes = result.boxes  # Bounding boxes
       for box in boxes:
           x1, y1, x2, y2 = box.xyxy[0]  # Coordenadas
           conf = box.conf[0]             # Confianza
           print(f"Mano detectada: ({x1}, {y1}, {x2}, {y2}) conf={conf}")
   ```

2. INFERENCIA EN VIDEO
   ```python
   from ultralytics import YOLO

   model = YOLO('ruta/al/modelo.pt')
   results = model('video.mp4', conf=0.5, stream=True)

   for frame_idx, result in enumerate(results):
       print(f"Frame {frame_idx}: {len(result.boxes)} manos detectadas")
   ```

3. INFERENCIA EN TIEMPO REAL (WEBCAM)
   ```python
   from ultralytics import YOLO

   model = YOLO('ruta/al/modelo.pt')
   model.predict(source=0, conf=0.5, show=True)  # source=0 es la webcam
   ```

================================================================================
                         MÉTRICAS Y EVALUACIÓN
================================================================================

Para ver las métricas detalladas de cada modelo:

1. Resultados de entrenamiento:
   05_experiments/hand_bbox_baseline/runs_yolo/

2. Comparación de modelos:
   05_experiments/hand_bbox_baseline/hands_model_comparison.csv

3. Gráficas de entrenamiento:
   05_experiments/hand_bbox_baseline/runs_yolo/[nombre_run]/
   - results.png: Gráficas de pérdida y métricas
   - confusion_matrix.png: Matriz de confusión
   - F1_curve.png, PR_curve.png: Curvas de evaluación

================================================================================
                         RE-ENTRENAR MODELOS
================================================================================

Para re-entrenar o fine-tunear estos modelos, consulta los notebooks:

- 01_notebooks/hand_detection/YOLOv8n_Only_Training_Final.ipynb
- 01_notebooks/hand_detection/01_yolo_hands_mixed_fixed.ipynb

Estos notebooks incluyen:
- Descarga y preparación de datasets
- Configuración de hiperparámetros
- Entrenamiento con diferentes learning rates
- Evaluación y comparación de resultados

================================================================================
                            CONFIGURACIÓN
================================================================================

Parámetros recomendados para inferencia:

- Confianza mínima (conf): 0.5 - 0.6
  * Valores más altos: menos falsos positivos, más detecciones perdidas
  * Valores más bajos: más detecciones, más falsos positivos

- IoU threshold (iou): 0.45 (por defecto)
  * Para NMS (Non-Maximum Suppression)

- Tamaño de imagen (imgsz): 416
  * IMPORTANTE: usar el mismo tamaño que en entrenamiento

================================================================================
                          INFORMACIÓN ADICIONAL
================================================================================

Clases detectadas:
- 0: hand (mano)

Formato de salida:
- Bounding boxes en formato XYXY (x1, y1, x2, y2)
- Coordenadas en píxeles relativos a la imagen de entrada
- Confianza entre 0.0 y 1.0

Compatibilidad:
- Python >= 3.8
- ultralytics >= 8.0.0
- torch >= 2.0.0

================================================================================
Última actualización: 2025-12-30
================================================================================
