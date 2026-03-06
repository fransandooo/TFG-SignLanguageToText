# GUÍA DE NOTEBOOKS - TFG Sign Language Recognition

Esta guía explica qué hace cada notebook y cuáles son los "definitivos" que debes usar.

---

## 📁 PREPARACIÓN DE DATASETS (`01_notebooks/dataset_preparation/`)

### ✅ **USAR ESTE:** `Datasetv2.ipynb`
**Propósito:** Preparación completa del dataset combinado para entrenamiento de reconocimiento de letras

**Qué hace:**
- Descarga automáticamente (con kagglehub) dos datasets:
  - Sign Language MNIST (24 letras en CSV 28x28)
  - ASL Alphabet (imágenes RGB de letras ASL)
- Convierte MNIST CSV → PNG en escala de grises
- Convierte ASL Alphabet JPG RGB → PNG escala de grises
- **Organiza TODO por letras (A, B, C, ... Y)** en carpetas
- Crea estructura unificada: `Dataset/SignLanguageMNIST_ASLAlphabet/train/` y `test/`
- Resultado: ~79,000 imágenes de entrenamiento + ~7,000 test

**Por qué usar este:**
- Dataset más completo y variado
- Ya unifica múltiples fuentes
- Organizado por letras (no por números)
- Necesario para entrenar modelos de clasificación de letras

---

### ⚠️ `Dataset.ipynb` (OBSOLETO - no usar)
**Propósito:** Versión antigua, solo procesa Sign Language MNIST

**Qué hace:**
- Solo convierte CSV de Sign Language MNIST → PNG
- Organiza por número de label (0, 1, 2, etc.)
- NO combina con ASL Alphabet
- Dataset más pequeño y limitado

**Por qué NO usar:**
- Versión antigua y limitada
- Datasetv2.ipynb es superior en todos los aspectos

---

## 🖐️ DETECCIÓN DE MANOS (`01_notebooks/hand_detection/`)

### ✅ **USAR ESTE:** `YOLOv8n_Only_Training_Final.ipynb`
**Propósito:** Entrenar modelo YOLO para detectar manos (bounding boxes) con dataset mixto

**Qué hace:**
1. **Descarga y combina dos datasets de manos:**
   - EgoHands (perspectiva egocéntrica - primera persona)
   - Kaggle Hand Dataset (Oxford Hand) - perspectivas variadas
   - Total: ~5,223 imágenes anotadas

2. **Procesa anotaciones:**
   - EgoHands: convierte polígonos .mat → YOLO format
   - Kaggle: convierte XML Pascal VOC → YOLO format
   - Unifica todo en formato YOLO (cx, cy, w, h normalizados)

3. **Entrena 2 experimentos de YOLOv8 Nano:**
   - Experimento 1: Learning rate base (1e-3)
   - Experimento 2: Learning rate bajo (1e-4)
   - 100 épocas, batch 16, imagen 416x416

4. **Evalúa exhaustivamente:**
   - mAP50, mAP50-95
   - Mean IoU, MAE, MSE, R² de coordenadas
   - Parámetros del modelo, FLOPs
   - Latencia GPU/CPU
   - Guarda tabla comparativa CSV

5. **Incluye visualización de inferencia**

**Por qué usar este:**
- Dataset MÁS COMPLETO (perspectivas variadas)
- Resuelve el problema que mencionaste de "solo detectaba egocéntrico"
- Experimentos bien configurados con métricas completas
- Código limpio y comentado

**Resultados típicos:**
- mAP50: ~98%
- Mean IoU: ~87%
- GPU latency: ~3ms por imagen

---

### `YOLOv8n_Only_Training.ipynb` (Versión anterior)
**Propósito:** Entrenar YOLO solo con EgoHands (sin Kaggle dataset)

**Limitaciones:**
- Solo usa EgoHands (~4,787 imágenes)
- Solo perspectiva egocéntrica
- Es la versión que comentaste que "solo detecta bien en fotos ego"

**Cuándo usar:**
- Si solo necesitas detección de primera persona
- Para comparar rendimiento EgoHands vs mixto

---

### `01_yolo_hands_mixed_fixed.ipynb` (Alternativa válida)
**Propósito:** Similar al Final, entrena YOLO con dataset mixto

**Diferencias con el Final:**
- Código ligeramente diferente en la estructura
- También combina EgoHands + Kaggle
- Puede tener ligeras diferencias en hiperparámetros

**Usar si:**
- Quieres comparar dos implementaciones del mismo concepto
- El Final te da problemas y necesitas alternativa

---

### `02_other_detectors_hands.ipynb`
**Propósito:** Entrenar detectores alternativos a YOLO para comparación

**Qué hace:**
- Usa el MISMO dataset creado por YOLOv8_Final
- Entrena con TorchVision:
  - **SSDLite320-MobileNetV3**
  - **Faster R-CNN-MobileNetV3-FPN**
- 100 épocas, evaluación con mismas métricas
- Tabla comparativa vs YOLO

**Cuándo usar:**
- Para tu TFG: comparar arquitecturas diferentes
- Demostrar que probaste múltiples enfoques
- Análisis de trade-offs (velocidad vs precisión)

---

### `BBHD.ipynb`
**Propósito:** Experimento con KerasCV

**Qué hace:**
- Entrena con KerasCV (Keras 3+):
  - YOLOv8 Nano
  - YOLOv8 Small
  - RetinaNet (ResNet50)
- Usa dataset de Kaggle
- Visualización comparativa

**Limitaciones:**
- KerasCV puede ser más lento que PyTorch
- Menos flexible que Ultralytics YOLO

**Cuándo usar:**
- Si prefieres ecosistema TensorFlow/Keras
- Para comparar frameworks en tu TFG

---

### `BoundingBoxHandDetectionModelV2.ipynb`
**Propósito:** Comparar modelos custom de Keras con YOLO

**Qué hace:**
- Entrena 3 modelos:
  1. **CNN Baseline** (desde cero)
  2. **MobileNetV2** (transfer learning)
  3. **YOLOv8 Nano** (referencia)
- 60 épocas, imagen 224x224
- Regresión directa de bounding boxes (4 coordenadas)
- Métrica IoU custom en Keras

**Resultados típicos:**
- Baseline: Mean IoU ~3% (MALO)
- MobileNet: Mean IoU ~5% (MALO)
- YOLO: Mean IoU ~40% (MUCHO MEJOR)

**Conclusión del experimento:**
- Demuestra que YOLO es superior
- Regresión directa de coordenadas es muy difícil
- Útil para mostrar en TFG que exploraste alternativas

**Cuándo usar:**
- Para demostrar en TFG que intentaste múltiples enfoques
- Comparar arquitectura específica de detección vs regresión simple

---

### `BoundingBoxRegressionModel.ipynb` (MUY GRANDE)
**Propósito:** Experimentos exhaustivos de comparación

**Nota:** Es un notebook muy grande (>30k tokens) con múltiples experimentos
- Probablemente contiene iteraciones y pruebas
- Puede ser versión antigua o experimental

**Recomendación:** Revisar solo si necesitas ver historial de experimentos

---

## 🎯 RESUMEN EJECUTIVO - QUÉ USAR

### Para PREPARAR DATASETS de letras ASL:
```
✅ Usar: Datasetv2.ipynb
```

### Para ENTRENAR DETECTOR DE MANOS (bounding boxes):
```
✅ Primero usar: YOLOv8n_Only_Training_Final.ipynb
   (Dataset mixto - mejor generalización)

🔄 Para comparar en TFG:
   - 02_other_detectors_hands.ipynb (SSDLite, Faster R-CNN)
   - BoundingBoxHandDetectionModelV2.ipynb (mostrar que YOLO es mejor)
```

---

## 📋 FLUJO DE TRABAJO RECOMENDADO

### 1. Preparación de datos
```bash
# Para reconocimiento de letras:
Ejecutar: Datasetv2.ipynb
Output: Dataset/SignLanguageMNIST_ASLAlphabet/
```

### 2. Entrenamiento de detector de manos
```bash
# Principal (mejor rendimiento):
Ejecutar: YOLOv8n_Only_Training_Final.ipynb
Output:
  - hand_bbox_baseline/runs_yolo/yolov8n_mix_base_lr/
  - hand_bbox_baseline/runs_yolo/yolov8n_mix_low_lr/
  - hands_model_comparison.csv

# Para comparaciones (TFG):
Ejecutar: 02_other_detectors_hands.ipynb
Output: hands_model_comparison_torchvision.csv
```

---

## 🚀 DATASETS - SE DESCARGAN AUTOMÁTICAMENTE?

### Sign Language MNIST + ASL Alphabet
**En Datasetv2.ipynb:** ❌ NO automático
- Necesitas ejecutar manualmente las celdas de kagglehub
- Ver DATASETS_REFERENCES.txt para instrucciones

### EgoHands
**En notebooks YOLO:** ✅ SÍ automático
- Se descarga automáticamente desde URL directa
- urllib.request.urlretrieve()

### Kaggle Hand Dataset
**En notebooks YOLO:** ✅ SÍ automático (si tienes kagglehub)
- Necesitas kaggle.json configurado
- kagglehub.dataset_download() lo hace automáticamente

---

## ❓ ¿QUÉ ME FALTABA POR HACER?

Según tu memoria, probablemente:

1. **Detector de manos entrenado:** ✅ HECHO
   - YOLOv8n_Only_Training_Final.ipynb lo hace

2. **Clasificador de letras:** ❓ NO VEO NOTEBOOKS DE ESTO
   - Faltaría crear notebook que use:
     - Detector entrenado para extraer manos
     - Clasificador CNN para reconocer letras (A-Y)
     - Pipeline completo: imagen → detectar mano → clasificar letra

3. **Comparación de modelos:** ✅ PARCIALMENTE HECHO
   - Tienes comparaciones de detectores
   - Faltaría comparación de clasificadores

---

## 📝 PRÓXIMOS PASOS SUGERIDOS

1. **Ejecutar notebooks definitivos:**
   ```
   1. Datasetv2.ipynb → generar dataset de letras
   2. YOLOv8n_Only_Training_Final.ipynb → entrenar detector
   3. 02_other_detectors_hands.ipynb → comparativas
   ```

2. **Crear notebook nuevo de clasificación:**
   - Entrenar CNN/MobileNet/EfficientNet para clasificar letras (A-Y)
   - Usar dataset de Datasetv2.ipynb
   - Comparar múltiples arquitecturas

3. **Crear notebook de pipeline completo:**
   - Cargar detector de manos
   - Cargar clasificador de letras
   - Demo: foto → detección → clasificación → resultado

---

¿Quieres que renombre los notebooks y les agregue explicaciones en sus celdas markdown para que quede todo más claro?
