# Hand Detection Notebooks

Este directorio contiene los notebooks para entrenar y evaluar detectores de manos.

## 📓 Notebooks Principales (USAR ESTOS)

### 01_YOLO_HandDetector_MixedDataset_MAIN.ipynb
**⭐ NOTEBOOK PRINCIPAL - USAR ESTE PRIMERO**

- **Propósito:** Entrenar YOLOv8 Nano para detectar manos con dataset mixto
- **Dataset:** EgoHands + Kaggle Hand Dataset (~5,223 imágenes)
- **Por qué es el mejor:** Combina perspectivas egocéntricas y variadas para mejor generalización
- **Experimentos:** 2 configuraciones de learning rate
- **Métricas:** mAP50 (~98%), Mean IoU (~87%), latencia GPU (~3ms)
- **Output:**
  - Modelos entrenados en `hand_bbox_baseline/runs_yolo/`
  - Tabla comparativa: `hands_model_comparison.csv`

**Cuándo ejecutar:**
- Primera vez configurando el proyecto
- Necesitas entrenar detector de manos desde cero
- Quieres el modelo con mejor rendimiento

---

### 02_SSDLite_FasterRCNN_Comparison.ipynb
**📊 PARA COMPARACIÓN EN TFG**

- **Propósito:** Comparar arquitecturas alternativas a YOLO
- **Modelos:**
  - SSDLite320-MobileNetV3
  - Faster R-CNN-MobileNetV3-FPN
- **Por qué ejecutar:** Para demostrar que exploraste múltiples enfoques
- **Output:** `hands_model_comparison_torchvision.csv`

**Cuándo ejecutar:**
- Después de ejecutar el notebook principal
- Para análisis de trade-offs velocidad/precisión en tu TFG
- Cuando necesites comparar diferentes arquitecturas

---

## 📁 Carpeta `archived/`

Contiene notebooks experimentales, versiones antiguas o alternativas. Ver `archived/README.md` para detalles.

**No necesitas ejecutar estos** a menos que quieras:
- Ver historial de experimentos
- Comparar con versiones antiguas
- Explorar enfoques experimentales (KerasCV, CNNs custom, etc.)

---

## 🚀 Flujo de Trabajo Recomendado

1. **Ejecutar:** `01_YOLO_HandDetector_MixedDataset_MAIN.ipynb`
   - Entrena el detector principal
   - Genera modelos y métricas base

2. **Ejecutar (opcional):** `02_SSDLite_FasterRCNN_Comparison.ipynb`
   - Solo si necesitas comparaciones para el TFG

3. **Resultado:** Tendrás múltiples modelos entrenados para elegir el mejor

---

## 📋 Requisitos

```bash
pip install ultralytics torch torchvision kagglehub scipy opencv-python matplotlib pandas
```

## ⚙️ Configuración Datasets

Los datasets se descargan automáticamente:
- **EgoHands:** Descarga directa desde URL
- **Kaggle Hand Dataset:** Necesitas `utils/kaggle.json` configurado

Ver `DATASETS_REFERENCES.txt` en la raíz para más detalles.
