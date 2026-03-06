# Archived Hand Detection Notebooks

Esta carpeta contiene notebooks experimentales, versiones antiguas y enfoques alternativos.

**No necesitas ejecutar estos** para el flujo normal de trabajo.

---

## 📂 Contenido

### Versiones Antiguas de YOLO

#### YOLOv8n_OnlyEgoHands_OLD.ipynb
- **Por qué está archivado:** Solo usa dataset EgoHands (perspectiva egocéntrica)
- **Problema:** Detecta bien solo en fotos de primera persona
- **Reemplazado por:** `01_YOLO_HandDetector_MixedDataset_MAIN.ipynb`

#### YOLO_MixedDataset_Alternative.ipynb
- **Por qué está archivado:** Versión alternativa con código ligeramente diferente
- **Reemplazado por:** `01_YOLO_HandDetector_MixedDataset_MAIN.ipynb`
- **Cuándo usar:** Si la versión principal te da problemas

---

### Experimentos con CNNs Custom

#### CNN_vs_YOLO_Comparison.ipynb
- **Propósito:** Comparar CNN custom vs YOLO
- **Modelos probados:**
  - Baseline CNN (desde cero)
  - MobileNetV2 (transfer learning)
  - YOLOv8 Nano (referencia)
- **Resultado:** YOLO es muy superior (IoU ~40% vs ~3-5% CNNs)
- **Cuándo revisar:** Para tu TFG, demostrar que exploraste múltiples enfoques

#### CNN_Experiments_PrePooling.ipynb
- **Propósito:** Experimentos con arquitecturas CNN pre-pooling
- **Estado:** Experimental

#### CNN_Regression_FullExperiments.ipynb
- **Propósito:** Experimentos exhaustivos de regresión de bounding boxes
- **Nota:** Archivo muy grande (~30k tokens)
- **Estado:** Historial de iteraciones y pruebas

---

### Otros Experimentos

#### KerasCV_Experimental.ipynb
- **Propósito:** Entrenar con KerasCV (Keras 3+)
- **Modelos:** YOLOv8 Nano/Small, RetinaNet
- **Por qué está archivado:** KerasCV es más lento y menos flexible que Ultralytics
- **Cuándo usar:** Si prefieres ecosistema TensorFlow/Keras

#### boundingboxregressionmodel.py
- **Propósito:** Script de utilidad para experimentos CNN
- **Estado:** Código auxiliar de notebooks archivados

---

## 💡 ¿Cuándo Revisar Estos Notebooks?

**Para el TFG:**
- `CNN_vs_YOLO_Comparison.ipynb` → Demostrar que YOLO es superior
- `YOLOv8n_OnlyEgoHands_OLD.ipynb` → Explicar por qué dataset mixto es mejor

**Para debugging:**
- `YOLO_MixedDataset_Alternative.ipynb` → Si versión principal falla

**Por curiosidad:**
- Todos los demás → Ver historial de experimentos y decisiones

---

## ⚠️ Nota Importante

Estos notebooks pueden:
- Tener configuraciones antiguas
- Requerir dependencias específicas
- No estar completamente documentados
- Tener resultados subóptimos

**Usa bajo tu propio riesgo. La carpeta principal tiene las versiones definitivas y probadas.**
