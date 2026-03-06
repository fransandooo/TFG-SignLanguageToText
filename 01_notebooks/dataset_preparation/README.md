# Dataset Preparation Notebooks

Este directorio contiene notebooks para preparar los datasets de reconocimiento de letras ASL.

---

## 📓 Notebook Principal (USAR ESTE)

### 01_Prepare_ASL_Dataset_Combined.ipynb
**⭐ ÚNICO NOTEBOOK A EJECUTAR**

**Propósito:** Preparar dataset completo combinando múltiples fuentes

**Qué hace:**
1. **Descarga automáticamente 2 datasets:**
   - Sign Language MNIST (CSV 28x28, 24 letras)
   - ASL Alphabet (imágenes RGB de letras)

2. **Procesa y convierte:**
   - MNIST: CSV → PNG escala de grises
   - ASL Alphabet: JPG RGB → PNG escala de grises

3. **Organiza por letras:**
   - Carpetas: A, B, C, D, ..., Y (sin J, Z que son dinámicas)
   - Estructura clara: `Dataset/SignLanguageMNIST_ASLAlphabet/train/` y `test/`

4. **Resultado final:**
   - ~79,000 imágenes de entrenamiento
   - ~7,000 imágenes de test
   - Dataset unificado listo para entrenar clasificadores

**Cuándo ejecutar:**
- Primera vez configurando el proyecto
- Cuando necesites regenerar el dataset
- Si quieres entrenar modelos de clasificación de letras

**Output:**
```
Dataset/SignLanguageMNIST_ASLAlphabet/
├── train/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ... (24 letras)
└── test/
    ├── A/
    ├── B/
    └── ...
```

**Requisitos:**
```bash
pip install kagglehub numpy matplotlib pillow
```

**Configuración necesaria:**
- Archivo `utils/kaggle.json` con tus credenciales de Kaggle
- Ver `DATASETS_REFERENCES.txt` en la raíz para obtener API key

---

## 📁 Carpeta `archived/`

### Dataset_OLD_OnlyMNIST.ipynb
- **Por qué está archivado:** Solo procesa Sign Language MNIST
- **Limitaciones:**
  - No combina con ASL Alphabet
  - Organiza por números (0, 1, 2...) en vez de letras
  - Dataset más pequeño
- **Reemplazado por:** `01_Prepare_ASL_Dataset_Combined.ipynb`

**No usar** a menos que solo necesites el dataset MNIST básico.

---

## 🚀 Flujo de Trabajo

1. **Configurar credenciales de Kaggle:**
   ```bash
   # Copiar tu kaggle.json a:
   # utils/kaggle.json
   ```

2. **Ejecutar notebook:**
   - Abrir: `01_Prepare_ASL_Dataset_Combined.ipynb`
   - Ejecutar todas las celdas
   - Esperar descarga y procesamiento (~5-10 minutos)

3. **Verificar output:**
   ```bash
   ls Dataset/SignLanguageMNIST_ASLAlphabet/train/
   # Deberías ver carpetas A, B, C, ... Y
   ```

4. **Siguiente paso:**
   - Usar este dataset para entrenar clasificador de letras
   - Integrar con detector de manos para pipeline completo

---

## 📊 Datasets Incluidos

| Dataset | Fuente | Imágenes | Resolución | Formato |
|---------|--------|----------|------------|---------|
| Sign Language MNIST | Kaggle | ~34k | 28x28 | CSV → PNG |
| ASL Alphabet | Kaggle | ~87k | Variable | JPG → PNG |
| **TOTAL COMBINADO** | - | **~121k** | 28x28+ | PNG grayscale |

---

## ⚙️ Configuración de Kaggle

Para descargar automáticamente los datasets necesitas:

1. **Crear cuenta en Kaggle:** https://www.kaggle.com

2. **Generar API token:**
   - Ve a: Account → API → Create New API Token
   - Descarga `kaggle.json`

3. **Copiar a tu proyecto:**
   ```bash
   cp ~/Downloads/kaggle.json utils/kaggle.json
   ```

Ver `DATASETS_REFERENCES.txt` en la raíz del proyecto para más detalles.

---

## 🔍 Referencias

- **Sign Language MNIST:** `datamunge/sign-language-mnist`
- **ASL Alphabet:** `grassknoted/asl-alphabet`
- Más info en: `DATASETS_REFERENCES.txt`
