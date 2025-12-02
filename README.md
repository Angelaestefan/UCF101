# UCF101 Skeleton-Based Action Recognition

## 📋 Descripción del Proyecto

Este proyecto implementa una **arquitectura de Deep Learning (CNN+RNN)** para reconocimiento de acciones usando datos de esqueletos 2D derivados del dataset UCF101.

### Características Principales:
- **Representación**: Esqueletos 2D (17 articulaciones × 2 coordenadas)
- **Clases**: 5 acciones seleccionadas (JumpRope, JumpingJack, PushUps, Lunges, BodyWeightSquats)
- **Arquitectura**: CNN espacial + LSTM temporal
- **Pipeline**: Group-aware validation, data augmentation, 3-fold cross-validation
- **Framework**: PyTorch

---

## 🏗️ Arquitectura del Modelo

### CNN+RNN Baseline
- **CNN Espacial**: 2 capas Conv2D (64 → 128 canales) que extraen features por frame
- **LSTM Temporal**: Procesa secuencia de features (hidden_size=128)
- **Classifier**: Capa FC para 5 clases

### CNN+RNN Mejorado
- **+ BatchNorm**: Normalización después de cada Conv2D
- **+ Dropout (0.3)**: Regularización en LSTM y FC
- **+ Bidirectional LSTM**: Contexto temporal pasado y futuro
- **+ Data Augmentation**: Random temporal crop en training

---

## 📊 Metodología

### 1. Preprocesamiento
- Filtrado a 5 clases de interés
- Normalización espacial (center en joint 0, scale)
- Temporal crop/padding a 32 frames

### 2. Validación Group-Aware
Para evitar **data leakage**, dividimos el train set por **grupos de videos**:
- Videos del mismo grupo (e.g., `v_JumpRope_g01_c01`, `v_JumpRope_g01_c02`) NO se separan
- Train/Val split: 80/20 respetando grupos

**¿Por qué?** Los splits oficiales train1/train2/train3 NO son mutuamente excluyentes (el mismo video puede aparecer en múltiples splits). Por lo tanto, usar train2 como validación causaría leakage.

### 3. Evaluación 3-Fold
- Entrenamos en train1/val → test en test1
- Entrenamos en train2/val → test en test2
- Entrenamos en train3/val → test en test3
- Reportamos **mean ± std** de las 3 accuracies

### 4. Mejoras Documentadas
| Mejora | Justificación | Efecto Obtenido |
|--------|--------------|------------------|
| BatchNorm | Estabiliza gradientes, acelera convergencia | ✅ Mejora convergencia visible en curvas |
| Dropout (0.3) | Previene overfitting, mejora generalización | ✅ Reduce gap train/val |
| Bidirectional LSTM | Contexto temporal completo (pasado+futuro) | ✅ +16.28% total sobre baseline |
| Data Augmentation | Variabilidad temporal, reduce overfitting | ✅ Estabiliza validación |
| Group-Aware Split | Evita data leakage entre train/val | ✅ Validación realista |

**Resultado Combinado**: El modelo mejorado logra **58.14% test accuracy** vs **41.86%** del baseline (+39% mejora relativa)

---

## 🚀 Ejecución

### Requisitos
```bash
# Crear entorno virtual
python3 -m venv ucf101_env
source ucf101_env/bin/activate

# Instalar dependencias
pip install torch torchvision numpy matplotlib scikit-learn seaborn jupyter
```

### Opción 1: Jupyter Notebook (Recomendado)
```bash
# Activar entorno
source ucf101_env/bin/activate

# Lanzar Jupyter
jupyter notebook ucf101_cnn_rnn.ipynb
```


### Opción 2: Script Standalone
```bash
# Entrenar modelos en 3 splits
./ucf101_env/bin/python3 train_cnn_rnn.py
```

---

## 📁 Estructura de Archivos

```
UCF101/
├── README.md                          # Este archivo
├── ucf101_cnn_rnn.ipynb              # Notebook principal
├── train_cnn_rnn.py                  # Script de entrenamiento standalone
├── ucf101_2d.pkl                     # Dataset original
├── ucf101_5classes_skeleton.pkl      # Dataset filtrado (5 clases)
├── cnn_rnn_baseline.pth              # Modelo baseline guardado
├── cnn_rnn_improved.pth              # Modelo mejorado guardado
├── results_cnn_rnn.pkl               # Resultados de experimentos
├── training_curves_cnn_rnn.png       # Curvas de entrenamiento
└── confusion_matrix_improved.png     # Matriz de confusión
```

---

## 📈 Resultados Obtenidos

### Baseline (sin regularización):
- **Val Accuracy**: ~60.5%
- **Test Accuracy**: **41.86%**
- **Problema**: Overfitting visible (train accuracy llega a 57% pero no generaliza bien)

### Improved (con todas las mejoras):
- **Val Accuracy**: ~72-82%
- **Test Accuracy**: **58.14%**
- **Mejora**: **+16.28%** sobre baseline
- **Generalización**: Gap train/val reducido significativamente

### Resultados por Clase (Modelo Mejorado):

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| BodyWeightSquats | 0.83 | 0.17 | 0.28 | 30 |
| JumpRope | 0.58 | 0.29 | 0.39 | 38 |
| **JumpingJack** | **0.66** | **0.78** | **0.72** | 37 |
| Lunges | 0.41 | 0.76 | 0.53 | 37 |
| **PushUps** | **0.77** | **0.90** | **0.83** | 30 |
| | | | | |
| **Accuracy** | | | **0.58** | 172 |
| Macro avg | 0.65 | 0.58 | 0.55 | 172 |
| Weighted avg | 0.64 | 0.58 | 0.55 | 172 |

**Observaciones**:
- ✅ **Mejor clasificadas**: PushUps (90% recall) y JumpingJack (78% recall)
- ⚠️ **Desafíos**: BodyWeightSquats (17% recall) y JumpRope (29% recall) se confunden frecuentemente con Lunges
- 💡 **Razón**: Movimientos de piernas similares entre estas clases

### 3-Fold Evaluation:
```
Split 1: 58.14%
Split 2: 60.47%
Split 3: 56.98%
Mean: 58.53% ± 1.43%
```
---

## 📚 Referencias

- **Dataset**: UCF101 - https://www.crcv.ucf.edu/data/UCF101.php
- **Esqueletos 2D**: Derivados de pose estimation (17 joints COCO format)
- **Framework**: PyTorch - https://pytorch.org/

---

## 👨‍💻 Autor

Angela Aguilar

**Fecha**: Diciembre 2025

---

