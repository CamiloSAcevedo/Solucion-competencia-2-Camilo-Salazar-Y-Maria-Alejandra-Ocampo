# Laboratorio #2 — Módulo II: Transfer Learning y Segmentación Semántica
### Redes Neuronales y Aprendizaje Profundo — Universidad EAFIT

---

## Integrantes

- **Camilo Salazar**
- **María Alejandra Ocampo**

---

## Descripción general

Este laboratorio explora dos de las técnicas más importantes en visión por computador con deep learning: **transfer learning** para clasificación de imágenes y **segmentación semántica** con arquitecturas encoder-decoder. El trabajo está dividido en dos partes independientes, cada una con su propio notebook entregable.

---

## Parte A — Transfer Learning con PyTorch

**Notebook:** `grupoX_LAB2A.ipynb`  
**Dataset:** Kaggle Cats vs. Dogs (~25,000 imágenes)

Se implementaron y compararon tres estrategias de transfer learning sobre el problema binario de clasificación gato vs. perro:

- **M1 — VGG-16 como extractor fijo:** se congelaron todas las capas convolucionales y se entrenó únicamente un clasificador lineal sobre los features precomputados de `vgg.features`. Esto permite entrenar en cuestión de segundos sin GPU.

- **M2 — VGG-16 con fine-tuning completo:** se reemplazó el clasificador original por una capa `Linear(25088, 2)`, se entrenó primero con el backbone congelado para estabilizar los pesos, y luego se descongeló todo el modelo para un ajuste fino con learning rate reducido (`1e-4`).

- **M3 — ResNet-18 adaptado:** se congelaron todos los parámetros del backbone y se reemplazó la capa `fc` por una `Linear(512, 2)`. Se comparó velocidad de entrenamiento y accuracy contra VGG-16, destacando el impacto de los residual blocks y BatchNorm.

Todos los experimentos fueron registrados con **TensorBoard** para comparar curvas de loss y accuracy entre modelos.

---

## Parte B — Segmentación Semántica con U-Net

**Notebook:** `grupoX_LAB2B.ipynb`  
**Dataset:** Oxford-IIIT Pet (segmentación pixel-wise, problema binario foreground/background)

Se construyó una pipeline completa de segmentación semántica implementando tres modelos progresivos:

- **M1 — U-Net desde cero:** se implementaron los bloques `DoubleConv` y `UpBlock`, y se completó el método `forward` con el mecanismo de skip connections. Sirve como línea base sin conocimiento previo.

- **M2 — ResNetUNet con encoder congelado:** se integró ResNet-18 preentrenado en ImageNet como backbone del encoder, extrayendo skip connections de cada uno de sus cuatro bloques residuales. Solo se entrenó el decoder en esta fase.

- **M3 — ResNetUNet con fine-tuning completo:** se descongeló el encoder y se realizó un ajuste fino end-to-end con learning rate reducido, obteniendo la mejor precisión en bordes y regiones de transición.

Los modelos se evaluaron con métricas de segmentación estándar (**IoU** y **Dice score**), se visualizaron predicciones sobre el test set y se registraron todas las curvas en **TensorBoard**.

---

## Estructura del repositorio

```
├── grupoX_LAB2A.ipynb                  # Parte A — Transfer Learning
├── grupoX_LAB2B.ipynb                  # Parte B — Segmentación U-Net
├── preguntas_investigacion_LAB2A.md    # Respuestas preguntas Parte A
├── preguntas_investigacion_LAB2B.md    # Respuestas preguntas Parte B
└── README.md
```

---

## Tecnologías utilizadas

- Python 3.10
- PyTorch + torchvision
- torchinfo
- TensorBoard
- Matplotlib / NumPy / PIL
