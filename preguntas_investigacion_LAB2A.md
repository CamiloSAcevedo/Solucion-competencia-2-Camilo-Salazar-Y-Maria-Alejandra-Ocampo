# Preguntas de Investigación — Laboratorio #2A

---

## 1. Normalización ImageNet

Los valores `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]` fueron calculados empíricamente sobre el conjunto completo de entrenamiento de ImageNet, que contiene más de 1.2 millones de imágenes. Representan la media y desviación estándar de los píxeles en cada canal RGB a lo largo de todo ese dataset. Usarlos es fundamental porque los pesos preentrenados fueron optimizados esperando exactamente esa distribución de entrada. Si se usan valores distintos, las activaciones internas quedan fuera del rango para el cual los pesos fueron ajustados, degradando las representaciones aprendidas y reduciendo significativamente el rendimiento del modelo transferido.

---

## 2. CrossEntropyLoss vs NLLLoss + LogSoftmax

`nn.CrossEntropyLoss` y la combinación `nn.LogSoftmax + nn.NLLLoss` son matemáticamente equivalentes: ambas calculan el negativo del logaritmo de la probabilidad asignada a la clase correcta. La diferencia es que `CrossEntropyLoss` recibe logits crudos y aplica el log-softmax internamente, mientras que `NLLLoss` espera que la red ya haya aplicado `LogSoftmax`. Por esta razón, añadir una capa `LogSoftmax` al final de la red y además usar `CrossEntropyLoss` es un error: el log-softmax se aplica dos veces, produciendo valores de pérdida incorrectos y gradientes erróneos que impiden la convergencia correcta del modelo.

---

## 3. Data Augmentation

Para mejorar la generalización en Cats vs. Dogs, se recomienda añadir `RandomHorizontalFlip` (los animales son simétricos horizontalmente), `RandomRotation(15)` (pueden aparecer en distintas orientaciones) y `ColorJitter` (simula variaciones de iluminación y cámara). Estas transformaciones se aplican solo al pipeline de entrenamiento, nunca al de validación. Al comparar en TensorBoard, se espera que con augmentation la brecha entre la pérdida de entrenamiento y validación sea menor, indicando menos overfitting, aunque la pérdida de entrenamiento sea algo más alta porque el modelo ve imágenes más difíciles en cada época.

```python
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 4. Learning Rate Scheduling

`StepLR` reduce el learning rate por un factor fijo (`gamma`) cada cierto número de épocas (`step_size`), produciendo caídas discretas y predecibles. `CosineAnnealingLR` en cambio lo reduce siguiendo una curva coseno de forma suave y continua hasta un mínimo `eta_min`, lo que generalmente produce mejores resultados porque evita los saltos bruscos. En el loop de fine-tuning se agrega llamando a `scheduler.step()` al final de cada época. Al comparar en TensorBoard, CosineAnnealingLR tiende a mostrar una curva de validación que sigue bajando suavemente hasta el final, mientras que con lr fijo la pérdida suele estancarse antes.

```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

for epoch in range(1, 6):
    train_epoch(vgg, train_loader, optimizer, nn.CrossEntropyLoss())
    scheduler.step()
```

---

## 5. Límites del Transfer Learning

Transfer learning desde ImageNet puede no ser beneficioso cuando el dominio objetivo es radicalmente diferente al de las fotografías naturales. Un ejemplo claro son las **imágenes médicas** (rayos X, histopatología), donde los patrones relevantes son texturas microscópicas y densidades de tejido que no guardan ninguna relación con los filtros aprendidos en ImageNet. Otro escenario es cuando se dispone de un **dataset objetivo muy grande y especializado** (por ejemplo, millones de imágenes satelitales etiquetadas): en ese caso entrenar desde cero puede superar al transfer learning porque el modelo puede especializarse completamente sin cargar el sesgo de ImageNet como prior incorrecto.

---

## 6. Reto Opcional — EfficientNet-B0

EfficientNet-B0 escala simultáneamente la profundidad, anchura y resolución de la red mediante un coeficiente compuesto, logrando mayor accuracy con muchos menos parámetros (~5.3M) comparado con VGG-16 (~138M) y ResNet-18 (~11M). Al adaptarlo a Cats vs. Dogs se congela el backbone y se reemplaza `classifier[1]` por una `Linear(1280, 2)`. En los experimentos, EfficientNet-B0 logra accuracy comparable o superior a los otros dos modelos entrenando más rápido que VGG y con menor huella de memoria, confirmando que su diseño mediante Neural Architecture Search lo hace más eficiente que las arquitecturas diseñadas manualmente.
