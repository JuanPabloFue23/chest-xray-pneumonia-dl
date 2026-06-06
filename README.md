# Diagnóstico Asistido por IA: Detección de Neumonía en Radiografías de Tórax

Este repositorio contiene el proyecto final desarrollado para el **Diplomado en Inteligencia Artificial con Deep Learning de la Universidad de la Sabana**. El objetivo principal es construir y comparar modelos de aprendizaje profundo capaces de clasificar de forma automatizada imágenes de rayos X de tórax en dos categorías: **NORMAL** y **PNEUMONIA**.

## 📊 1. Análisis Exploratorio de Datos (EDA)

El conjunto de datos original (*Chest X-Ray Images*) presentaba una distribución altamente desbalanceada en el conjunto de entrenamiento, un factor crítico que fue diagnosticado antes del modelado:

* **PNEUMONIA:** 73.0% (5,982 imágenes)
* **NORMAL:** 27.0% (2,216 imágenes)

### Decisiones Técnicas Basadas en el EDA:
1. **Métrica de Optimización:** Debido al marcado desbalanceo, el *Accuracy* (Exactitud) se descartó como métrica principal de éxito. Se priorizó el **Recall (Sensibilidad)**, dado que en el entorno clínico un Falso Negativo (enviar a un paciente enfermo a su hogar libre de sospecha) representa el escenario de mayor riesgo para la vida del paciente.
2. **Estandarización de Dimensiones:** El análisis reveló que las imágenes originales poseían resoluciones promedio de $1289 \times 927$ píxeles. Para viabilizar el entrenamiento en memoria sin perder los patrones macro de infiltración pulmonar, se estableció un redimensionamiento estándar de $224 \times 224$ píxeles.
3. **Corrección de la Validación:** Se detectó que la partición de validación original de Kaggle contaba con apenas 16 imágenes (insuficiente para una validación estadística fiable). Se consolidaron los datos y se aplicó un split estratificado moderno: **70% Entrenamiento, 15% Validación y 15% Prueba**.

---

## 🏗️ 2. Arquitecturas Evaluadas

### Modelo 1: CNN Baseline (Desde Cero)
Diseño secuencial iterativo optimizado específicamente para la detección de características opacas en los campos pulmonares:
* **Extracción de Características:** 3 bloques convolucionales secuenciales (`Conv2D` con activaciones ReLU + `MaxPooling2D`) con incremento progresivo de filtros (32, 64, 128) para capturar desde bordes hasta texturas complejas de consolidación neumónica.
* **Regularización:** Inclusión de una capa de aplanado (`Flatten`) acoplada a una capa densa intermedia de 128 neuronas con un `Dropout(0.5)` estricto para mitigar el sobreajuste y apagar el sesgo hacia la clase mayoritaria.
* **Clasificación:** Capa de salida con una única neurona y activación `Sigmoid` utilizando el optimizador `Adam` con una tasa de aprendizaje controlada de $1\times 10^{-4}$.

### Modelo 2: Transfer Learning (ResNet50V2)
Implementación de una arquitectura profunda de vanguardia preentrenada, dividida en dos etapas metodológicas:
* **Etapa 1 (Feature Extraction):** Congelación total de los pesos del modelo base *ResNet50V2* (aprovechando el conocimiento previo de ImageNet). Se acopló una cabeza clasificadora con `GlobalAveragePooling2D`, una capa densa de 256 neuronas con activación ReLU, `Dropout(0.4)` y la neurona de salida sigmoide, optimizada con una tasa de aprendizaje estándar de $1\times 10^{-3}$.
* **Etapa 2 (Fine-Tuning):** Descongelación controlada de las capas para permitir el ajuste fino de los pesos. Se utilizó una tasa de aprendizaje milimétrica de $1\times 10^{-5}$ junto con el callback `ReduceLROnPlateau` para adaptar los filtros abstractos a la morfología de las radiografías médicas sin destruir las características generales previamente aprendidas.

---

## 📈 3. Resultados y Comparación en el Conjunto de Prueba (Test)

El rendimiento definitivo de ambos modelos fue evaluado de forma estricta utilizando el conjunto de datos de prueba (`test_ds`), el cual se mantuvo completamente aislado durante todo el proceso de diseño, entrenamiento y ajuste de hiperparámetros.

| Métrica | CNN Baseline (Desde Cero) | ResNet50V2 (Transfer Learning + Fine-Tuning) |
| :--- | :---: | :---: |
| **Test Accuracy** | **98.17%** | **99.51%** |
| **Test Recall (Sensibilidad)** | **98.50%** | **99.41%** |
| **Test Loss** | **0.0542** | **0.0182** |

### 🔍 Interpretación Analítica y Diagnóstico Clínico (Defensa ante el Jurado)

1. **El Poder de las Conexiones Residuales:** Mientras que la red convolucional construida desde cero (*Baseline*) entregó un rendimiento sobresaliente del **98.17%**, la arquitectura avanzada **ResNet50V2** logró optimizar el límite de generalización alcanzando un **99.51% de Accuracy** y reduciendo la pérdida de validación a un mínimo de **0.0182**. Esto demuestra la efectividad de transferir características abstractas preentrenadas y adaptarlas mediante un proceso controlado de *Fine-Tuning*.
2. **Mitigación del Error Crítico (Falsos Negativos):** En el contexto de la salud pública, la métrica dorada de este proyecto es el **Recall**. El modelo avanzado alcanzó un **99.41% de Recall en validación**, lo que significa que el sistema reduce el riesgo de omitir pacientes enfermos a menos del **0.6%**. Esto valida técnicamente la viabilidad de la solución como una herramienta de triaje de alta confiabilidad para entornos hospitalarios con alta demanda o escasez de radiólogos especialistas.
3. **Eficiencia en el Entrenamiento:** Gracias a la implementación de estrategias de regularización como *Dropout*, tasas de aprendizaje asimétricas ($1\times 10^{-3}$ para extracción y $1\times 10^{-5}$ para ajuste fino) y el uso de callbacks dinámicos como `ReduceLROnPlateau`, ambos modelos convergieron de manera estable sin presentar síntomas de sobreajuste (*overfitting*), devolviendo curvas de pérdida en validación sumamente limpias y decrecientes.

---

## 🛠️ Requisitos y Guía de Reproducción

Debido a restricciones de almacenamiento y siguiendo las buenas prácticas de ingeniería de software para Inteligencia Artificial, las carpetas de datos originales (`data/`) y los archivos de pesos de los modelos entrenados (`models/`) han sido excluidos del control de versiones mediante el archivo `.gitignore`. 

Sigue estos pasos exactos para reconstruir el entorno local, descargar los datos y reproducir los experimentos desde cero:

### 1. Clonar el Repositorio y Configurar el Entorno
Abre tu terminal en tu máquina local y ejecuta de forma secuencial:

```bash
# Clonar el proyecto
git clone [https://github.com/JuanPabloFue23/chest-xray-pneumonia-dl.git](https://github.com/JuanPabloFue23/chest-xray-pneumonia-dl.git)
cd chest-xray-pneumonia-dl

# Crear el entorno virtual de aislamiento
python -m venv venv

# Activar el entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Actualizar pip e instalar dependencias del proyecto
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Adquisición y Descarga del Dataset
1. Descarga el conjunto de datos oficial *Chest X-Ray Images (Pneumonia)* directamente desde Kaggle.
2. Crea una carpeta llamada `data` en la raíz de este proyecto y dentro de ella una subcarpeta llamada `raw`.
3. Descomprime los archivos descargados de modo que la estructura interna coincida exactamente con el siguiente esquema:

```text
chest-xray-pneumonia-dl/
├── data/
│   ├── raw/
│   │   └── chest_xray/
│   │       ├── train/       # Imágenes de entrenamiento originales
│   │       ├── val/         # Imágenes de validación originales
│   │       └── test/        # Imágenes de prueba originales

### 3. Pipeline de Ejecución (Secuencia de Notebooks)
Abre el entorno en tu IDE preferido (como VS Code), asegúrate de seleccionar el entorno virtual `.venv` recién creado como tu Kernel de Jupyter, y ejecuta los cuadernos ubicados en la carpeta `notebooks/` en el siguiente orden estricto:

1. **`01_eda_preprocessing.ipynb`**: 
   * Ejecuta este cuaderno primero. Se encargará de leer los datos de `data/raw/`, realizar el análisis de dimensiones, procesar el desbalanceo de clases y reestructurar de forma automática la partición de datos, guardando los resultados listos en una nueva carpeta automática llamada `data/processed/`.
   
2. **`02_baseline_model.ipynb`**:
   * Entrena la arquitectura de Red Neuronal Convolucional (CNN) desde cero utilizando el flujo optimizado de datos procesados. Al finalizar la ejecución, el cuaderno creará de forma automática la carpeta `models/` localmente y guardará el archivo de pesos `baseline_cnn.h5`.

3. **`03_advanced_model.ipynb`**:
   * Corre este cuaderno al final para ejecutar las fases de *Transfer Learning* y *Fine-Tuning* con la arquitectura **ResNet50V2**. Generará el archivo de pesos definitivo `advanced_resnet.h5` en tu carpeta local de modelos y desplegará las matrices de confusión comparativas.