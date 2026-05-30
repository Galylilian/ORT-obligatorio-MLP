obligatorio/
│
├── data/                         # Dataset (NO subir a GitHub)
│   ├── raw/                      # dataset original de Roboflow
│   ├── processed/                # dataset convertido a clasificación
│   │   ├── train/
│   │   │   ├── acostado/
│   │   │   └── no_acostado/
│   │   ├── valid/
│   │   └── test/
│
├── models/                       # Modelos entrenados
│   ├── resnet18.pth
│
├── notebooks/                    # EDA y experimentación
│   └── eda.ipynb                 # falta hacerlo
│
├── scripts/                      # Scripts auxiliares
│   ├── download_dataset.py       # descarga Roboflow
│   ├── convert_dataset.py        # convierte a clasificación
│
├── src/                          # Código fuente principal
│
│   ├── api/                      # API (FastAPI)
│   │   ├── routers/
│   │   │   ├── predict.py        # endpoint CNN
│   │   │   └── predict_yolo.py   # endpoint YOLO
│   │   └── app.py                # integración FastAPI
│
│   ├── core/                     # Lógica ML
│   │   ├── model.py              # ResNet18
│   │   ├── train.py              # revisar mejorar
│   │   ├── evaluate.py
│   │   ├── yolo_model.py         # baseline YOLO
│   │   ├── gradcam.py            # explicabilidad
│   │
│   ├── preprocessing/            # Transformaciones
│   │   └── transforms.py         # falta mejorarlo
│
│   ├── data/                     # Carga de datos
│   │   └── dataset.py
│
│   ├── utils/                    # Utilidades
│   │   ├── metrics.py
│   │   └── logger.py
│
│   ├── settings/                 # Configuración
│   │   └── config.py             # aca esta el modelo a utilizar y OS
│
├── app/                          # UI (Streamlit)
│   └── streamlit_app.py          # este es el frontend
│
├── tests/                        # Tests
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore



# Hospital Bed Detector

## Problema
Clasificación binaria para detectar si un paciente está acostado.

## Modelos
- CNN ResNet18 (modelo de clasificacion)
- YOLO baseline (modelo de deteccion)

## API
- /predict
- /predict_yolo

## Run en local

pip install -r requirements.txt
python src/core/train.py
uvicorn src.api.app:app --reload



# Pasos

#offline 
1. pip install -r requirements.txt
2. python scripts/download_dataset.py
3. python scripts/convert_dataset.py
4. python -m src.core.train

download → convert → train → evaluate

#Produccion
5. docker-compose up --build 

API (FastAPI)


#TEST
6. python -m src.core.evaluate
7. python scripts/compare_models.py

# front
8. streamlit run app/streamlit_app.py

El sistema se compone de:

un pipeline offline de entrenamiento,
una API para inferencia en producción,
un script para evaluación comparativa,
y una interfaz de usuario con Streamlit para interacción con el modelo.

# en sistemas de ML en producción, el entrenamiento y la inferencia suelen separarse. El entrenamiento es costoso y se ejecuta offline, mientras que la API en producción debe ser ligera, rápida y estable.




train.py → crea modelo ✅
model.py → define modelo ✅
gradcam.py → explica ✅
API → usa modelo ✅
Streamlit → usa API ✅



Pipeline completo
  ┌───────────────────────┐
                │   ROBFLOW DATASET     │
                │ (imágenes + labels)   │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ download_dataset.py   │
                │ (descarga dataset)    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │   data/raw            │
                │ (formato YOLO)        │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ convert_dataset.py    │
                │ (detección → clases)  │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ data/processed        │
                │ (ImageFolder ready)   │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │    train.py           │
                │ (entrena ResNet18)    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │ models/resnet18.pth   │
                │ (modelo final)        │
                └──────────┬────────────┘

Produccion 

  ┌───────────────────────┐
                │ docker-compose up     │
                │ (levanta API)         │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │    FastAPI API        │
                │  (src/api/app.py)     │
                └──────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  /predict     │  │ /predict_yolo │  │  /gradcam     │
│  CNN model    │  │ YOLO model    │  │ GradCAM       │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
   model.py         YOLO model         gradcam.py
   (ResNet18)       baseline           explicación

Visualizacion
                ┌───────────────────────┐
                │   Streamlit App       │
                │  (src/app/...)        │
                └──────────┬────────────┘
                           │
                           ▼
                (envía imagen a API)
                           │
                           ▼
          ┌──────────────────────────────────┐
          │       RESULTADOS MOSTRADOS       │
          │                                  │
          │ ✅ CNN predicción                │
          │ ✅ YOLO predicción               │
          │ ✅ GradCAM (heatmap)             │
          └──────────────────────────────────┘