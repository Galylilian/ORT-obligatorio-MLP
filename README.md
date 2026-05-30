[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hKg-8jSY)
# Práctico 4: Model Serving

En este práctico se implementará un sistema de clasificación de texto utilizando modelos pre-entrenados de Hugging Face Transformers y FastAPI para crear una API REST.

## Contenido

- Usamos un sistema de clasificación de texto usando modelos de Transformers
- Creamos una API REST con FastAPI para servir predicciones
- Realizamos pruebas de rendimiento y calidad del modelo


## Estructura del Proyecto

```
practico-4-2026/
├── src/                     # Código fuente
│   ├── api/                # Endpoints de la API
│   │   ├── routers/       # Definición de rutas
│   │   └── app.py         # Aplicación FastAPI
│   ├── core/              # Lógica principal
│   │   ├── classification.py
│   │   └── preprocessing.py
│   ├── settings/          # Configuración
│   └── utils/             # Utilidades
├── tests/                 # Tests unitarios
├── Dockerfile            # Configuración de Docker (Extra)
├── docker-compose.yml    # Configuración de Docker Compose (Extra)
└── requirements.txt      # Dependencias del proyecto
```

## 1. FastAPI y Transformers

### FastAPI

FastAPI es un framework moderno y de alto rendimiento para construir APIs con Python. Sus principales ventajas son:

- **Rendimiento**: Uno de los frameworks más rápidos disponibles, comparable a NodeJS y Go
- **Documentación Automática**: Genera automáticamente documentación interactiva (Swagger/OpenAPI)
- **Validación de Tipos**: Integración nativa con el sistema de tipos de Python
- **Async/Await**: Soporte nativo para operaciones asíncronas
- **Fácil de Usar**: Sintaxis intuitiva y minimalista
- **Basado en Estándares**: Compatible con OpenAPI y JSON Schema

### Transformers

La biblioteca Transformers de Hugging Face proporciona acceso a múltiples modelos pre-entrenados para procesamiento de lenguaje natural. Sus características principales incluyen:

- **Modelos Pre-entrenados**: Acceso a modelos de última generación
- **Fácil de Usar**: API simple y consistente para todos los modelos
- **Optimización**: Soporte para aceleración por hardware (GPU/TPU)
- **Multilingüe**: Modelos disponibles en múltiples idiomas

## 2. Configuración del Entorno

### Configuración del Entorno de Desarrollo

1. Crear entorno virtual:

   ```bash
   # macOS/Linux
   virtualenv venv
   source venv/bin/activate

   # Windows
   virtualenv venv
   .\venv\Scripts\activate
   ```
2. Instalar dependencias:

   ```bash
   # Instalar dependencias principales
   pip install -r requirements.txt

   # Instalar dependencias de desarrollo
   pip install -r dev-requirements.txt
   ```

Para levantar el servidor de forma local, usar el comando de Poetry:


```bash
poetry run python src/api/app.py

```
Si no usa Poetry, el comando es:

```bash
python src/api/app.py

```
También se puede crear un contenedor de Docker para ejecutar el práctico, más información al final de este documento.

## 3. Clasificación de Texto

La clasificación de texto es una tarea fundamental en NLP que consiste en asignar una o más categorías a un texto dado. En nuestro caso, utilizaremos modelos pre-entrenados de Hugging Face para realizar esta tarea.

### Procesamiento de Texto

El pipeline de procesamiento incluye:

1. **Tokenización**: Convertir texto en tokens
2. **Embedding**: Generar representaciones vectoriales
3. **Clasificación**: Asignar categorías basadas en los embeddings

### Batch Processing

Para optimizar el rendimiento, implementamos procesamiento por lotes:

- Procesar múltiples textos simultáneamente
- Reducir overhead de comunicación
- Mejorar throughput del sistema


## 4. Ejemplo de Resultados

El sistema es capaz de clasificar textos con alta precisión. Por ejemplo, al enviar el siguiente request:

```json
{
    "texts": [
        "I love this class",
        "I don't like this music!"
    ]
}
```

El sistema responde con las clasificaciones detalladas:

```json
{
  "texts": [
    {
      "text": "I love this class",
      "label": "positive",
      "score": 0.9686873555183411,
      "metadata": {
        "scores": {
          "positive": 0.9686873555183411,
          "neutral": 0.02231706492602825,
          "negative": 0.008995560929179192
        }
      }
    },
    {
      "text": "I don't like this music!",
      "label": "negative",
      "score": 0.9389562606811523,
      "metadata": {
        "scores": {
          "negative": 0.9389562606811523,
          "neutral": 0.053173452615737915,
          "positive": 0.007870309986174107
        }
      }
    }
  ],
  "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

Ejemplo de request por consola:

```
curl -X POST http://localhost:8080/classification/texts   -H "Content-Type: application/json"   -d '{
    "texts": [
        "I love this class"
    ]
}'

```

La respuesta incluye:

- El texto original
- La etiqueta predicha (label)
- El score de confianza para la predicción
- Metadatos con los scores para todas las clases
- El ID del modelo utilizado


## 5. Testing

El proyecto incluye tests de integración para asegurar la calidad del código:

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

```

### Tipos de Tests

- **API**: Endpoints, respuestas y manejo de errores
- **Clasificación**: Precisión del modelo y casos límite
- **Preprocesamiento**: Limpieza y normalización de texto

### Mejores Prácticas

- Usar fixtures de pytest para configuración
- Mockear dependencias externas
- Probar casos límite y errores

## 6. Ejercicio Práctico

### Clasificación de Imágenes

Implementa un sistema de clasificación de imágenes, usando Hugging Face. Recomendaciones:

- Buscar modelos que no tengan un peso muy grande (<800M de parámetros).
- Verificar el tiempo de inferencia.
- Priorizar modelos que funcionen bien en CPU.

#### Tareas:

1. Crear nuevo endpoint para la clasificación de imágenes.
2. Probar con diferentes inputs de imágenes.

#### Entregables:

1. Código adaptado
2. README con:
   - Modelo usado y su tamaño
   - Resultados de ejemplos de clasificación de imágenes.
   - Tiempos de inferencia
  
Opcionalmente puede agregar preprocesamiento a las imágenes de las requests antes de hacer la clasificación.

## 7. Recursos Adicionales




### Configuración de Docker

Hay dos formas de ejecutar la aplicación con Docker:

##### Opción 1: Docker Directo

1. Construir la imagen:

   ```bash
   docker build -t text-classification-api .
   ```
2. Ejecutar el contenedor:

   ```bash
   # Ejecutar en modo detached
   docker run -d -p 8080:8080 text-classification-api

   # Ver logs
   docker logs -f <container_id>
   ```

##### Opción 2: Docker Compose

Docker Compose es una herramienta para definir y ejecutar aplicaciones multi-contenedor. Sus principales ventajas son:

- **Definición de Servicios**: Permite definir todos los servicios necesarios en un archivo YAML
- **Entorno Aislado**: Cada servicio corre en su propio contenedor
- **Desarrollo Consistente**: Garantiza que todos los desarrolladores usen la misma configuración
- **Fácil Despliegue**: Un solo comando para levantar toda la aplicación
- **Gestión de Dependencias**: Maneja automáticamente las dependencias entre servicios
- **Variables de Entorno**: Centraliza la configuración de variables de entorno
- **Volúmenes**: Facilita el manejo de datos persistentes y desarrollo en tiempo real

1. Ejecutar con Docker Compose (construye la imagen automáticamente):

   ```bash
   # Ejecutar en modo detached
   docker-compose up -d

   # Ver logs
   docker-compose logs -f
   ```
2. Para detener los contenedores:

   ```bash
   # Si usaste Docker directo
   docker stop <container_id>

   # Si usaste Docker Compose
   docker-compose down
   ```



- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Docker Documentation](https://docs.docker.com/)
