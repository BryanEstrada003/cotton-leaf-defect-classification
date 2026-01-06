# 🌱 CotVision

CotVision es una plataforma de diagnóstico agrícola basada en inteligencia artificial
para la detección de enfermedades foliares en hojas de algodón, utilizando visión por
computador, deep learning e interpretabilidad mediante Grad-CAM.

El sistema está diseñado con una arquitectura desacoplada frontend–backend que permite
comparar distintos modelos de IA (KAN y VGG16) y visualizar sus decisiones de manera
explicable, facilitando la toma de decisiones en el ámbito agrícola.

---

## 🚀 Características principales

* 📷 Carga de imágenes de hojas de algodón
* 🤖 Inferencia mediante modelos de IA (mock y modelos reales)
* 📊 Visualización de probabilidades por clase
* 🔍 Interpretabilidad visual mediante Grad-CAM
* 🔁 Comparación entre modelos KAN y VGG16
* 🧩 Arquitectura desacoplada (React + FastAPI)
* 📄 Documentación automática de la API (Swagger)

---

## 🏗️ Arquitectura del proyecto

```text
cotvision/
├── frontend/   # Interfaz gráfica (React + TypeScript + Material UI)
└── backend/    # API REST (FastAPI)
```

---

## 🖥️ Frontend

### Tecnologías utilizadas

* React
* TypeScript
* Material UI
* React Router
* Axios
* Recharts
* Vite

### 📦 Instalación del frontend

```bash
cd frontend
npm install
```

### ▶️ Ejecución del frontend

```bash
npm run dev
```

La aplicación estará disponible en:

```
http://localhost:5173
```

---

## ⚙️ Backend

### Tecnologías utilizadas

* Python 3.9+
* FastAPI
* Uvicorn
* Pydantic
* Pillow
* python-multipart

### 📦 Instalación del backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Copiar models a backend

Crear carpeta models/ si no existe y copiar modelo VGG16 y renombrar a vgg16_model.h5 y pesos de KAN y renombrar a kan_model.pth

### ▶️ Ejecución del backend

```bash
uvicorn app.main:app --reload --port 8000
```

La API estará disponible en:

```
http://localhost:8000
```

Documentación interactiva de la API (Swagger UI):

```
http://localhost:8000/docs
```

---

## 🔄 Flujo de funcionamiento

1. El usuario carga una imagen desde la interfaz web
2. La imagen se envía al backend mediante el endpoint `POST /predict`
3. El backend procesa la imagen (mock o modelo real)
4. Se devuelve la predicción con:

   * Clase detectada
   * Nivel de confianza
   * Probabilidades por clase
   * Modelo utilizado
   * Tiempo de inferencia
   * Grad-CAM (heatmap)
5. El frontend muestra los resultados, métricas e interpretabilidad visual

---

## 🔬 Comparación de modelos

CotVision está diseñado para trabajar con dos enfoques de aprendizaje profundo:

* **VGG16**
  Modelo CNN tradicional utilizado como baseline para la clasificación de imágenes.

* **KAN (Kolmogorov–Arnold Network)**
  Modelo alternativo que busca mejorar la interpretabilidad y el enfoque funcional del aprendizaje.

La interfaz permite:

* Seleccionar el modelo a utilizar
* Visualizar resultados individuales
* Comparar métricas y Grad-CAM entre modelos

---

## 🧪 Estado actual del proyecto

* ✅ Interfaz gráfica completa y funcional
* ✅ Backend FastAPI con modelo mock
* ✅ Pipeline imagen → predicción → visualización
* ✅ Grad-CAM simulado integrado
* ⏳ Integración de modelos reales (en progreso)

---

## 📌 Consideraciones académicas

Este proyecto fue desarrollado con fines académicos y de investigación,
siguiendo buenas prácticas de ingeniería de software, diseño de APIs,
arquitectura desacoplada e interpretabilidad de modelos de inteligencia artificial.

---

## 👥 Autores

Proyecto desarrollado por estudiantes de ingeniería como parte de un trabajo
académico en inteligencia artificial y visión por computador.

---

##
