# 🌱 CotVision

CotVision es una plataforma de diagnóstico agrícola basada en inteligencia artificial
para la detección de enfermedades foliares en hojas de algodón, utilizando visión por
computador, deep learning e interpretabilidad mediante Grad-CAM.

---

## 🚀 Características principales

* 📷 Carga de imágenes de hojas de algodón
* 🤖 Inferencia mediante modelo de IA
* 📊 Visualización de probabilidades por clase
* 🔍 Interpretabilidad visual mediante Grad-CAM
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
3. El backend procesa la imagen
4. Se devuelve la predicción con:

   * Clase detectada
   * Nivel de confianza
   * Probabilidades por clase
   * Tiempo de inferencia
   * Grad-CAM (heatmap)
5. El frontend muestra los resultados, métricas e interpretabilidad visual

---
