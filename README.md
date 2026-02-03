# 🌱 CotVision

CotVision is an AI-based agricultural diagnosis platform for the detection of foliar diseases in cotton leaves, utilizing computer vision, deep learning, and interpretability via Grad-CAM.

---

## 🚀 Key Features

* 📷 Cotton leaf image upload
* 🤖 Inference via AI model
* 📊 Visualization of probabilities per class
* 🔍 Visual interpretability via Grad-CAM
* 🧩 Decoupled architecture (React + FastAPI)
* 📄 Automatic API documentation (Swagger)

---

## 🏗️ Project Architecture

```text
cotvision/
├── frontend/   # User Interface (React + TypeScript + Material UI)
└── backend/    # REST API (FastAPI)

```

---

## 🖥️ Frontend

### Technologies Used

* React
* TypeScript
* Material UI
* React Router
* Axios
* Recharts
* Vite

### 📦 Frontend Installation

```bash
cd frontend
npm install

```

### ▶️ Running the Frontend

```bash
npm run dev

```

The application will be available at:

```
http://localhost:5173

```

---

## ⚙️ Backend

### Technologies Used

* Python 3.9+
* FastAPI
* Uvicorn
* Pydantic
* Pillow
* python-multipart

### 📦 Backend Installation

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### ▶️ Running the Backend

```bash
uvicorn app.main:app --reload --port 8000

```

The API will be available at:

```
http://localhost:8000

```

Interactive API documentation (Swagger UI):

```
http://localhost:8000/docs

```

---

## 🔄 Workflow

1. The user uploads an image via the web interface
2. The image is sent to the backend via the `POST /predict` endpoint
3. The backend processes the image
4. The prediction is returned containing:
* Detected class
* Confidence level
* Probabilities per class
* Inference time
* Grad-CAM (heatmap)


5. The frontend displays the results, metrics, and visual interpretability

---
