# Industrial Pipe Counter 🔍

A high-performance computer vision application for automated pipe and steel rod counting with dual ONNX inference models, specialized type-breaker rules, and Gemini glare/structural analysis.

---

## 🌟 Key Features

- **Dual-Stream Detection**:
  - **Hollow Pipe Stream**: Trained YOLO model (`best_hollow.onnx`) with circle-detection type-breaker validation.
  - **Solid Pipe / Steel Rod Stream**: Trained YOLO model (`best_solid.onnx`) specialized for solid round steel bars and rods.
- **Multimodal AI Analysis**: Integrated Gemini Flash analysis for intelligent glare reduction evaluation and structural inspection reporting.
- **Side-by-Side Synchronized Comparison**: Synchronized visual comparison of raw inspection photos alongside detection bounding circles.
- **Sample Test Suite**: 10 preloaded inspection test images (3 hollow, 7 solid) with automatic stream alignment and 1-click test runs.
- **Full-Stack Architecture**:
  - **Backend**: Node.js + Express serving the API and ONNX runtime (`onnxruntime-node`).
  - **Frontend**: React dashboard with adjustable confidence thresholds, NMS IOU cutoffs, and YOLO input resolution scaling.

---

## 📁 Repository Structure

```
├── hollow_specific/       # Hollow pipe ONNX model & type-breaker logic
├── solid_specific/        # Solid pipe ONNX model & type-breaker logic
├── pip-counter-backend/   # Express.js REST API & ONNX runtime server
├── pipe-counter-ui/       # React inspection dashboard
├── Sample_images/         # Test dataset (hollow_*.jpg, solid_*.jpg)
├── gemini_process.py      # Multimodal Gemini glare/structural inspection script
├── main_app.py            # Streamlit dashboard interface
└── .env.example           # Example environment configuration
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js** (v18+)
- **Python** (3.10+) with required packages: `pip install google-genai pillow python-dotenv`

### 2. Environment Setup
Copy the example environment file and add your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5005
```

### 3. Backend Setup
```bash
cd pip-counter-backend
npm install
node server.js
```
The backend runs on `http://localhost:5005`.

### 4. Frontend Setup
```bash
cd ../pipe-counter-ui
npm install
npm start
```
The frontend dashboard will be available at `http://localhost:3000`.

---

## ⚙️ Cloudflare Tunnel Deployment

The application is pre-configured to run behind a Cloudflare Tunnel:
```powershell
cloudflared.exe tunnel run --token <YOUR_TUNNEL_TOKEN>
```
Route the tunnel ingress to `http://localhost:5005` to serve both the static UI and the `/detect` API under a unified hostname without CORS restrictions.
