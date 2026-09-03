# AI Vision & Object Detection Suite

An intelligent multi-mode computer vision and AI analysis suite featuring:
1. **Interactive Web Application**: Client-side object detection (COCO-SSD) paired with **Google Gemini 3.6 Flash** for live scene description and real-time misclassification verification.
2. **AI Bill & Document Video Scanner**: Rapid multi-document keyframe extraction and structured JSON/CSV export for invoices, receipts, and shipping labels (Delhivery, Amazon, Flipkart, etc.).
3. **Python YOLO CLI**: High-performance detection on images, videos, and webcam streams via Ultralytics YOLO (`yolo26s.pt`).

---

## 📁 Project Structure

```
object-detection/
├── index.html           # Web app UI
├── app.js               # Web app frontend logic, video rendering, Gemini integration
├── style.css            # Cyber-industrial dark theme UI styles
├── bill_scanner.py      # Python CLI tool for video bill extraction
├── detect.py            # Python YOLO detection script (image / video / webcam)
├── config.example.js    # Browser configuration template
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
├── setup_env.ps1        # PowerShell setup script
└── README.md
```

---

## 🚀 Quick Start

### 1. Configure Secrets

Copy the example configuration files and add your [Google AI Studio Gemini API key](https://aistudio.google.com/app/apikey):

```powershell
# Copy environment variable template
Copy-Item .env.example .env

# Copy web browser config template
Copy-Item config.example.js config.js
```

Edit `.env` and `config.js` with your API key:
```javascript
// config.js
window.GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE';
```

### 2. Run the Web Application

Simply open `index.html` in your browser (or use a local HTTP server like Live Server or `python -m http.server 8000`):

- **Image Mode**: Upload any photo to detect objects and generate an observational AI scene description.
- **Video Mode**: Upload a video with full interactive scrubbing and live bounding box tracking.
- **Webcam Mode**: Live webcam object detection with automated Gemini label correction (e.g. computer mouse vs. cell phone).
- **Bill Scanner Mode**: Upload a video showcasing bills, receipts, or courier tags; automatically extracts vendor, AWB/invoice number, line items, and totals into interactive cards and CSV/JSON downloads.

---

## 🐍 Python Setup & CLI Tools

### Setup Virtual Environment

```powershell
.\setup_env.ps1
# Or manually:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run YOLO Object Detection

```powershell
# Image
python detect.py --mode image --source inputs/photo.jpg

# Video
python detect.py --mode video --source inputs/clip.mp4

# Live Webcam
python detect.py --mode webcam
```

### Run Video Bill Scanner (Python)

```powershell
python bill_scanner.py --video inputs/bills.mp4 --output outputs/bills.json --debug
```

---

## 🛡️ Security

`.env` and `config.js` are configured in `.gitignore` to ensure private API keys are never committed to version control.

