# DeepLence

### AI-Powered Digital Media Forensics Platform

> **Investigate suspicious images and videos using transparent forensic evidence.**

---

## The Problem

In 2026, AI-generated images and videos have become incredibly realistic. Deepfakes, identity theft, misinformation, fake evidence, and impersonation scams are becoming more common, making it increasingly difficult to trust what we see online.

Most AI detection tools return only a confidence score, giving users little insight into *why* a piece of media was flagged.

DeepLence was built to provide a more transparent approach by combining AI-powered analysis with digital forensic techniques to produce an evidence-based report.

---

## Our Solution

DeepLence is a digital media forensics platform that helps users investigate suspicious images and videos.

Instead of relying on a single AI prediction, DeepLence gathers multiple independent signals and presents them as forensic evidence, allowing users to better understand the reasoning behind each result.

The goal is not to prove that media is real or fake, but to provide an explainable forensic assessment that supports informed decision-making.

---

## Features

* 🖼️ Image analysis
* 🎥 Video analysis
* 🔍 AI-generated media risk assessment
* 📄 Metadata inspection
* 🛡️ C2PA Content Credentials (Provenance) verification
* 🧬 Generator fingerprint analysis
* 📊 Explainable forensic report
* ⚠️ Confidence and analysis limitations
* 📈 Evidence-based risk scoring

---

## How DeepLence Works

Every uploaded file goes through multiple stages of analysis:

1. Upload image or video
2. Provenance verification (C2PA)
3. Metadata inspection
4. AI model analysis
5. Digital forensic analysis
6. Evidence fusion
7. Forensic report generation

The final report combines all available evidence into a single explainable assessment rather than relying on a single prediction.

---

## Digital Forensics Techniques

DeepLence combines several complementary techniques, including:

* Pixel-level AI analysis
* Metadata inspection
* Error Level Analysis (ELA)
* Frequency-domain (FFT) analysis
* Noise residual analysis
* Generator fingerprint confidence
* Video temporal consistency analysis
* C2PA provenance verification

Each technique contributes supporting evidence to the final report.

---

## Built With

### Backend

* Python
* FastAPI

### Computer Vision & AI

* Sightengine GenAI API
* Hugging Face models
* Vision Transformers
* OpenCV
* Pillow
* NumPy

### Digital Forensics

* Error Level Analysis
* FFT Frequency Analysis
* Noise Residual Analysis
* Metadata Analysis
* C2PA Content Credentials

---

## How OpenAI Helped Build DeepLence

DeepLence was developed with significant assistance from **GPT-5.6** and **OpenAI Codex**.

GPT-5.6 acted as an engineering collaborator throughout development by helping:

* Design the overall system architecture.
* Plan the forensic analysis pipeline.
* Recommend suitable Hugging Face models and AI detection approaches.
* Design how multiple forensic signals should be combined into an explainable report.
* Review engineering decisions and suggest improvements.
* Debug backend logic and API integration.
* Improve documentation and code quality.

OpenAI Codex accelerated implementation by assisting with code generation, debugging, refactoring, testing, and development across the project.

The project author made all final engineering decisions, selected the technologies, implemented the application, integrated external services, and designed the overall user experience.


## Installation

```bash
git clone https://github.com/salmansanusisani/deeplence.git

cd deeplence

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open:

```
http://localhost:8000
```



## OpenAI Build Week Hackathon 2026

DeepLence was created for the **OpenAI Build Week Hackathon 2026**.

The project explores how AI can be used to to help people investigate suspicious digital content through transparent, explainable forensic analysis.
