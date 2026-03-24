# GUVI-HACKATHON 🚀

An AI-powered project developed as part of the **GUVI Hackathon**, focused on building a practical, real-world solution using modern technologies.

---

## 🧠 Project Overview

This project aims to solve a real-world problem by leveraging **Artificial Intelligence and modern software tools**.  
The solution is designed to be scalable, user-friendly, and impactful, aligning with hackathon evaluation criteria such as **innovation, feasibility, and technical depth**.

---

## 🎯 Problem Statement

(Briefly explain the problem your project addresses)

Example:
> With the increasing demand for intelligent systems, there is a need for an automated solution that can efficiently handle and analyze user data to provide accurate results in real time.

---

## 💡 Solution

Our solution uses AI-driven logic to:
- Automate decision-making
- Improve accuracy and efficiency
- Reduce manual effort
- Deliver fast and reliable outputs

The system processes input data, applies intelligent algorithms, and produces meaningful results through an easy-to-use interface.

---

## 🛠️ Tech Stack

- **Programming Language:** TypeScript / Python  
- **Frontend:** React 19, Vite, Tailwind CSS  
- **AI / ML:**  
  - Google Gemini 3-Flash (voice authenticity classification)  
  - **FAISS** (Facebook AI Similarity Search) – 128-dim voice embedding index for fast similarity search  
  - Librosa – audio feature extraction (MFCCs, spectral centroid, chroma, …)  
- **Backend:** Python Flask + Flask-CORS  
- **Tools:** Git, GitHub, VS Code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/rahulsvt-1907/GUVI-HACKATHON.git
cd GUVI-HACKATHON
```

### 2️⃣ Frontend (React)
```bash
npm install
# Create a .env file and add your Gemini API key:
# GEMINI_API_KEY=your_google_genai_api_key_here
npm run dev        # http://localhost:3000
```

### 3️⃣ Backend – FAISS Similarity Search (Python)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py               # http://localhost:5000
```

> See [backend/README.md](backend/README.md) for full API reference and usage examples.


---

## ✅ Next Steps (Recommended)
I can:
1. Customize this README **exactly** to your project idea  
2. Add badges (AI, Hackathon, Python, Node)  
3. Rewrite it to score **maximum hackathon points**  
4. Add a **proper problem–solution narrative** for judges  

Just tell me:
👉 **What exactly does your project do in 1–2 lines?**


