# Adaptive UI Recommender System (AI-HCI Project)

This project implements an **AI-based adaptive user interface** that changes its behavior based on user interactions. It was developed as part of the **Artificial Intelligence – Human-Computer Interaction (AI-HCI)** course.

The system combines:
- A lightweight **recommendation engine**
- An adaptive UI response through a **conversational bot**
- Optional API-driven recommendations

---

## 🚀 Features

✅ Adaptive UI behavior based on user inputs  
✅ `/recommend <userId> <k>` dynamic recommendations  
✅ Graceful handling of invalid or cold-start users  
✅ Hero Card carousel UI responses (Bot Framework Emulator)  
✅ FastAPI service for real-time recommendations  
✅ Fully local — no external dependencies required  
✅ Ready for extension into personalization or usability studies

---

## 📂 Repository Structure


---

## 🧠 How the Adaptive Behavior Works

1. The user interacts with the bot (e.g., `hello`, `/recommend 1 3`)
2. The bot detects intent and adapts response:
   - Greeting → contextual help
   - Valid recommendation → carousel UI
   - Invalid input → friendly fallback message
3. Recommendations are generated using:
   - Item-item similarity
   - Content-based TF-IDF features
   - Popularity fallback for cold users

This demonstrates **AI-based adaptive HCI**, where the interface *changes based on the user*, instead of being static.

---

## ▶️ Quick Start

### 1️⃣ Create & activate environment (optional)
```bash
conda create -n ai_hci python=3.11 -y
conda activate ai_hci
pip install -r requirements.txt
python serve.py
GET http://127.0.0.1:8000/recommend?user=<id>&k=<k>
set PORT=3978
python bot_app.py
hello
/help
/recommend 1
/recommend 2 5
/recommend 42         # cold-start user
/recommend 3 -1       # invalid → adaptive fallback

