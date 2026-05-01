# 🩻 MedVision AI – Multimodal Medical Assistant

## 📌 Project Overview

MedVision AI is a **multimodal AI system** that analyzes medical images along with user queries to generate structured diagnostic insights.

It combines:

* Computer Vision (image understanding)
* Vector Database (semantic search)
* Large Language Models (text generation)

---

## 🚀 Features

* 📷 Upload medical images (X-rays, scans)
* 📝 Enter symptoms or medical queries
* 🧠 AI-generated:

  * Diagnosis
  * Findings
  * Recommendations
* 🔍 Retrieves similar medical cases using vector search
* ⚡ Real-time interface using Gradio

---

## 🧠 How It Works

### 🔄 Pipeline

1. **Image Input**

   * User uploads a medical image

2. **Image Captioning**

   * Image → text description
   * Model: BLIP

3. **Embedding Generation**

   * Converts text & images → vectors
   * Library: FastEmbed

4. **Vector Search (RAG)**

   * Uses Qdrant to retrieve similar cases

5. **LLM Processing**

   * Combines:

     * user query
     * image caption
     * retrieved context
   * Generates diagnosis

---

## 🏗️ Tech Stack

* **Frontend**: Gradio
* **Backend**: Python
* **Image Captioning**: BLIP
* **Embeddings**: FastEmbed
* **Vector DB**: Qdrant
* **LLM**: LLaMA (Groq API)

---

## 📂 Project Structure

```
healthcare_multimodal_ai/
│
├── data/                  # Dataset (images + captions)
├── src/
│   ├── main.py           # Gradio UI
│   ├── multimodal_rag_system.py
│   ├── create_data_embeddings.py
│   ├── embeddings_utils.py
│   ├── gpt_utils.py
│   └── image_captioning.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repo

```
git clone https://github.com/ziya1122/llm_project.git
cd llm_project
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 🔑 API Setup

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run Project

```
cd src
python main.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

## ⚠️ Known Issue (IMPORTANT)

You may encounter this error:

```
FileNotFoundError: No such file or directory
```

### ✅ Fix

Ensure directory exists before saving image:

```python
os.makedirs(base_dir, exist_ok=True)
```

Also avoid hardcoded paths like:

```
/Users/sarthak/...
```

Use dynamic path:

```python
os.path.join(os.path.dirname(__file__), "..", "data")
```

---

## 📊 Sample Output

```
Diagnosis:
Possible fracture of forearm.

Findings:
Bone misalignment observed.

Recommendation:
Further imaging (CT scan) suggested.
```

---

## ⚠️ Limitations

* Model does not directly see images (uses captions)
* Accuracy depends on dataset quality
* Not a replacement for professional diagnosis

---

## 🔮 Future Improvements

* Use vision-language models (true multimodal LLM)
* Improve dataset quality
* Add confidence score
* Deploy online (cloud)

---

## 👩‍💻 Author

* Ziya Dias

---

## 📜 License

This project is for educational purposes only.
