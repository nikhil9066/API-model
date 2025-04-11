
# 💬 Chat App — Lightweight LLM Interface

This is a simple web-based chatbot built using Python and Flask (or a similar framework) that interacts with a language model engine (LLM). It's part of a larger project and is located in the `model/chat/` directory.

---

## 📂 Project Structure

```
model/chat/
├── app.py              # Main app entry point
├── llm_engine.py       # Handles LLM interaction logic
├── requirements.txt    # Dependencies specific to the chat app
├── Dockerfile          # Docker configuration file
├── docker-compose.yml  # Optional Docker Compose file (for multi-container setups)
├── static/             # Static assets (CSS, JS)
└── templates/
    └── index.html      # Chat interface template
```

---

## 🚀 Getting Started

Follow these steps to install dependencies and run the chat app both **locally** or **on Docker**.

### 1. Navigate to the Chat App Directory

```bash
cd model/chat
```

### 2. (Optional) Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

- **Locally**: After installing dependencies, you can run the app directly using Flask:

  ```bash
  python app.py
  ```

  Once running, open your browser and visit:

  ```
  http://127.0.0.1:5000/
  ```

- **Using Docker**: If you prefer to run the app inside a container, use the following steps.

  1. Build the Docker image:

      ```bash
      docker build -t my-python-app .
      ```

  2. Run the Docker container:

      ```bash
      docker run -p 5000:5000 my-python-app
      ```

  After the container is up and running, access the app at:

  ```
  http://127.0.0.1:5000/
  ```

---

## 🧠 How It Works

- `app.py` sets up a web server and routes requests.
- `llm_engine.py` manages communication with a language model (e.g., OpenAI, Ollama, etc.).
- `index.html` provides the browser-based chat interface.

---

## ⚙️ Configuration

Make sure to set up your LLM engine configuration in `llm_engine.py`. This may include:

- API keys for services like OpenAI
- Local endpoints for models like Ollama or LLaMA.cpp

### Example (OpenAI):

```python
import openai
openai.api_key = "your-api-key"
```

### Example (Ollama):

```python
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3",
    "prompt": "Hello!",
    "stream": False
})
```

---

## ✅ Example Use

1. Start the server (locally or via Docker).
2. Go to `http://127.0.0.1:5000/`
3. Type your message and hit "Send".
4. Get responses directly from the LLM.

---

## 🛠 Requirements

- Python 3.7 or higher
- All Python dependencies listed in `requirements.txt`
- Docker (if running the app in a container)

---

## 📬 Feedback & Contributions

Feel free to open issues or submit pull requests to improve the app or add support for more LLM backends.

---