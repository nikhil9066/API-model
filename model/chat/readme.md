
# 💬 Chat App — Lightweight LLM Interface

This is a simple web-based chatbot built using Python and Flask that interacts with a language model engine (LLM). It's part of a larger project and is located in the `model/chat/` directory.

---

## 📂 Project Structure

```
chat/
├── app.py              # Main app entry point
├── llm_engine.py       # Handles LLM interaction logic
├── requirements.txt    # Dependencies specific to the chat app
├── setup_and_run.sh    # Script to setup, launch and clean the app
├── chat_history.json   # File to store interaction history
├── static/             # Static assets (e.g., plots)
└── templates/
    └── index.html      # Chat interface template
```

---

## 🚀 Getting Started

### ✅ One-Step Setup & Run

Run the following command inside the `chat/` directory:

```bash
bash setup_and_run.sh
```

This script will:

1. Check if Python is installed (and prompt installation if not).
2. Set up a virtual environment.
3. Install dependencies.
4. Start the Flask server.
5. Open `http://127.0.0.1:5000/` in your browser.
6. Press ENTER anytime to:
   - Kill the server.
   - Clear chat history and static files.
   - Delete `__pycache__`, virtual environment, and clean up.

---

## 🧠 How It Works

- `app.py`: Launches a web server and manages routes.
- `llm_engine.py`: Communicates with the language model.
- `setup_and_run.sh`: Automates setup, run, and teardown of the app.

---

## ⚙️ LLM Configuration

Update the `llm_engine.py` file to configure your model (e.g., OpenAI, Ollama).

### Example (Ollama):

```python
import ollama

response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": "Hello!"}
])
```

Make sure you have Ollama installed and running locally: https://ollama.com

---

## ✅ Example Use

1. Run `bash setup_and_run.sh`
2. Chat with the LLM at `http://127.0.0.1:5000/`
3. View responses and plots based on your dataset.

---

## 🛠 Requirements

- Python 3.7 or higher
- Ollama (if using a local model)
- Bash shell (for running the script on Mac/Linux)

---

## 📬 Feedback & Contributions

Feel free to open issues or submit pull requests to improve the app or add support for other LLMs.

---