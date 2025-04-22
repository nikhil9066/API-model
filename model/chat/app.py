from flask import Flask, render_template, request, jsonify
from llm_engine import wrap_prompt, call_llm, run_code  # from your existing code

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query", "")
    if not user_input:
        return jsonify({"type": "error", "result": "No query received."})

    wrapped = wrap_prompt(user_input)
    llm_code = call_llm(wrapped)

    if not llm_code:
        return jsonify({"type": "error", "result": "LLM did not return any code."})

    result_type, result = run_code(llm_code)

    if result_type == "TEXT":
        return jsonify({"type": "text", "result": result})
    elif result_type == "PLOT":
        return jsonify({"type": "plot", "result": "/static/plot.png"})
    else:
        return jsonify({"type": "error", "result": result})

if __name__ == "__main__":
    app.run(debug=True)