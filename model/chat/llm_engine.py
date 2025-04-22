import ollama
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import contextlib
import os
import json

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import traceback

# Assuming df is loaded from an external file, you can replace this line with the actual file path
df = pd.read_csv('/Users/nikhilprao/Documents/Data/Boston.csv', index_col=0)
chat_history = []

basic_info = """
You are a Python data analyst. The dataframe `df` has already been loaded and it contains the following columns:
"""

df_columns_list = df.columns.tolist()
column_string = " and the sample dataset looks like this:\n"
sample = df.head(3)

pre_information = basic_info + ", ".join(df_columns_list) + column_string + sample.to_string(index=False)

def export_chat_history(filename="chat_history.json"):
    # Export chat history to a JSON file
    with open(filename, "w") as f:
        json.dump(chat_history, f, indent=4)

def wrap_prompt(user_input):
    """
    Wraps the user's query into a format that the LLM can understand.
    This ensures no plot is generated if the user mentions 'plot' or 'visualize'.
    """
    
    # Check if the user asked for a plot (as already implemented)
    if 'plot' in user_input.lower() or 'visualize' in user_input.lower():
        print("User asked for a plot, return code for plotting")
        return f"""
        {pre_information}

        User asked: "{user_input}"

        Write Python code using `df`. If it’s a plot, use matplotlib and save it to 'static/plot.png'. 
        If it’s just a textual output, assign it to `output = ...` using `to_string()` if it’s a DataFrame.

        DO NOT use print(). DO NOT use df.to_png().
        Only do what the user asked — do not add extra visualizations or summaries unless explicitly requested.
        Only return the code.
        """
    
    # Check for descriptive statistics or analysis requests
    elif re.search(r'(summary|describe|statistics|mean|median|std|count|info)', user_input, re.IGNORECASE):
        print("User asked for descriptive statistics or analysis, return Python code")
        return f"""
        {pre_information}

        User asked: "{user_input}"

        Write Python code using `df` to return relevant descriptive statistics or perform the analysis requested.
        For example, use `df.describe()`, `df.info()`, or `df.dtypes` as appropriate.
        DO NOT create plots or figures unless the user explicitly asks for one.
        Only return the code.
        """
    
    # Generalized analysis or code queries
    else:
        print("User asked a generalized analysis question, return code for analysis")
        return f"""
        {pre_information}

        User asked: "{user_input}"

        Write Python code using `df` to return relevant data or perform the requested analysis.
        DO NOT create plots or figures unless the user explicitly asks for one.
        Only return the code.
        """

def call_llm(prompt):
    """
    Calls the local LLM (via Ollama) to generate Python code for the user's query.
    """
    chat_history.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(
            model="llama3.2",  # Use the local LLM model you downloaded
            messages=chat_history,
            options={"temperature": 0.1}
        )

        code = response['message']['content']
        chat_history.append({"role": "assistant", "content": code})
        export_chat_history()
        return code
    except Exception as e:
        # If an error occurs, simply move on and do not add the funny message
        return None

def run_code(code):
    if "```python" in code and "```" in code:
        parts = code.split("```python")
        code_block = parts[1].split("```")[0]
        code = code_block.strip()
    else:
        code = code.strip()

    local_env = {"df": df, "plt": plt}
    plot_path = "static/plot.png"

    try:
        # Clean previous plot if any
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Capture print output
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {}, local_env)

        # Priority 1: return 'output = ...' variable
        if 'output' in local_env:
            return "TEXT", str(local_env['output'])

        # Priority 2: check if a plot was saved
        if os.path.exists(plot_path):
            return "PLOT", plot_path

        # Priority 3: return anything that was printed (e.g., df.info())
        printed = stdout_capture.getvalue().strip()
        if printed:
            return "TEXT", printed

        return "TEXT", "Execution completed, but nothing was returned."

    except Exception as e:
        funny_line = "Oops! Something went wrong... Even pandas have bad days 🐼💥"
        return "ERROR", funny_line


def main():
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            break

        # Check if the user asks for plot or visualization
        if 'plot' in user_input.lower() or 'visualize' in user_input.lower():
            # Modify the prompt to handle plot-specific requests
            wrapped_prompt = wrap_prompt(user_input)
            llm_code = call_llm(wrapped_prompt)
            
            if llm_code:  # If LLM responds
                result_type, result = run_code(llm_code)
                if result_type == "PLOT":
                    print(f"Plot request detected. No plot generated.")
                    print(result)
                else:
                    print(result)
            else:
                print("Something went wrong! Moving on.")
                continue
        else:
            # For textual output (like summaries or descriptions)
            wrapped_prompt = wrap_prompt(user_input)
            llm_code = call_llm(wrapped_prompt)
            
            if llm_code:  # If LLM responds
                result_type, result = run_code(llm_code)

                if result_type == "PLOT":
                    print(f"Plot saved to: {result}")
                else:
                    print(result)
            else:
                print("Something went wrong! Moving on.")
                continue

            print("--------------------------------------------")
    
    # Print chat history at the end to inspect the conversation
    # print("Chat History:")
    # for entry in chat_history:
    #     print(f"{entry['role']}: {entry['content']}")

if __name__ == "__main__":
    main()