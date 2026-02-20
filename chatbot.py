import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client (OpenAI-compatible)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    #base_url="https://api.groq.com/openai/v1"
)

app = Flask(__name__)

# Sales system prompt
SYSTEM_PROMPT = """
You are a professional sales assistant.
Your goal is to:
- Understand customer needs
- Recommend suitable products
- Highlight benefits
- Handle objections
- Encourage purchase decisions
Be persuasive but friendly.
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # You can change model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
