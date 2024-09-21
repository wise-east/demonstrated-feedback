from flask import Flask, render_template, request
import json

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jsonl_data = request.form["jsonl_data"]
        parsed_data = []
        for line in jsonl_data.split("\n"):
            if line.strip():
                try:
                    parsed_data.append(json.loads(line))
                except json.JSONDecodeError:
                    parsed_data.append({"error": "Invalid JSON"})
        return render_template("viewer.html", data=parsed_data)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5005)
