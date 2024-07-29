import transformers
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize the Meta-Llama-3.1-8B-Instruct model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    user_message = data.get('user_message')

    # Ensure user_message is provided
    if not user_message:
        return jsonify({"error": "user_message is required"}), 400

    messages = [
        {"role": "system", "content": "You are professional writer who"
                                      "writes most astonished poems. "
                                      "Poems must be written in simple terms "
                                      "and words but at the same time splendid"
                                      "Write a short poem that will follow the structure:"
                                      "introduction, culmination, ending."
        },
        {"role": "user", "content": user_message},
    ]

    try:
        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        generated_text = outputs[0]["generated_text"][-1]
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
