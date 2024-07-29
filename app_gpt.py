from flask import Flask, render_template, request, jsonify
from transformers import pipeline, set_seed

app = Flask(__name__)

# Load the text generation pipeline
generator = pipeline('text-generation', model='gpt2')
set_seed(42)  # Ensure reproducibility


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    topic = data.get('topic')

    if not topic:
        return jsonify({'error': 'Topic is required'}), 400

    try:
        response = generator(topic, max_length=500, num_return_sequences=1)
        article = response[0]['generated_text'].strip()
        return jsonify({'article': article})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
