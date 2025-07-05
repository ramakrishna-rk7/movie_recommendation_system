
from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json.get('review')
    inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({"prediction": "real" if prediction == 1 else "fake"})

if __name__ == '__main__':
    app.run(debug=True)
