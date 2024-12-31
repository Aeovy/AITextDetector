from flask import Flask, request, render_template
from Use_model_Functions import load_model, encoding,tokenizer
from Use_Model_By_Input import predict


app = Flask(__name__)

model = load_model('robertaLargeBiLSTMTextCNN')
tokenizer = tokenizer

@app.route('/', methods=['GET', 'POST'])
def index():
    text = ""
    result = None
    if request.method == 'POST':
        text = request.form['input_text']
        if text:
            result = predict(text, model, tokenizer)
    return render_template('index.html', input_text=text, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)