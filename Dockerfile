FROM python:3.11.10-slim
RUN mkdir /code
ADD ./code ./code
COPY Dockerfile .
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')"
WORKDIR /code
EXPOSE 5000
CMD ["python", "app.py"]