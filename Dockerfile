FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model ./model
COPY assets/news_favicon.png ./assets/news_favicon.png
COPY website.py .
EXPOSE 8501
CMD ["streamlit", "run", "website.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
