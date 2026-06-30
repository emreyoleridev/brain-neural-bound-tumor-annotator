# Python 3.10 baz al (Stabil sürüm)
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli sistem kütüphanelerini kur (OpenCV için gerekli)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje kodlarını kopyala
COPY . .

# Streamlit portunu aç
EXPOSE 8501

# Uygulamayı başlat
CMD ["streamlit", "run", "streamlit_app/app.py"]