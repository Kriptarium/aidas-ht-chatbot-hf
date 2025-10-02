
# AIDAS-HT Chatbot (HF açık model, anahtarsız)

**Hugging Face**'ten **google/flan-t5-small** modelini sunucu içinde yükeleyip çalıştırır.
- API anahtarı gerekmez.
- Render/Python Web Service ile deploy edilir (CPU).
- İsterseniz RAG ile PDF bağlamı eklenebilir.

## Lokal
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

## Render
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Not
İlk istek model indirme/yükleme sebebiyle biraz zaman alabilir.
