import os
import io
import json
import numpy as np
import cv2
import whisper
from flask import render_template, Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from google.cloud.vision_v1 import types
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from PIL import Image

# Ortam değişkenlerini yükle
load_dotenv()

# OpenAI API ve Google Vision API Anahtarı
api_key = os.getenv("OPENAI_API_KEY")
vision_client = vision.ImageAnnotatorClient()
if not api_key:
    raise ValueError("HATA: OPENAI_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")

# Flask Uygulaması Başlat
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Whisper Modelini Yükle
whisper_model = whisper.load_model("base")

# Son yüklenen resim analiz sonucunu saklamak için global değişken
last_image_analysis = ""

def transcribe_audio(audio_file):
    """Whisper kullanarak sesi metne çevirir."""
    try:
        temp_path = "temp_audio.wav"
        audio_file.save(temp_path)
        result = whisper_model.transcribe(temp_path, language="tr")
        os.remove(temp_path)
        return result["text"]
    except Exception as e:
        return f"Ses işleme hatası: {str(e)}"

# GPT-4o Modelini Kullan
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    openai_api_key=api_key,
    temperature=0.7  # Daha yaratıcı ve doğal yanıtlar için sıcaklığı artırdım
)

# acne22.json Veritabanını Yükle (Opsiyonel olarak kullanılacak)
documents = []
try:
    with open('acne22.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            documents = [Document(page_content=item.get("content", ""), metadata={"source": "acne22.json"}) for item in data]
        else:
            raise ValueError("JSON formatı hatalı!")
except FileNotFoundError:
    raise FileNotFoundError("HATA: acne22.json dosyası bulunamadı!")
except json.JSONDecodeError:
    raise ValueError("HATA: JSON dosyası bozuk veya hatalı formatta!")

# FAISS Vektör Veritabanı (Cilt bakımı için özel bilgi)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_db = FAISS.from_documents(documents, embeddings)

# Güncellenmiş Prompt Şablonu
prompt_template = """
Siz, hem akne, siyah nokta, cilt kızarıklığı, morarma, sivilce, leke, cilt ürünleri ve genel cilt bakımı konusunda uzman bir asistan, hem de günlük hayattan sorulara doğal ve samimi bir şekilde yanıt verebilen bir arkadaşsınız. Kullanıcının sorusuna, aşağıdaki resim analiz sonuçlarını da dikkate alarak detaylı, doğru ve kapsamlı cevap verin. Eğer bir resim yüklendiyse ve soru resimle ilgiliyse, analiz sonuçlarını kullanarak öneriler sunun veya soruyu yanıtlayın. Genel günlük hayat sorularında ise doğal, arkadaşça ve doğaçlama yanıtlar verin (örneğin, "Merhaba, nasılsın?" gibi bir soruya "İyiyim, teşekkür ederim! Sen nasılsın?" gibi cevap verebilirsiniz).

Resim Analizi Bağlamı: {context}

Soru: {question}
Cevap:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),  # Daha iyi sonuçlar için k artırıldı
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False  # Gereksiz belgeleri döndürmemek için
)

def analyze_image(image: Image.Image) -> str:
    """Google Vision API kullanarak detaylı resim analizi yapar."""
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        vision_image = types.Image(content=image_bytes)
        response = vision_client.annotate_image({'image': vision_image, 'features': [{'type': vision.Feature.Type.LABEL_DETECTION},
                                                                                      {'type': vision.Feature.Type.FACE_DETECTION},
                                                                                      {'type': vision.Feature.Type.OBJECT_LOCALIZATION}]})
        
        results = []
        
        if response.label_annotations:
            for label in response.label_annotations:
                results.append(label.description)
        
        if response.face_annotations:
            for face in response.face_annotations:
                if face.joy_likelihood >= 3:
                    results.append("Mutlu bir yüz var")
                elif face.sorrow_likelihood >= 3:
                    results.append("Üzgün bir yüz var")
                elif face.anger_likelihood >= 3:
                    results.append("Sinirli bir yüz var")
                elif face.surprise_likelihood >= 3:
                    results.append("Şaşkın bir yüz var")
        
        return "Resimde görülenler: " + ", ".join(results)
    except Exception as e:
        return f"Resim analizi hatası: {str(e)}"

# Flask API Endpointleri
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data_json = request.get_json()
        user_input = data_json.get("message", "")
        if not user_input:
            return jsonify({"response": "Lütfen mesajınızı girin."})
        # Resim analizi bağlamını ekleyerek soruyu işliyoruz
        response = qa_chain.invoke({"query": user_input, "context": last_image_analysis})["result"]
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/voice', methods=['POST'])
def voice():
    if 'audio' not in request.files:
        return jsonify({"message": "Ses dosyası bulunamadı."}), 400
    try:
        audio_file = request.files['audio']
        text = transcribe_audio(audio_file)
        return jsonify({"message": text})
    except Exception as e:
        return jsonify({"message": f"Ses işleme hatası: {str(e)}"}), 500

@app.route('/image-analysis', methods=['POST'])
def image_analysis():
    global last_image_analysis
    if 'image' not in request.files:
        return jsonify({"analysis": "Resim dosyası bulunamadı."}), 400
    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        analysis_result = analyze_image(image)
        last_image_analysis = analysis_result  # Analiz sonucunu global olarak sakla
        return jsonify({"analysis": analysis_result})
    except Exception as e:
        return jsonify({"analysis": f"Resim analizi hatası: {str(e)}"}), 500

if __name__ == '__main__':
    print("Uygulama başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=5000)