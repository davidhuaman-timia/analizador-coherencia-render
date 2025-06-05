from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import os

# Inicializa la aplicación
app = FastAPI()

# Configura CORS para permitir comunicación con el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el modelo (se carga una vez)
modelo = None

def get_modelo():
    global modelo
    if modelo is None:
        modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return modelo

# Modelo de datos que espera el endpoint
class TextoRequest(BaseModel):
    texto: str

def dividir_oraciones(texto: str) -> list:
    oraciones = re.split(r'(?<=[.!?]) +|\n', texto.strip())
    return [s for s in oraciones if s]

def obtener_embeddings(oraciones: list) -> list:
    modelo = get_modelo()
    embeddings = modelo.encode(oraciones)
    return embeddings

def calcular_similaridad(embedding1, embedding2) -> float:
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    similitud = cosine_similarity(emb1, emb2)[0][0]
    return similitud

def interpretar_coherencia(score: float) -> str:
    if score >= 0.8:
        return "Muy coherente"
    elif score >= 0.6:
        return "Coherente"
    elif score >= 0.4:
        return "Moderadamente coherente"
    elif score >= 0.2:
        return "Poco coherente"
    else:
        return "Muy poco coherente"

def analizar_coherencia(oraciones: list, embeddings: list, umbral: float = 0.3) -> dict:
    puntajes = []
    oraciones_no_coherentes = []

    for i in range(len(oraciones) - 1):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]
        similitud = calcular_similaridad(emb1, emb2)
        puntajes.append(similitud)

        if similitud < umbral:
            oraciones_no_coherentes.append({
                "oracion_1": oraciones[i],
                "oracion_2": oraciones[i + 1],
                "coherencia": similitud
            })
    
    coherencia_consecutiva_promedio = float(sum(puntajes) / len(puntajes)) if puntajes else 0
    matriz_similitudes = cosine_similarity(embeddings)
    
    mascara = np.ones(matriz_similitudes.shape, dtype=bool)
    np.fill_diagonal(mascara, False)
    similitud_general_promedio = matriz_similitudes[mascara].mean()
    
    coherencia_global = similitud_general_promedio
    coherencia_local = coherencia_consecutiva_promedio
    coherencia_final = (coherencia_local * 0.6) + (coherencia_global * 0.4)
    intepretacion = interpretar_coherencia(coherencia_final)
    
    return {
        "coherencia_promedio": coherencia_final,
        "oraciones_no_coherentes": oraciones_no_coherentes,
        "intepretacion": intepretacion
    }

@app.get("/")
def read_root():
    return {"message": "API de Análisis de Coherencia Textual", "status": "funcionando"}

@app.post("/analizar")
async def analizar_texto(request: TextoRequest):
    texto = request.texto
    oraciones = dividir_oraciones(texto)
    embeddings = obtener_embeddings(oraciones)
    resultado = analizar_coherencia(oraciones, embeddings, umbral=0.3)

    return {
        "coherencia_promedio": float(resultado["coherencia_promedio"]),
        "intepretacion": (resultado["intepretacion"]),
        "oraciones_no_coherentes": [
            (x["oracion_1"], x["oracion_2"], float(x["coherencia"]))
            for x in resultado["oraciones_no_coherentes"]
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
