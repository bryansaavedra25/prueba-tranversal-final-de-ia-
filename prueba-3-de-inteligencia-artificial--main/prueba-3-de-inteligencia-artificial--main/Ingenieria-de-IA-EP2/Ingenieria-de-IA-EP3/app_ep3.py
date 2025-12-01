from __future__ import annotations
import os

# ==========================================
# 1. CONFIGURACIÓN DE CLAVES Y ENTORNO
# ==========================================
# Configuración para GitHub Models (OpenAI compatible)
os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"
os.environ["GITHUB_TOKEN"] = "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"
os.environ["OPENAI_API_KEY"] = os.environ["GITHUB_TOKEN"]

# Configuración de Trazabilidad (LangSmith)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "duoc_proyecto"

# ==========================================
# 2. IMPORTS
# ==========================================
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import logging
import uuid
import json
from datetime import datetime
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge
from starlette_exporter import PrometheusMiddleware, handle_metrics
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# -------------------- CONFIGURACIÓN --------------------
load_dotenv()

CARPETA_DOCUMENTOS = os.getenv("DOCS_DIR", "./docs")
RUTA_VECTORSTORE = os.getenv("VECTOR_DIR", "./vectorstore/faiss_index")
MODELO_EMBEDDING = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODELO_LLM = os.getenv("LLM_MODEL", "gpt-4o") 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
K_VECINOS = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

PALABRAS_SENSIBLES = [
    "clave", "password", "contraseña", "número de tarjeta", "cvv", "rut", "domicilio",
    "tarjeta de crédito", "nro de cuenta", "código de verificación"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("BancoAndinoRAG_EP3")

NOTES_FILE = os.getenv("NOTES_FILE", "./data/notas_operacionales.jsonl")
LOG_FILE = os.getenv("LOG_FILE", "./logs/ep3_logs.jsonl")

# -------------------- MODELOS API (Pydantic) --------------------
class SolicitudConsulta(BaseModel):
    pregunta: str
    cliente_id: Optional[str] = "anonimo"
    canal: Optional[str] = "web"

class DocumentoFuente(BaseModel):
    contenido_parcial: str
    puntuacion: float
    metadatos: Dict[str, Any]

class RespuestaConsulta(BaseModel):
    request_id: str
    respuesta: str
    documentos_fuente: List[DocumentoFuente]
    planner: Dict[str, Any]
    tooltrace: List[Dict[str, Any]]

class SolicitudNota(BaseModel):
    titulo: str
    contenido: str

# -------------------- UTILIDADES DE SEGURIDAD --------------------
def contiene_informacion_sensible(texto: str) -> bool:
    texto_lower = texto.lower()
    for palabra in PALABRAS_SENSIBLES:
        if palabra in texto_lower:
            return True
    return False

def sanitizar(texto: str) -> str:
    texto_limpio = texto
    for palabra in PALABRAS_SENSIBLES:
        texto_limpio = texto_limpio.replace(palabra, "[REDACTED]")
    return texto_limpio

# -------------------- DOCUMENTOS & VECTORSTORE --------------------
def cargar_vectorstore_guardado(ruta: str):
    logger.info(f"Cargando VectorStore desde {ruta}...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
    try:
        return FAISS.load_local(ruta, embeddings, allow_dangerous_deserialization=True)
    except:
        return FAISS.load_local(ruta, embeddings)

def construir_y_guardar_vectorstore(carpeta_docs: str, ruta_destino: str):
    logger.info(f"Construyendo VectorStore desde {carpeta_docs}...")
    
    if not os.path.exists(carpeta_docs):
        os.makedirs(carpeta_docs)
        with open(os.path.join(carpeta_docs, "info_banco.txt"), "w", encoding="utf-8") as f:
            f.write("El Banco Andino ofrece cuentas corrientes y créditos de consumo. Horario: 9am a 14pm.")

    docs = []
    if os.path.exists(carpeta_docs):
        for archivo in os.listdir(carpeta_docs):
            if archivo.endswith(".txt") or archivo.endswith(".md"):
                path = os.path.join(carpeta_docs, archivo)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            
    if not docs:
        docs = [Document(page_content="Información genérica del Banco Andino.", metadata={"source": "dummy"})]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(ruta_destino)
    logger.info("VectorStore guardado exitosamente.")
    return vectorstore

def construir_chain_rag(vectorstore):
    llm = ChatOpenAI(model_name=MODELO_LLM, temperature=TEMPERATURE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_VECINOS})
    
    template = """Usa los siguientes fragmentos de contexto para responder a la pregunta al final. 
    Si no sabes la respuesta, di que no lo sabes.
    
    Contexto: {context}
    
    Pregunta: {question}
    Respuesta Útil:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def plantilla_para(pregunta: str) -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
        template="Responde como experto del Banco Andino: {question}"
    )

# -------------------- HERRAMIENTAS (Tools) --------------------
def tool_search_docs(query: str, retriever, k: int = K_VECINOS) -> Tuple[str, List[Any]]:
    # CORRECCIÓN AQUÍ: Usamos .invoke() en lugar de .get_relevant_documents()
    results = retriever.invoke(query)
    top_docs = results[:k]
    context = "\n\n".join([d.page_content for d in top_docs])
    return context, top_docs

if "/" in NOTES_FILE:
    os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)

def tool_write_note(titulo: str, contenido: str) -> Dict[str, Any]:
    note = {
        "id": str(uuid.uuid4()),
        "titulo": titulo,
        "contenido": contenido,
        "ts": datetime.utcnow().isoformat()
    }
    with open(NOTES_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\n")
    return note

def tool_reason_policy(pregunta: str, ctx_found: bool) -> str:
    if not ctx_found:
        return "derivar"
    return "responder"

# -------------------- MEMORIA & PLANNER --------------------
class ShortMemory:
    def __init__(self):
        self.buffer = []
    
    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content, "ts": time.time()})
        if len(self.buffer) > 10:
            self.buffer.pop(0)
            
    def as_text(self) -> str:
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.buffer])

SHORT_MEMORY = ShortMemory()

class TaskPlanner:
    def plan(self, query: str) -> List[str]:
        return ["seguridad", "recuperar_ctx", "razonar", "responder", "registrar"]

PLANNER = TaskPlanner()

# Variables Globales
VECTORSTORE_GLOBAL: Optional[FAISS] = None
CHAIN_RAG_GLOBAL: Optional[RetrievalQA] = None

# -------------------- PROMETHEUS METRICS --------------------
rag_requests_total = Counter(
    "rag_requests_total",
    "Total de consultas procesadas por el agente RAG",
    ["decision", "canal", "sensible"]
)

rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "Latencia total del endpoint /consultar (segundos)",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds",
    "Latencia del llamado al RAG/LLM (segundos)",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

system_cpu_percent = Gauge("system_cpu_percent", "Uso actual de CPU del proceso del agente")
system_memory_percent = Gauge("system_memory_percent", "Uso actual de RAM del proceso del agente")
PROCESS = psutil.Process(os.getpid())

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="Banco Andino - Agente RAG (EP2/EP3)")

app.add_middleware(PrometheusMiddleware, app_name="banco_andino_rag")
app.add_route("/metrics", handle_metrics)

@app.on_event("startup")
async def init():
    global VECTORSTORE_GLOBAL, CHAIN_RAG_GLOBAL
    try:
        if "/" in LOG_FILE: os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        if os.path.exists(RUTA_VECTORSTORE):
            VECTORSTORE_GLOBAL = cargar_vectorstore_guardado(RUTA_VECTORSTORE)
        else:
            VECTORSTORE_GLOBAL = construir_y_guardar_vectorstore(CARPETA_DOCUMENTOS, RUTA_VECTORSTORE)
            
        CHAIN_RAG_GLOBAL = construir_chain_rag(VECTORSTORE_GLOBAL)
        logger.info(f"EP3: Logs en {LOG_FILE}")
        logger.info("EP3: Sistema inicializado.")
    except Exception as e:
        logger.exception(f"EP3: Error inicializando: {e}")

@app.get("/salud")
async def salud():
    listo = VECTORSTORE_GLOBAL is not None and CHAIN_RAG_GLOBAL is not None
    return {"status": "ok" if listo else "inicializando"}

@app.post("/consultar", response_model=RespuestaConsulta)
async def consultar(solicitud: SolicitudConsulta):
    global CHAIN_RAG_GLOBAL, VECTORSTORE_GLOBAL
    start_time = time.time()
    is_sensible = "false"
    tokens_consumed = 0

    if CHAIN_RAG_GLOBAL is None or VECTORSTORE_GLOBAL is None:
        raise HTTPException(status_code=503, detail="Inicialización en curso")

    req_id = str(uuid.uuid4())
    pregunta = solicitud.pregunta.strip()
    tooltrace: List[Dict[str, Any]] = []
    planner_trace: Dict[str, Any] = {"plan": PLANNER.plan(pregunta), "decisiones": []}

    # 1. Seguridad
    if contiene_informacion_sensible(pregunta):
        is_sensible = "true"
        decision = "derivar"
        planner_trace["decisiones"].append({"paso": "seguridad", "accion": "derivar_por_datos_sensibles"})
        respuesta = "⚠️ Consulta contiene datos sensibles. Derivando a ejecutivo."
        SHORT_MEMORY.add("user", pregunta)
        SHORT_MEMORY.add("assistant", respuesta)

        end_time = time.time()
        latency_total = end_time - start_time
        rag_requests_total.labels(decision=decision, canal=solicitud.canal or "web", sensible=is_sensible).inc()
        rag_request_latency_seconds.observe(latency_total)

        traza = {
            "ts": datetime.utcnow().isoformat(), "request_id": req_id, "cliente_id": solicitud.cliente_id,
            "canal": solicitud.canal, "pregunta": pregunta, "decision": decision,
            "latencia_total": latency_total, "planner": planner_trace, "tooltrace": tooltrace,
            "short_memory_tail": SHORT_MEMORY.as_text()[-500:], "tokens": 0
        }
        with open(LOG_FILE, "a", encoding="utf8") as f: f.write(json.dumps(traza, ensure_ascii=False) + "\n")
        
        return RespuestaConsulta(request_id=req_id, respuesta=respuesta, documentos_fuente=[], planner=planner_trace, tooltrace=tooltrace)
    
    decision = "responder"

    # 2. Contexto
    retriever = VECTORSTORE_GLOBAL.as_retriever(search_type="similarity", search_kwargs={"k": K_VECINOS})
    contexto, docs = tool_search_docs(pregunta, retriever, k=K_VECINOS)
    ctx_found = len(docs) > 0 and len(contexto.strip()) > 0
    tooltrace.append({"tool": "search_docs", "k": K_VECINOS, "ctx_found": ctx_found})

    # 3. Razonar
    decision = tool_reason_policy(pregunta, ctx_found)
    planner_trace["decisiones"].append({"paso": "razonar", "decision": decision, "ctx_found": ctx_found})

    # 4. Generar
    documentos_fuente = []
    respuesta_texto = ""

    if decision == "responder":
        try:
            llm_start_time = time.time()
            # CORRECCIÓN AQUÍ: Usamos .invoke() en la cadena también para seguridad
            resultado = CHAIN_RAG_GLOBAL.invoke({"query": pregunta})
            rag_llm_latency_seconds.observe(time.time() - llm_start_time)
            
            salida = resultado.get("result", "")
            respuesta_texto = sanitizar(salida)
            
            for d in resultado.get("source_documents", []):
                score = float(d.metadata.get("score", 0.0))
                documentos_fuente.append(DocumentoFuente(contenido_parcial=d.page_content[:600], puntuacion=score, metadatos=d.metadata))
            
            tooltrace.append({"tool": "llm_rag", "ok": True})
            tokens_consumed = len(pregunta.split()) + len(respuesta_texto.split()) + len(contexto.split())
        except Exception as e:
            logger.exception(f"Error RAG: {e}")
            respuesta_texto = "Error al procesar. Derivando a ejecutivo."
            decision = "derivar"
            tooltrace.append({"tool": "llm_rag", "ok": False, "error": str(e)})

    if decision == "derivar" and not respuesta_texto:
        respuesta_texto = "Derivaremos este caso a un ejecutivo."

    # 5. Registrar Nota
    try:
        tool_write_note("Consulta cliente", f"req={req_id} decision={decision}")
        tooltrace.append({"tool": "write_note", "ok": True})
    except Exception as e:
        tooltrace.append({"tool": "write_note", "ok": False, "error": str(e)})

    SHORT_MEMORY.add("user", pregunta)
    SHORT_MEMORY.add("assistant", respuesta_texto)

    # Métricas Finales
    latency_total = time.time() - start_time
    rag_requests_total.labels(decision=decision, canal=solicitud.canal or "web", sensible=is_sensible).inc()
    rag_request_latency_seconds.observe(latency_total)
    system_cpu_percent.set(PROCESS.cpu_percent(interval=None))
    system_memory_percent.set(PROCESS.memory_percent())

    traza = {
        "ts": datetime.utcnow().isoformat(), "request_id": req_id, "pregunta": pregunta,
        "decision": decision, "latencia_total": latency_total, "tokens": tokens_consumed,
        "planner": planner_trace, "tooltrace": tooltrace
    }
    with open(LOG_FILE, "a", encoding="utf8") as f: f.write(json.dumps(traza, ensure_ascii=False) + "\n")

    return RespuestaConsulta(request_id=req_id, respuesta=respuesta_texto, documentos_fuente=documentos_fuente, planner=planner_trace, tooltrace=tooltrace)

@app.post("/nota")
async def crear_nota(nota: SolicitudNota):
    return {"ok": True, "note": tool_write_note(nota.titulo, nota.contenido)}

@app.get("/memoria/corto")
async def memoria_corto():
    return {"turns": SHORT_MEMORY.buffer}