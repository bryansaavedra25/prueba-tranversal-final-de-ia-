INFORME TÉCNICO: PROYECTO AGENTE INTELIGENTE "BANCO ANDINO"
Asignatura: Ingeniería de Soluciones con Inteligencia Artificial (ISY0101)
Nombre: Bryan Saavedra
Seccion:002D
1. Análisis del Caso Organizacional
Contexto y Desafíos
El proyecto se enmarca en el Banco Andino, una institución financiera que enfrenta la necesidad de modernizar sus canales de atención interna y externa. El desafío principal consiste en implementar un asistente inteligente capaz de resolver consultas sobre normativas y procedimientos operativos, garantizando el estricto cumplimiento de la regulación financiera chilena (CMF).
Los requerimientos críticos identificados son:
•	Privacidad de Datos: La solución debe detectar y filtrar información sensible (PII) antes de que sea procesada por modelos externos.
•	Fundamentación (Grounding): Las respuestas deben basarse exclusivamente en documentación oficial interna para evitar "alucinaciones".
•	Auditoría: Se requiere trazabilidad completa de las decisiones del agente y registro de evidencias operacionales.
2. Diseño de la Solución Basada en LLM y RAG
a. Formulación de Prompts
Se implementó una estrategia de Prompt Engineering utilizando plantillas dinámicas (PromptTemplate de LangChain). La estructura del prompt inyecta explícitamente el contexto recuperado y las reglas de negocio, instruyendo al modelo para actuar como un asistente bancario formal. Esto asegura que la respuesta generada se adhiera estrictamente a la información provista por el sistema de recuperación.




b. Implementación de Pipelines RAG
El flujo de Recuperación Aumentada (RAG) se diseñó con los siguientes componentes para enriquecer las respuestas:
1.	Ingesta y Fragmentación: Los documentos normativos se cargan y dividen mediante RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100) para mantener coherencia semántica.
2.	Embeddings: Se utiliza el modelo sentence-transformers/all-MiniLM-L6-v2 para generar representaciones vectoriales locales, optimizando la latencia y privacidad.
3.	Vector Store: FAISS actúa como motor de búsqueda de similitud, permitiendo recuperar los 4 documentos (top_k=4) más relevantes para cada consulta.
 








c. Diseño de Arquitectura
La arquitectura sigue un patrón modular orquestado por FastAPI. El núcleo es un Agente Planificador que coordina herramientas especializadas. El flujo no es lineal; incluye puntos de decisión lógica (compuertas) que determinan si se invoca al LLM o se deriva la consulta.
 


d. Justificación de Decisiones de Diseño
•	FAISS (Local): Se eligió FAISS sobre bases vectoriales en la nube para mantener la arquitectura on-premise en lo posible, crucial para un entorno bancario.
•	FastAPI: Seleccionado por su rendimiento asíncrono y capacidad de exponer métricas vía middleware (starlette_exporter), facilitando la observabilidad.
•	TaskPlanner Determinista: En lugar de dejar que el LLM decida libremente los pasos (tipo ReAct), se implementó un planificador determinista (Planner) para garantizar que la verificación de seguridad ocurra siempre antes de cualquier otra acción.
3. Desarrollo de Agente Funcional
a. Integración de Herramientas
El agente dispone de tres herramientas principales integradas en su bucle de ejecución:
1.	search_docs: Realiza la búsqueda semántica en la base de conocimiento.
2.	reason_policy: Evalúa reglas de negocio para decidir entre "responder" o "derivar" (ej. si el contexto es insuficiente).
3.	write_note: Permite al agente registrar notas operacionales y evidencias en formato JSONL (notas_operacionales.jsonl) para la continuidad del negocio.
b. Configuración de Memoria y Contexto
Se implementó un sistema de Memoria Híbrida:
•	Memoria de Corto Plazo: Clase ShortMemory que mantiene un buffer de los últimos 10 turnos de conversación en RAM, permitiendo al usuario hacer preguntas de seguimiento sin perder el hilo.
•	Memoria de Largo Plazo: Persistencia documental a través de FAISS, cargada al inicio (startup) de la aplicación.
c. Estrategias de Planificación y Toma de Decisiones
La clase TaskPlanner define un flujo de ejecución estricto: ["seguridad", "recuperar_ctx", "razonar", "responder", "registrar"]. Este diseño asegura que si el paso de "seguridad" detecta una amenaza, el plan se "cortocircuita", derivando inmediatamente al usuario y saltando la llamada al LLM, lo que ahorra costos y protege datos.
d. Documentación de la Orquestación
La orquestación se maneja en el endpoint /consultar. Cada solicitud genera un request_id único y un objeto planner_trace que documenta qué decisión tomó cada paso del planificador. Esto permite reconstruir exactamente por qué el agente actuó de cierta manera ante una consulta específica.
4. Implementación de Observabilidad, Trazabilidad y Seguridad
a. Métricas de Observabilidad
Se instrumentó la aplicación utilizando Prometheus Client. Las métricas clave expuestas en /metrics incluyen:
•	rag_requests_total: Conteo de solicitudes segregado por decisión (responder/derivar) y canal.
•	rag_request_latency_seconds: Histograma para medir latencia total y detectar cuellos de botella.
•	rag_llm_latency_seconds: Métrica específica para aislar el tiempo de respuesta del modelo de lenguaje.
b. Análisis de Registros (Trazabilidad)
El sistema genera logs estructurados en logs/ep3_logs.jsonl. A diferencia de un log de texto plano, cada línea es un objeto JSON que incluye:
•	Entrada del usuario y decisión tomada.
•	Latencias desglosadas por herramienta.
•	Consumo estimado de tokens. Esto facilita la auditoría automatizada y la visualización en el Dashboard de Streamlit implementado.
c. Protocolos de Seguridad
La seguridad se aborda mediante un enfoque de "Defensa en Profundidad":
1.	Filtrado de Palabras Clave: Una lista de términos sensibles (rut, clave, cvv) bloquea consultas peligrosas en la etapa de pre-procesamiento.
2.	Sanitización de Salida: Se aplica limpieza a la respuesta final para prevenir fugas accidentales.
3.	Gestión de Secretos: Uso de variables de entorno (python-dotenv) para credenciales, evitando hardcoding en el repositorio.

