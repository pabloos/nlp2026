from pathlib import Path

import gradio as gr
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RUNBOOKS_DIR = "runbooks"
MODEL        = "llama3.2:3b"
OLLAMA_URL   = "http://localhost:11434"
SYSTEM_PROMPT = (
    "Eres un agente de soporte técnico de primer nivel. "
    "Usa el contexto para dar pasos concretos y accionables. "
    "Nunca menciones que consultas documentación interna. "
    "Intenta resolver el problema directamente. "
    "Solo menciona escalar si los pasos no resuelven el problema. "
    "No menciones los runbooks ni sus fragmentos, solo da la respuesta al usuario. "
    "Responde siempre en español, sin preámbulos."
)

# ─── LLAMAINDEX SETUP ──────────────────────────────────────────────────────────
Settings.llm         = Ollama(model=MODEL, base_url=OLLAMA_URL,
                               request_timeout=120, temperature=0.3)
Settings.embed_model = FastEmbedEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def build_retriever():
    docs      = SimpleDirectoryReader(RUNBOOKS_DIR, required_exts=[".md"]).load_data()
    index     = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=3)
    return retriever, len(docs)


def build_ui(retriever, n_docs: int) -> gr.Blocks:

    def _text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return str(content) if content is not None else ""

    def _clean(history):
        return [{"role": t["role"], "content": _text(t["content"])} for t in history]

    def add_user_message(message: str, history: list[dict]):
        return _clean(history) + [{"role": "user", "content": message}], ""

    def generate_response(history: list[dict]):
        clean   = _clean(history)
        message = clean[-1]["content"]

        nodes   = retriever.retrieve(message)
        context = "\n\n---\n\n".join(n.text[:500] for n in nodes)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM,
                        content=f"{SYSTEM_PROMPT}\n\nCONTEXTO:\n{context}"),
            *[
                ChatMessage(
                    role=MessageRole.USER if t["role"] == "user" else MessageRole.ASSISTANT,
                    content=t["content"],
                )
                for t in clean[:-1]
            ],
            ChatMessage(role=MessageRole.USER, content=message),
        ]

        response = Settings.llm.chat(messages)

        sources = "\n\n".join(
            f"▸ [{node.metadata.get('file_name', '—')}]\n  {node.text[:120]}..."
            for node in nodes
        )
        return (
            clean + [{"role": "assistant", "content": response.message.content}],
            sources or "(sin contexto recuperado)",
        )

    with gr.Blocks(title="Soporte IT — RAG Chatbot (LlamaIndex)") as demo:
        gr.Markdown("## Soporte IT — RAG Chatbot")
        gr.Markdown(f"{n_docs} runbooks · LlamaIndex · Ollama `{MODEL}`")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460, label="Conversación")
                with gr.Row():
                    msg_box = gr.Textbox(placeholder="Describe tu problema técnico...",
                                         scale=5, container=False, autofocus=True)
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)
                gr.ClearButton([msg_box, chatbot], value="Limpiar conversación")

            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Fragmentos recuperados")
                retrieved_box = gr.Textbox(label="Último turno", lines=20, interactive=False)

        for trigger in (msg_box.submit, send_btn.click):
            (trigger(add_user_message, [msg_box, chatbot], [chatbot, msg_box])
               .then(generate_response, [chatbot], [chatbot, retrieved_box]))

    return demo

if __name__ == "__main__":
    print("Cargando runbooks y construyendo índice...")
    retriever, n_docs = build_retriever()
    print(f"Listo: {n_docs} runbooks indexados.")

    demo = build_ui(retriever, n_docs)
    print("Iniciando Gradio en http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
