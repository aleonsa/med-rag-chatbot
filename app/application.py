from markupsafe import Markup
from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os
import traceback
import logging

# Configurar logging más detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app = Flask(__name__)
app.secret_key = os.urandom(24)

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []
    
    if request.method == "POST":
        user_input = request.form.get("prompt")
        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages
            
            try:
                logger.info(f"Processing user input: {user_input}")
                
                # Validar API key antes de proceder
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY no está configurada")
                
                qa_chain = create_qa_chain()
                logger.info("QA chain created successfully")
                
                response = qa_chain.invoke({"query": user_input})
                logger.info(f"Response received: {response}")
                
                result = response.get("result", "No response")
                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages
                logger.info("Response processed successfully")
                
            except Exception as e:
                # Logging más detallado del error
                logger.error(f"Exception occurred: {type(e).__name__}")
                logger.error(f"Exception message: '{str(e)}'")
                logger.error(f"Exception args: {e.args}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # Crear mensaje de error más informativo
                error_type = type(e).__name__
                error_message = str(e) if str(e) else "Sin mensaje específico"
                error_args = str(e.args) if e.args else "Sin argumentos"
                
                # Si el error está completamente vacío, usar información del traceback
                if not error_message or error_message.strip() == "":
                    error_message = f"Error de tipo {error_type} sin mensaje específico"
                
                error_msg = f"Error ({error_type}): {error_message}"
                
                # También agregar a los logs para debugging
                logger.error(f"Error message sent to template: {error_msg}")
                
                return render_template("index.html", 
                                     messages=session["messages"], 
                                     error=error_msg)
            
            return redirect(url_for("index"))
    
    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear", methods=["GET"])  # Cambiado a GET para que funcione con tu HTML
def clear():
    session.pop("messages", None)
    logger.info("Chat cleared")
    return redirect(url_for("index"))

# Ruta adicional para debugging - puedes acceder a /debug para ver información del sistema
@app.route("/debug")
def debug_info():
    debug_data = {
        "session_messages": len(session.get("messages", [])),
        "openai_key_set": bool(OPENAI_API_KEY),
        "python_version": os.sys.version,
        "flask_debug": app.debug
    }
    return f"<pre>{debug_data}</pre>"

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)  # Activé debug=True