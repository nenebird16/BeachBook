import logging
from app import app

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# This is my comment - ThirstyPiglet

if __name__ == "__main__":
    logger.info("Starting Knowledge Graph RAG System")
    logger.debug("Checking initialized services:")
    logger.debug(f"Graph DB initialized: {'Yes' if app.config.get('graph_db') else 'No'}")
    logger.debug(f"Semantic Processor initialized: {'Yes' if app.config.get('semantic_processor') else 'No'}")
    logger.debug(f"Document Processor initialized: {'Yes' if app.config.get('document_processor') else 'No'}")
    logger.debug(f"LlamaService initialized: {'Yes' if app.config.get('llama_service') else 'No'}")

    app.run(host="0.0.0.0", port=5000, debug=True)