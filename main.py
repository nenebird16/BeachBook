import logging
from app import app

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Knowledge Graph RAG System")
    app.run(host="0.0.0.0", port=5000, debug=True)
