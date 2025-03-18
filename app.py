import os
import logging
import config  # Import config first to load environment variables

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log Neo4j configuration status
logger.debug(f"Initial NEO4J_URI: {'Present' if os.environ.get('NEO4J_URI') else 'Not present'}")
logger.debug(f"Config NEO4J_URI: {'Present' if config.NEO4J_URI else 'Not present'}")

# Remove NEO4J_URI from environment after config is loaded
original_uri = os.environ.pop('NEO4J_URI', None)

from flask import Flask, request, render_template, jsonify
from services.document_processor import DocumentProcessor
from services.graph_service import GraphService
from services.llama_service import LlamaService

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Initialize services
try:
    logger.info("Initializing Neo4j and LlamaIndex services...")
    graph_service = GraphService()
    llama_service = LlamaService()
    doc_processor = DocumentProcessor(graph_service, llama_service)
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process the document
        doc_info = doc_processor.process_document(file)
        return jsonify({'message': 'Document processed successfully', 'doc_info': doc_info})

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_knowledge():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Process query through RAG pipeline
        response = llama_service.process_query(query)
        return jsonify({
            'response': response['chat_response'],
            'technical_details': {
                'queries': response['queries'],
                'results': response['results']
            }
        })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/graph', methods=['GET'])
def get_graph():
    try:
        graph_data = graph_service.get_visualization_data()
        return jsonify(graph_data)

    except Exception as e:
        logger.error(f"Error fetching graph data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)