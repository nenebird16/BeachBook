import os
import logging
import config
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from storage.factory import StorageFactory
from llama_service import LlamaService
from services.semantic_processor import SemanticProcessor
from routes.journal_routes import journal_routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.register_blueprint(journal_routes)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.register_blueprint(journal_routes)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def init_services():
    """Initialize all required services"""
    services = {'graph_db': None}

    try:
        # Initialize graph database
        logger.info("Initializing graph database...")
        graph_db = StorageFactory.create_graph_database("neo4j")
        if graph_db:
            logger.info("Graph database initialized successfully")
            services['graph_db'] = graph_db
        else:
            logger.error("Failed to initialize graph database")

        # Initialize semantic processor
        logger.info("Initializing semantic processor...")
        services['semantic_processor'] = SemanticProcessor()

        # Initialize llama service with graph db
        logger.info("Initializing LlamaService...")
        services['llama_service'] = LlamaService(graph_db=graph_db)

    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")

    return services

# Initialize services
services = init_services()

# Make services available to the app context
app.config['graph_db'] = services.get('graph_db')
app.config['semantic_processor'] = services.get('semantic_processor')
app.config['llama_service'] = services.get('llama_service')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_knowledge():
    """Handle knowledge graph queries"""
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'response': 'Sorry, there was an error processing your request.'
            }), 400

        query = request.json.get('query')
        if not query:
            return jsonify({
                'error': 'No query provided',
                'response': 'Please provide a question to answer.'
            }), 400

        # Process the query using LlamaService
        llama_service = app.config.get('llama_service')
        if not llama_service:
            return jsonify({
                'error': 'Service unavailable',
                'response': 'The knowledge service is currently unavailable. Please try again later.'
            }), 503

        # Process the query
        result = llama_service.process_query(query)

        # Log the response structure for debugging
        logger.debug("Query result structure:")
        logger.debug(f"Response text: {result.get('response')}")
        logger.debug(f"Technical details: {result.get('technical_details')}")

        if not result:
            logger.error("LlamaService returned None response")
            return jsonify({
                'error': 'Service error',
                'response': 'Sorry, I encountered an error while processing your request. Please try again.'
            }), 500

        # Format response for frontend
        response = {
            'response': result.get('response', 'I apologize, but I was unable to generate a response.'),
            'technical_details': {
                'queries': result.get('technical_details', {}).get('queries', {})
            }
        }

        # Log the final response being sent to frontend
        logger.debug(f"Sending response to frontend: {response}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing query with LlamaService: {str(e)}")
        return jsonify({
            'error': 'Failed to process query',
            'response': 'I encountered an error while processing your question. Please try again.'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Get required services from app config
            graph_service = app.config.get('graph_db')
            llama_service = app.config.get('llama_service')
            semantic_processor = app.config.get('semantic_processor')

            if not all([graph_service, llama_service, semantic_processor]):
                logger.error("Required services not available")
                return jsonify({'error': 'Document processing service unavailable'}), 503

            # Initialize document processor with required services
            from services.document_processor import DocumentProcessor
            doc_processor = DocumentProcessor(
                graph_service=graph_service,
                llama_service=llama_service,
                semantic_processor=semantic_processor
            )

            # Process the document
            logger.info(f"Processing document: {file.filename}")
            result = doc_processor.process_document(file)
            logger.info(f"Document processing result: {result}")

            if result.get('error'):
                logger.error(f"Error processing document: {result['error']}")
                return jsonify({'error': result['error']}), 500

            return jsonify({
                'message': 'Document processed successfully',
                'doc_info': result
            }), 200

        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}", exc_info=True)
            return jsonify({'error': f'Document processing error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error handling document upload: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to process document'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)