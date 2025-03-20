import os
import logging
import config
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from storage.factory import StorageFactory
from services.semantic_processor import SemanticProcessor
from services.document_processor import DocumentProcessor
from services.graph_service import GraphService
from services.llama_service import LlamaService
from routes.journal_routes import journal_routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.register_blueprint(journal_routes)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

def verify_environment():
    """Verify required environment variables are present"""
    required_vars = {
        'NEO4J_URI': os.environ.get('NEO4J_URI'),
        'NEO4J_USER': os.environ.get('NEO4J_USER'),
        'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD'),
    }

    missing = [var for var, value in required_vars.items() if not value]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False

    logger.info("All required environment variables are present")
    return True

def init_services():
    """Initialize all required services"""
    services = {}

    # Verify environment variables first
    if not verify_environment():
        logger.warning("Missing environment variables - services will be initialized in degraded state")

    try:
        # Initialize semantic processor
        logger.info("Initializing semantic processor...")
        semantic_processor = SemanticProcessor()
        services['semantic_processor'] = semantic_processor
        logger.info("Semantic processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize semantic processor: {str(e)}", exc_info=True)

    try:
        # Initialize graph service
        logger.info("Initializing graph service...")
        graph_service = GraphService()
        services['graph_db'] = graph_service
        logger.info("Graph service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize graph service: {str(e)}", exc_info=True)

    try:
        # Initialize document processor if dependencies are available
        if services.get('graph_db') and services.get('semantic_processor'):
            logger.info("Initializing document processor...")
            doc_processor = DocumentProcessor(
                graph_service=services['graph_db'],
                semantic_processor=services['semantic_processor']
            )
            services['document_processor'] = doc_processor
            logger.info("Document processor initialized successfully")
        else:
            logger.warning("Skipping document processor initialization - required services unavailable")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}", exc_info=True)

    try:
        # Initialize LlamaService if dependencies are available
        if services.get('semantic_processor'):
            logger.info("Initializing LlamaService...")
            llama_service = LlamaService()
            services['llama_service'] = llama_service
            logger.info("LlamaService initialized successfully")
        else:
            logger.warning("Skipping LlamaService initialization - required services unavailable")
    except Exception as e:
        logger.error(f"Failed to initialize LlamaService: {str(e)}", exc_info=True)

    return services

# Initialize services
try:
    services = init_services()
    # Make services available to the app context
    app.config['graph_db'] = services.get('graph_db')
    app.config['semantic_processor'] = services.get('semantic_processor')
    app.config['document_processor'] = services.get('document_processor')
    app.config['llama_service'] = services.get('llama_service')
except Exception as e:
    logger.error(f"Application startup failed: {str(e)}", exc_info=True)
    # Allow the app to start in a degraded state
    services = {}

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    try:
        # Check if required services are initialized
        services_status = {
            'graph_db': app.config.get('graph_db') is not None,
            'semantic_processor': app.config.get('semantic_processor') is not None,
            'document_processor': app.config.get('document_processor') is not None,
            'llama_service': app.config.get('llama_service') is not None
        }

        # Verify Neo4j environment variables
        env_vars_present = verify_environment()

        return jsonify({
            'status': 'healthy' if all(services_status.values()) and env_vars_present else 'degraded',
            'services': services_status,
            'environment': env_vars_present
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/')
def index():
    """Render the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return "Service temporarily unavailable", 503

@app.route('/query', methods=['POST'])
def query_knowledge():
    """Handle knowledge graph queries"""
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'response': 'Sorry, there was an error processing your request.'
            }), 400

        query = request.get_json().get('query')
        if not query:
            return jsonify({
                'error': 'No query provided',
                'response': 'Please provide a question to answer.'
            }), 400

        # Check if LlamaService is available
        llama_service = app.config.get('llama_service')
        if not llama_service:
            logger.error("LlamaService not initialized")
            return jsonify({
                'error': 'Service unavailable',
                'response': 'The knowledge service is currently unavailable. Please check the /health endpoint for service status.'
            }), 503

        # Log query details
        logger.info(f"Processing query: {query}")
        logger.debug(f"Current service statuses: Graph DB: {bool(app.config.get('graph_db'))}, "
                    f"Semantic Processor: {bool(app.config.get('semantic_processor'))}")

        # Process the query
        try:
            result = llama_service.process_query(query)
            if not result:
                logger.error("Empty response from LlamaService")
                raise ValueError("Empty response from LlamaService")

            logger.debug(f"Query result: {result}")

            # Format response for frontend
            response = {
                'response': result.get('response', 'I apologize, but I was unable to generate a response.'),
                'technical_details': {
                    'queries': result.get('technical_details', {}).get('queries', {})
                }
            }

            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Error processing query with LlamaService: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Service error',
                'response': 'Sorry, I encountered an error while processing your request. Please try again.'
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'response': 'An unexpected error occurred. Please try again later.'
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
            # Get document processor service
            doc_processor = app.config.get('document_processor')
            if not doc_processor:
                logger.error("Document processing service unavailable")
                return jsonify({'error': 'Document processing service unavailable'}), 503

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