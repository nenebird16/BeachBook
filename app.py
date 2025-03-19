import os
import logging
import config
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from routes.journal_routes import journal_routes
from storage.factory import StorageFactory
from llama_service import LlamaService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def verify_env_variables():
    """Verify all required environment variables are set"""
    required_vars = {
        'NEO4J_URI': os.environ.get('NEO4J_URI'),
        'NEO4J_USER': os.environ.get('NEO4J_USER'),
        'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD')
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False

    # Log the presence of variables (not their values)
    for var_name in required_vars:
        logger.info(f"{var_name} is present and not empty")

    # For Neo4j URI, validate format
    neo4j_uri = required_vars['NEO4J_URI']
    if not neo4j_uri.startswith(('neo4j://', 'neo4j+s://', 'bolt://', 'bolt+s://')):
        logger.error(f"Invalid Neo4j URI format. URI must start with neo4j://, neo4j+s://, bolt://, or bolt+s://")
        return False

    logger.info("All required environment variables are present and valid")
    return True

def init_storage_services():
    """Initialize storage services with proper error handling"""
    services = {'graph_db': None, 'object_storage': None}

    if not verify_env_variables():
        logger.error("Cannot initialize storage services due to missing environment variables")
        return services

    try:
        # Initialize graph database
        try:
            logger.info("Initializing graph database...")
            graph_db = StorageFactory.create_graph_database("neo4j")
            if graph_db:
                logger.info("Graph database initialized successfully")
                services['graph_db'] = graph_db
            else:
                logger.error("Failed to initialize graph database")
        except Exception as e:
            logger.error(f"Error initializing graph database: {str(e)}")

        # Initialize object storage
        try:
            logger.info("Initializing object storage...")
            object_storage = StorageFactory.create_object_storage("replit")
            if object_storage:
                logger.info("Object storage initialized successfully")
                services['object_storage'] = object_storage
            else:
                logger.error("Failed to initialize object storage")
        except Exception as e:
            logger.error(f"Error initializing object storage: {str(e)}")

    except Exception as e:
        logger.error(f"Failed to initialize storage services: {str(e)}")

    return services

# Initialize storage services
services = init_storage_services()

# Make storage services available to the app context
app.config['graph_db'] = services.get('graph_db')
app.config['object_storage'] = services.get('object_storage')

if not any(services.values()):
    logger.warning("No storage services were initialized successfully")
else:
    logger.info("Some storage services initialized successfully")

# Health check endpoint
@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check each storage service
        storage_status = {
            'graph_db': {
                'available': bool(app.config.get('graph_db')),
                'connected': False,
                'error': None
            },
            'object_storage': {
                'available': bool(app.config.get('object_storage')),
                'connected': False,
                'error': None
            }
        }

        # Check environment variables
        env_vars = {
            'NEO4J_URI': bool(os.environ.get('NEO4J_URI')),
            'NEO4J_USER': bool(os.environ.get('NEO4J_USER')),
            'NEO4J_PASSWORD': bool(os.environ.get('NEO4J_PASSWORD'))
        }

        # Test graph database connection if available
        if storage_status['graph_db']['available']:
            try:
                app.config['graph_db'].query("RETURN 1 as test")
                storage_status['graph_db']['connected'] = True
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Graph database connection test failed: {error_msg}")
                storage_status['graph_db']['error'] = error_msg

        # Test object storage connection if available
        if storage_status['object_storage']['available']:
            try:
                app.config['object_storage'].list_files()
                storage_status['object_storage']['connected'] = True
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Object storage connection test failed: {error_msg}")
                storage_status['object_storage']['error'] = error_msg

        # Determine overall health status
        is_healthy = any(status['connected'] for status in storage_status.values())

        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'environment': env_vars,
            'storage': storage_status,
            'uri_format': os.environ.get('NEO4J_URI', '').startswith(('neo4j://', 'neo4j+s://', 'bolt://', 'bolt+s://'))
        }), 200 if is_healthy else 503

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Register blueprints
app.register_blueprint(journal_routes)

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
                'chat_response': 'Sorry, there was an error processing your request.'
            }), 400

        query = request.json.get('query')
        if not query:
            return jsonify({
                'error': 'No query provided',
                'chat_response': 'Please provide a question to answer.'
            }), 400

        # Check if graph database is available
        if not app.config.get('graph_db'):
            logger.error("Graph database not available in app config")
            return jsonify({
                'error': 'Graph database not available',
                'chat_response': 'I apologize, but the knowledge graph is currently unavailable. Please try again later.'
            }), 503

        try:
            # Initialize LlamaService for processing queries
            llama_service = LlamaService()
            response = llama_service.process_query(query)

            if not response:
                logger.error("LlamaService returned None response")
                return jsonify({
                    'error': 'Service error',
                    'chat_response': 'Sorry, I encountered an error while processing your request. Please try again.'
                }), 500

            return jsonify({
                'response': response.get('chat_response', 'No response generated'),
                'technical_details': {
                    'queries': response.get('queries', {}),
                    'results': response.get('results', 'No matches found in knowledge graph')
                }
            }), 200

        except Exception as e:
            logger.error(f"Error processing query with LlamaService: {str(e)}")
            return jsonify({
                'error': 'Failed to process query',
                'chat_response': 'I encountered an error while processing your question. Please try again.'
            }), 500

    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'chat_response': 'An unexpected error occurred. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)