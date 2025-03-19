import os
import logging
import config
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
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

def init_storage_services():
    """Initialize storage services with proper error handling"""
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

    except Exception as e:
        logger.error(f"Failed to initialize storage services: {str(e)}")

    return services

# Initialize storage services
services = init_storage_services()

# Make storage services available to the app context
app.config['graph_db'] = services.get('graph_db')

# Configure app routes
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

        try:
            # Initialize LlamaService with graph database from app config
            llama_service = LlamaService(graph_db=app.config.get('graph_db'))

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

    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': 'An unexpected error occurred. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)