import os
import logging
import config
from flask import Flask, request, render_template, jsonify
from llama_service import LlamaService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
            # Initialize LlamaService for processing queries
            llama_service = LlamaService()
            result = llama_service.process_query(query)

            if not result:
                logger.error("LlamaService returned None response")
                return jsonify({
                    'error': 'Service error',
                    'response': 'Sorry, I encountered an error while processing your request. Please try again.'
                }), 500

            # Ensure we have a chat response
            response = result.get('response', 'I apologize, but I was unable to generate a response.')

            return jsonify({
                'response': response,
                'technical_details': {
                    'queries': result.get('queries', {}),
                    'results': result.get('results', 'No matches found in knowledge graph')
                }
            }), 200

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