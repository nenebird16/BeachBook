import os
import logging
import config  # Import config first to load environment variables
from flask import Flask, request, render_template, jsonify, Response, stream_with_context, session
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log Neo4j configuration status
logger.debug(f"Initial NEO4J_URI: {'Present' if os.environ.get('NEO4J_URI') else 'Not present'}")
logger.debug(f"Config NEO4J_URI: {'Present' if config.NEO4J_URI else 'Not present'}")

# Remove NEO4J_URI from environment after config is loaded
original_uri = os.environ.pop('NEO4J_URI', None)

from services.document_processor import DocumentProcessor
from services.graph_service import GraphService
from services.llama_service import LlamaService

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

def process_document_with_progress(file_path):
    """Process document and yield progress events"""
    try:
        # Initial upload stage
        yield "data: {\"stage\": \"uploading\", \"progress\": 20}\n\n"

        # Open and process the file
        with open(file_path, 'r') as file:
            content = file.read()
            # Create a file-like object
            class FileWrapper:
                def __init__(self, content, filename):
                    self.content = content
                    self.filename = filename

                def read(self):
                    return self.content.encode('utf-8')

            filename = os.path.basename(file_path)
            file_obj = FileWrapper(content, filename)

            # Process document
            doc_info = doc_processor.process_document(file_obj)
            logger.debug(f"Document processed with info: {doc_info}")

            # Stream intermediate progress updates
            stages = [
                ('extracting', 40),
                ('processing', 60),
                ('analyzing', 80),
                ('storing', 90)
            ]

            for stage, progress in stages:
                yield f"data: {{\"stage\": \"{stage}\", \"progress\": {progress}}}\n\n"

            # Send completion event
            yield "data: {\"stage\": \"complete\", \"progress\": 100}\n\n"

    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}")
        error_msg = str(e).replace('"', '\\"')  # Escape quotes for JSON
        yield f"data: {{\"stage\": \"error\", \"error\": \"{error_msg}\"}}\n\n"

@app.route('/upload', methods=['POST', 'GET'])
def upload_document():
    """Handle document upload and processing with event streaming"""
    try:
        if request.method == 'GET':
            # Get the filename from the query parameter
            filename = request.args.get('filename')
            if not filename:
                return jsonify({'error': 'No filename provided'}), 400

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404

            # Return SSE response
            return Response(
                stream_with_context(process_document_with_progress(file_path)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:  # POST request
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            logger.info(f"File saved: {file_path}")
            return jsonify({'status': 'processing', 'filename': filename})

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