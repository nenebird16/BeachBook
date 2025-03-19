import os
import logging
import config
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from routes.journal_routes import journal_routes
from storage.factory import StorageFactory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize storage services
try:
    logger.info("Initializing storage services...")
    graph_db = StorageFactory.create_graph_database("neo4j")
    object_storage = StorageFactory.create_object_storage("replit")

    # Connect to storage services
    graph_db.connect()
    object_storage.connect()

    # Make storage services available to the app context
    app.config['graph_db'] = graph_db
    app.config['object_storage'] = object_storage

    logger.info("Storage services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize storage services: {str(e)}")
    raise

# Register blueprints
app.register_blueprint(journal_routes)

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

            class FileWrapper:
                def __init__(self, content, filename):
                    self.content = content
                    self.filename = filename

                def read(self):
                    return self.content.encode('utf-8')

            filename = os.path.basename(file_path)
            file_obj = FileWrapper(content, filename)

            # Process document using graph database
            doc_info = {'content': content, 'filename': filename}
            graph_db = app.config['graph_db']
            node = graph_db.create_node('Document', doc_info)
            logger.debug(f"Document node created: {node}")

            # Stream intermediate progress updates
            stages = [
                ('extracting', 40),
                ('processing', 60),
                ('analyzing', 80),
                ('storing', 90)
            ]

            for stage, progress in stages:
                yield f"data: {{\"stage\": \"{stage}\", \"progress\": {progress}}}\n\n"

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
            filename = request.args.get('filename')
            if not filename:
                return jsonify({'error': 'No filename provided'}), 400

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404

            return Response(
                stream_with_context(process_document_with_progress(file_path)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                    'Content-Type': 'text/event-stream',
                    'X-Content-Type-Options': 'nosniff'
                }
            )
        else:  # POST request
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Store file in object storage
            object_storage = app.config['object_storage']
            file_data = file.read()
            file_url = object_storage.store_file(
                file_data,
                secure_filename(file.filename),
                file.content_type
            )

            logger.info(f"File stored: {file_url}")
            return jsonify({
                'status': 'success',
                'file_url': file_url
            }), 200

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_knowledge():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Query graph database
        graph_db = app.config['graph_db']
        results = graph_db.query(
            "MATCH (d:Document) WHERE d.content CONTAINS $query RETURN d",
            {'query': query}
        )

        return jsonify({
            'results': results
        })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)