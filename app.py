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

    logger.info("All required environment variables are present")
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

        # Check environment variables
        env_vars = {
            'NEO4J_URI': bool(os.environ.get('NEO4J_URI')),
            'NEO4J_USER': bool(os.environ.get('NEO4J_USER')),
            'NEO4J_PASSWORD': bool(os.environ.get('NEO4J_PASSWORD'))
        }

        # Determine overall health status
        is_healthy = any(status['connected'] for status in storage_status.values())

        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'environment': env_vars,
            'storage': storage_status
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
            if graph_db: #Check if graph_db is initialized
                node = graph_db.create_node('Document', doc_info)
                logger.debug(f"Document node created: {node}")
            else:
                logger.warning("Graph database not initialized, skipping document processing.")

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
            if object_storage: #Check if object_storage is initialized
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
            else:
                logger.warning("Object storage not initialized, skipping file storage.")
                return jsonify({'error': 'Object storage not available'}), 503

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
        if graph_db: #Check if graph_db is initialized
            results = graph_db.query(
                "MATCH (d:Document) WHERE d.content CONTAINS $query RETURN d",
                {'query': query}
            )
            return jsonify({
                'results': results
            })
        else:
            logger.warning("Graph database not initialized, skipping query.")
            return jsonify({'error': 'Graph database not available'}), 503

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)