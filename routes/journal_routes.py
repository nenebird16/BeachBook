from datetime import datetime
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import logging
from models.journal import JournalEntry
from replit.object_storage import Client

logger = logging.getLogger(__name__)
journal_routes = Blueprint('journal', __name__)

ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a'}
# Initialize storage client with error handling
try:
    storage_client = Client()
    # Test bucket access
    storage_client.get_bucket()
except Exception as e:
    logger.error(f"Failed to initialize storage client: {str(e)}")
    storage_client = None

def allowed_audio_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@journal_routes.route('/journal')
def journal_page():
    """Render the journal page"""
    return render_template('journal.html')

@journal_routes.route('/journal/audio', methods=['POST'])
def upload_audio():
    """Handle audio file uploads"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if audio_file and allowed_audio_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            object_key = f"audio/{timestamp}_{filename}"

            if not storage_client:
                return jsonify({'error': 'Object Storage not configured. Please create a bucket first.'}), 500

            try:
                # Upload to Object Storage
                storage_client.upload_from_bytes(object_key, audio_file.read())
                
                # Get public URL
                audio_url = storage_client.get_url(object_key)
            except Exception as e:
                logger.error(f"Storage error: {str(e)}")
                return jsonify({'error': 'Failed to upload to storage. Please try again.'}), 500

            # Create journal entry in Neo4j
            entry = JournalEntry.create_audio_entry(audio_url)

            return jsonify({
                'message': 'Audio uploaded successfully',
                'entry_id': entry['id']
            }), 200

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({'error': 'Failed to upload audio'}), 500

@journal_routes.route('/journal/text', methods=['POST'])
def add_text_entry():
    """Add a text journal entry"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text content provided'}), 400

        text = data['text']

        # Create journal entry in Neo4j
        entry = JournalEntry.create_text_entry(text)

        return jsonify({
            'message': 'Journal entry saved successfully',
            'entry_id': entry['id']
        }), 200

    except Exception as e:
        logger.error(f"Error creating text entry: {str(e)}")
        return jsonify({'error': 'Failed to save journal entry'}), 500

@journal_routes.route('/journal/history')
def get_journal_history():
    """Get journal entry history"""
    try:
        entries = JournalEntry.get_recent_entries()
        return jsonify(entries), 200

    except Exception as e:
        logger.error(f"Error fetching journal history: {str(e)}")
        return jsonify({'error': 'Failed to fetch journal history'}), 500