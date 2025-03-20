
import logging
from services.llama_service import LlamaService
from services.test_data import setup_test_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_graph_rag():
    """Test the improved GraphRAG implementation"""
    try:
        # Setup test data
        logger.info("Setting up test data...")
        if not setup_test_data():
            logger.error("Failed to setup test data")
            return

        # Initialize service
        service = LlamaService()
        
        # Test queries
        test_queries = [
            "What affects muscle length?",
            "How does tension relate to movement?",
            "Tell me about power output"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            result = service.process_query(query)
            logger.info(f"Response: {result['response']}")
            logger.info(f"Technical details: {result['technical_details']}")
            
        logger.info("Test complete")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_graph_rag()
