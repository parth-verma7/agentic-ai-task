import logging

def setup_logging():
    logging.basicConfig(
        filename='logs/system.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
