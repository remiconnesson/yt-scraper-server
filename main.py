from app_factory import create_app
from downloader import get_file_size

app = create_app()

__all__ = ["app", "get_file_size"]
