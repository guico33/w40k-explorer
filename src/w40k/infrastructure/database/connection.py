"""Database connection and session management."""

from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine


class DatabaseManager:
    """Manages SQLite database connection and sessions."""
    
    def __init__(self, db_path: str = "data/articles.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine with SQLite optimizations
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "check_same_thread": False,  # Allow multiple threads
                "timeout": 30,  # 30 second timeout
            }
        )
        
        # Create tables
        self.create_tables()
    
    def create_tables(self) -> None:
        """Create all database tables."""
        SQLModel.metadata.create_all(self.engine)
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        with Session(self.engine) as session:
            yield session
    
    def drop_all_tables(self) -> None:
        """Drop all tables (for testing/reset)."""
        SQLModel.metadata.drop_all(self.engine)
    
    def get_db_info(self) -> dict:
        """Get database file information."""
        if not self.db_path.exists():
            return {"exists": False}
        
        stat = self.db_path.stat()
        return {
            "exists": True,
            "path": str(self.db_path),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime
        }