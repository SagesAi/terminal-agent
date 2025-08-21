"""
Memory Database Management

Uses SQLite to store session history, messages, and tool call records.
"""

import sqlite3
import os
import json
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional
import logging

# Get logger
logger = logging.getLogger(__name__)

class MemoryDatabase:
    """Manages SQLite database connections and operations"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection
        
        Args:
            db_path: Database file path, if None, use default path
        """
        if not db_path:
            # Default storage in user directory
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, ".terminal_agent", "memory")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database table structure"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Make query results accessible by column name
        
        # Create sessions table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        # Create messages table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            type TEXT DEFAULT 'message',
            created_at TEXT NOT NULL,
            is_llm_message BOOLEAN DEFAULT 1,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Create tool calls table
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS tool_calls (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            input TEXT NOT NULL,
            output TEXT,
            created_at TEXT NOT NULL,
            tool_call_id TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        
        self.conn.commit()
        logger.info(f"Initialized memory database: {self.db_path}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Destructor, ensure database connection is closed"""
        self.close()
