import os
import uuid
import threading
import json
from datetime import datetime
from typing import Optional
from fastapi import HTTPException
import logging
from models import MessageResponse, ChatHistoryItem, ChatHistoryResponse

# Import SQLAlchemy components
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

Base = declarative_base()

class SessionModel(Base):
    __tablename__ = 'sessions'
    
    session_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    user_email = Column(String)
    first_query = Column(Text)
    timestamp = Column(String, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class MessageModel(Base):
    __tablename__ = 'messages'
    
    message_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)
    model = Column(String)
    timestamp = Column(String, nullable=False)
    sources = Column(Text)
    metrics = Column(Text)
    trace_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class MessageRatingModel(Base):
    __tablename__ = 'message_ratings'
    
    message_id = Column(String, ForeignKey('messages.message_id', ondelete='CASCADE'), primary_key=True)
    user_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id', ondelete='CASCADE'), nullable=False)
    rating = Column(String, CheckConstraint("rating IN ('up', 'down')"))
    created_at = Column(DateTime, default=datetime.utcnow)

def build_db_uri(username, instance_name, use_sp=True):
    """
    Build PostgreSQL connection URI for Databricks database instance.
    
    Args:
        username (str): Database username
        instance_name (str): Databricks database instance name
        use_sp (bool): Whether to use SP credentials (default: True)
    
    Returns:
        str: PostgreSQL connection URI
    """
    if use_sp:
        w = WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"), 
            client_id=os.getenv("CLIENT_ID"), 
            client_secret=os.getenv("CLIENT_SECRET")
        )
    else:
        w = WorkspaceClient()
    
    # Get database instance details
    instance = w.database.get_database_instance(name=instance_name)
    host = instance.read_write_dns
    
    # Get access token for database authentication
    cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[instance_name])
    pgpassword =cred.token

    # Use dynamic username for non-SP mode, otherwise use provided username
    if not use_sp:
        username = w.current_user.me().emails[0].value
    
    if "@" in username:
        username = username.replace("@", "%40")

    # Connection parameters with chatbot_schema
    db_uri = f"postgresql://{username}:{pgpassword}@{host}:5432/databricks_postgres?sslmode=require&options=-csearch_path%3Dchatbot_schema"
    return db_uri

class ChatDatabase:
    def __init__(self, db_file='chat_history.db'):
        self.db_file = db_file
        self.db_lock = threading.Lock()
        self.first_message_cache = {}
        self.postgres_config = None
        
        # Initialize database based on configuration
        if self._has_postgres_config():
            self._init_postgres()
        else:
            self._init_sqlite()
    
    def _has_postgres_config(self):
        """Check if PostgreSQL configuration is available"""
        return all(os.getenv(var) for var in ['DB_INSTANCE_NAME', 'CLIENT_ID', 'CLIENT_SECRET'])
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            username = os.getenv("CLIENT_ID")
            instance_name = os.getenv("DB_INSTANCE_NAME")
            
            # Store PostgreSQL configuration for dynamic token regeneration
            self.postgres_config = {
                'username': username,
                'instance_name': instance_name,
                'use_sp': True
            }
            
            # Build initial database URI
            self.db_uri = build_db_uri(username, instance_name, use_sp=True)
            
            # Create engine with dynamic token regeneration
            self.engine = create_engine(
                self.db_uri,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=300  # Reduced to 5 minutes for faster token refresh
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create schema and initialize database tables
            self._create_schema_if_not_exists()
            self.init_db()
            logger.info("Using PostgreSQL database with dynamic token regeneration")
            
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")
            logger.info("Falling back to SQLite")
            self._init_sqlite()
    

    
    def _create_schema_if_not_exists(self):
        """Create the chatbot_schema if it doesn't exist"""
        from sqlalchemy import text
        
        db = self.get_session()
        try:
            # Create schema using session
            result = db.execute(text("CREATE SCHEMA IF NOT EXISTS chatbot_schema"))
            db.commit()
            logger.info("Schema 'chatbot_schema' created or already exists")
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def _init_sqlite(self):
        """Initialize SQLite connection"""
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        
        # Create SQLite engine
        self.engine = create_engine(
            f'sqlite:///{self.db_file}',
            poolclass=StaticPool,
            connect_args={'check_same_thread': False}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize database tables
        self.init_db()
        logger.info("Using SQLite database")
    
    def _get_connection_with_fresh_token(self):
        """Get a fresh database connection with regenerated token"""
        if not self.postgres_config:
            raise Exception("PostgreSQL configuration not available")
        
        try:
            # Regenerate database URI with fresh token
            fresh_uri = build_db_uri(
                self.postgres_config['username'],
                self.postgres_config['instance_name'],
                self.postgres_config['use_sp']
            )
            
            # Create a new engine with the fresh URI
            from sqlalchemy import create_engine
            fresh_engine = create_engine(fresh_uri)
            
            # Get connection from the fresh engine
            connection = fresh_engine.connect()
            logger.debug("Created fresh database connection with regenerated token")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create fresh database connection: {e}")
            raise
    
    def _recreate_engine_with_fresh_token(self):
        """Recreate the engine and session factory with a fresh token"""
        if not self.postgres_config:
            raise Exception("PostgreSQL configuration not available")
        
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool
            
            # Regenerate database URI with fresh token
            fresh_uri = build_db_uri(
                self.postgres_config['username'],
                self.postgres_config['instance_name'],
                self.postgres_config['use_sp']
            )
            
            # Create new engine with fresh token
            self.engine = create_engine(
                fresh_uri,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=300  # 5 minutes for faster token refresh
            )
            
            # Create new session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Successfully recreated engine with fresh token")
            
        except Exception as e:
            logger.error(f"Failed to recreate engine with fresh token: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session with retry logic for token expiration"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.SessionLocal()
            except Exception as e:
                if "password authentication failed" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database authentication failed (attempt {attempt + 1}/{max_retries}), regenerating token...")
                    # Dispose old engine and create new one with fresh token
                    if hasattr(self, 'engine') and self.engine:
                        self.engine.dispose()
                    self._recreate_engine_with_fresh_token()
                    continue
                else:
                    logger.error(f"Failed to get database session after {max_retries} attempts: {e}")
                    raise
    
    def _execute_with_retry(self, operation_func, operation_name: str = "database operation"):
        """Execute a database operation with retry logic for token expiration"""
        max_retries = 3
        for attempt in range(max_retries):
            db = None
            try:
                db = self.get_session()
                result = operation_func(db)
                return result
            except Exception as e:
                if db:
                    db.close()
                if "password authentication failed" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"{operation_name} failed due to auth (attempt {attempt + 1}/{max_retries}), regenerating token...")
                    # Dispose old engine and create new one with fresh token
                    if hasattr(self, 'engine') and self.engine:
                        self.engine.dispose()
                    self._recreate_engine_with_fresh_token()
                    continue
                else:
                    logger.error(f"Failed {operation_name} after {max_retries} attempts: {e}")
                    raise
            finally:
                if db:
                    db.close()
    
    def init_db(self):
        """Initialize the database with required tables and indexes"""
        with self.db_lock:
            try:
                Base.metadata.create_all(bind=self.engine)
                logger.info("Database tables created successfully")                
                
            except Exception as e:
                logger.error(f"Error initializing database: {str(e)}")
                raise
    
    def save_message_to_session(self, session_id: str, user_id: str, message: MessageResponse, user_info: dict = None, is_first_message: bool = False):
        """Save a message to a chat session, creating the session if it doesn't exist"""
        with self.db_lock:
            db = self.get_session()
            try:
                logger.info(f"Saving message: session_id={session_id}, user_id={user_id}, message_id={message.message_id}")
                
                # Check if session exists, create if it doesn't
                existing_session = db.query(SessionModel).filter(
                    SessionModel.session_id == session_id
                ).first()
                
                if not existing_session:
                    logger.info(f"Creating new session: session_id={session_id}, user_id={user_id}")
                    session_model = SessionModel(
                        session_id=session_id,
                        user_id=user_id,
                        user_email=user_info.get('email') if user_info else None,
                        first_query=message.content if message.role == 'user' and is_first_message else None,
                        timestamp=message.timestamp.isoformat(),
                        is_active=1
                    )
                    db.add(session_model)
                    db.flush()  # Flush to ensure session is created before message
                
                # Save message
                message_model = MessageModel(
                    message_id=message.message_id,
                    session_id=session_id,
                    user_id=user_id,
                    content=message.content,
                    role=message.role,
                    model=message.model,
                    timestamp=message.timestamp.isoformat(),
                    sources=json.dumps(message.sources) if message.sources else None,
                    metrics=json.dumps(message.metrics) if message.metrics else None,
                    trace_id=message.trace_id
                )
                
                db.add(message_model)
                db.commit()
                
                # Update cache after saving message
                self.first_message_cache[session_id] = False
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error saving message to session: {str(e)}")
                raise
            finally:
                db.close()
    
    def update_message(self, session_id: str, user_id: str, message: MessageResponse):
        """Update an existing message in the database"""
        with self.db_lock:
            db = self.get_session()
            try:
                # Update message
                message_model = db.query(MessageModel).filter(
                    MessageModel.message_id == message.message_id,
                    MessageModel.session_id == session_id,
                    MessageModel.user_id == user_id
                ).first()
                
                if message_model:
                    message_model.content = message.content
                    message_model.role = message.role
                    message_model.model = message.model
                    message_model.timestamp = message.timestamp.isoformat()
                    message_model.sources = json.dumps(message.sources) if message.sources else None
                    message_model.metrics = json.dumps(message.metrics) if message.metrics else None
                    db.commit()
                    logger.info(f"Updated message: {message.message_id}")
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Message {message.message_id} not found in session {session_id}"
                    )
                    
            except Exception as e:
                db.rollback()
                logger.error(f"Error updating message: {str(e)}")
                raise
            finally:
                db.close()
    
    def get_chat_history(self, user_id: str = None) -> ChatHistoryResponse:
        """Retrieve chat sessions with their messages for a specific user"""
        def _get_history_operation(db):
            # Get all sessions for the user
            sessions_query = db.query(SessionModel).filter(
                SessionModel.user_id == user_id,
                SessionModel.is_active == 1
            ).order_by(SessionModel.created_at.desc())
            
            sessions = sessions_query.all()
            
            chat_history = []
            for session in sessions:
                # Get messages for this session
                messages = db.query(MessageModel).filter(
                    MessageModel.session_id == session.session_id,
                    MessageModel.user_id == session.user_id
                ).order_by(MessageModel.created_at.asc()).all()
                
                # Convert to ChatHistoryItem format
                chat_messages = []
                for msg in messages:
                    # Get the rating for this message
                    rating = db.query(MessageRatingModel).filter(
                        MessageRatingModel.message_id == msg.message_id,
                        MessageRatingModel.user_id == msg.user_id
                    ).first()
                    
                    chat_message = MessageResponse(
                        message_id=msg.message_id,
                        content=msg.content,
                        role=msg.role,
                        model=msg.model,
                        timestamp=datetime.fromisoformat(msg.timestamp),
                        created_at=msg.created_at,
                        sources=json.loads(msg.sources) if msg.sources else None,
                        metrics=json.loads(msg.metrics) if msg.metrics else None,
                        rating=rating.rating if rating else None,
                        trace_id=msg.trace_id
                    )
                    
                    chat_messages.append(chat_message)
                
                chat_history.append(ChatHistoryItem(
                    sessionId=session.session_id,
                    firstQuery=session.first_query,
                    messages=chat_messages,
                    timestamp=datetime.fromisoformat(session.timestamp),
                    created_at=session.created_at,
                    isActive=bool(session.is_active)
                ))
            
            return ChatHistoryResponse(sessions=chat_history)
        
        return self._execute_with_retry(_get_history_operation, "get chat history")
    
    def get_chat(self, session_id: str, user_id: str = None) -> ChatHistoryItem:
        """Retrieve a specific chat session"""
        db = self.get_session()
        try:
            logger.info(f"Getting chat for session_id: {session_id}, user_id: {user_id}")
            
            # Get session info with user check
            if user_id:
                session = db.query(SessionModel).filter(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id
                ).first()
            else:
                session = db.query(SessionModel).filter(
                    SessionModel.session_id == session_id
                ).first()
            
            if not session:
                logger.error(f"Session not found: session_id={session_id}, user_id={user_id}")
                raise HTTPException(status_code=404, detail="Chat not found")
            
            # Get messages ordered by created_at
            messages = db.query(MessageModel).filter(
                MessageModel.session_id == session_id,
                MessageModel.user_id == user_id
            ).order_by(MessageModel.created_at.asc()).all()
            
            chat_messages = []
            for msg in messages:
                # Get the rating for this message
                rating = db.query(MessageRatingModel).filter(
                    MessageRatingModel.message_id == msg.message_id,
                    MessageRatingModel.user_id == msg.user_id
                ).first()
                
                chat_message = MessageResponse(
                    message_id=msg.message_id,
                    content=msg.content,
                    role=msg.role,
                    model=msg.model,
                    timestamp=datetime.fromisoformat(msg.timestamp),
                    created_at=msg.created_at,
                    sources=json.loads(msg.sources) if msg.sources else None,
                    metrics=json.loads(msg.metrics) if msg.metrics else None,
                    rating=rating.rating if rating else None,
                    trace_id=msg.trace_id
                )
                
                chat_messages.append(chat_message)
            
            return ChatHistoryItem(
                sessionId=session_id,
                firstQuery=session.first_query,
                messages=chat_messages,
                timestamp=datetime.fromisoformat(session.timestamp),
                created_at=session.created_at,
                isActive=bool(session.is_active)
            )
            
        except Exception as e:
            logger.error(f"Error getting chat: {str(e)}")
            raise
        finally:
            db.close()
    
    def clear_session(self, session_id: str, user_id: str):
        """Clear a session and its messages"""
        with self.db_lock:
            db = self.get_session()
            try:
                # Mark session as inactive
                session = db.query(SessionModel).filter(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id
                ).first()
                
                if session:
                    session.is_active = 0
                    db.commit()
                    logger.info(f"Cleared session: {session_id}")
                else:
                    logger.warning(f"Session not found for clearing: {session_id}")
                
                # Clear cache
                if session_id in self.first_message_cache:
                    del self.first_message_cache[session_id]
                    
            except Exception as e:
                db.rollback()
                logger.error(f"Error clearing session: {str(e)}")
                raise
            finally:
                db.close()
    
    def delete_session(self, session_id: str, user_id: str):
        """Delete a session and all its associated data permanently"""
        with self.db_lock:
            db = self.get_session()
            try:
                # Delete session and all related data (cascade will handle messages and ratings)
                session = db.query(SessionModel).filter(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id
                ).first()
                
                if session:
                    db.delete(session)
                    db.commit()
                    logger.info(f"Deleted session and all data: {session_id}")
                else:
                    logger.warning(f"Session not found for deletion: {session_id}")
                
                # Clear cache
                if session_id in self.first_message_cache:
                    del self.first_message_cache[session_id]
                    
            except Exception as e:
                db.rollback()
                logger.error(f"Error deleting session: {str(e)}")
                raise
            finally:
                db.close()
    
    def delete_user_sessions(self, user_id: str):
        """Delete all sessions for a specific user"""
        with self.db_lock:
            db = self.get_session()
            try:
                # Get all sessions for the user
                sessions = db.query(SessionModel).filter(
                    SessionModel.user_id == user_id
                ).all()
                
                if sessions:
                    for session in sessions:
                        db.delete(session)
                    
                    db.commit()
                    logger.info(f"Deleted {len(sessions)} sessions for user: {user_id}")
                else:
                    logger.info(f"No sessions found for user: {user_id}")
                
                # Clear all cache entries for this user
                cache_keys_to_remove = [key for key in self.first_message_cache.keys() 
                                      if key.startswith(f"{user_id}_")]
                for key in cache_keys_to_remove:
                    del self.first_message_cache[key]
                    
            except Exception as e:
                db.rollback()
                logger.error(f"Error deleting user sessions: {str(e)}")
                raise
            finally:
                db.close()
    
    def is_first_message(self, session_id: str, user_id: str) -> bool:
        """Check if this is the first message in a session"""
        # Check cache first
        if session_id in self.first_message_cache:
            return self.first_message_cache[session_id]
        
        db = self.get_session()
        try:
            # Check if session exists
            session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            
            is_first = session is None
            self.first_message_cache[session_id] = is_first
            return is_first
            
        except Exception as e:
            logger.error(f"Error checking first message: {str(e)}")
            return False
        finally:
            db.close()

    def update_message_rating(self, message_id: str, user_id: str, rating: str | None) -> bool:
        """Update message rating"""
        with self.db_lock:
            db = self.get_session()
            try:
                # First verify the message exists and belongs to the user
                message = db.query(MessageModel).filter(
                    MessageModel.message_id == message_id,
                    MessageModel.user_id == user_id
                ).first()
                
                if not message:
                    logger.error(f"Message {message_id} not found for user {user_id}")
                    return False
                
                session_id = message.session_id
                
                if rating is None:
                    # Remove the rating
                    db.query(MessageRatingModel).filter(
                        MessageRatingModel.message_id == message_id,
                        MessageRatingModel.user_id == user_id
                    ).delete()
                else:
                    # Update or insert the rating
                    existing_rating = db.query(MessageRatingModel).filter(
                        MessageRatingModel.message_id == message_id,
                        MessageRatingModel.user_id == user_id
                    ).first()
                    
                    if existing_rating:
                        existing_rating.rating = rating
                    else:
                        new_rating = MessageRatingModel(
                            message_id=message_id,
                            user_id=user_id,
                            session_id=session_id,
                            rating=rating
                        )
                        db.add(new_rating)
                
                db.commit()
                logger.info(f"Updated rating for message {message_id}: {rating}")
                return True
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error updating message rating: {str(e)}")
                return False
            finally:
                db.close()

    def get_message_rating(self, message_id: str, user_id: str) -> str | None:
        """Get the rating of a message"""
        db = self.get_session()
        try:
            rating = db.query(MessageRatingModel).filter(
                MessageRatingModel.message_id == message_id,
                MessageRatingModel.user_id == user_id
            ).first()
            
            return rating.rating if rating else None
            
        except Exception as e:
            logger.error(f"Error getting message rating: {str(e)}")
            return None
        finally:
            db.close()
    
    def delete_session_api(self, session_id: str, user_id: str) -> dict:
        """Delete a specific session and all its chat history - API wrapper"""
        try:
            if not user_id:
                raise ValueError("User ID not found")
            
            self.delete_session(session_id, user_id)
            return {"message": f"Session {session_id} deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            raise ValueError(f"Failed to delete session: {str(e)}")
    
    def delete_user_sessions_api(self, user_id: str) -> dict:
        """Delete all sessions for a specific user - API wrapper"""
        try:
            if not user_id:
                raise ValueError("User ID not found")
            
            self.delete_user_sessions(user_id)
            return {"message": "All user sessions deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting user sessions: {str(e)}")
            raise ValueError(f"Failed to delete user sessions: {str(e)}")