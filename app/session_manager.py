import os
import uuid
from flask import g


# Path to the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_FILES_DIR = os.path.join(ROOT_DIR, 'data/tmp')

class SessionManager:
    """
    Handles per-thread session management in a Flask app.
    Ensures each request gets a unique session ID and folder.
    """
    
    @staticmethod
    def get_or_create_session_id():
        """
        Get or create a session ID for the current request.
        Uses Flask's `g` to store the session ID.
        """
        if not hasattr(g, 'session_id'):
            g.session_id = uuid.uuid4().hex
            print(f"🔹 New Session ID Created: {g.session_id}")
        else:
            print(f"🔹 Existing Session ID Used: {g.session_id}")

        return g.session_id

    @staticmethod
    def get_session_folder():
        """
        Creates and retrieves a unique session folder for the request.
        Uses the session ID to create a thread-safe directory.
        """
        session_id = SessionManager.get_or_create_session_id()
        session_folder = os.path.join(TMP_FILES_DIR, f"session_{session_id}")

        os.makedirs(session_folder, exist_ok=True)
        return session_folder

    @staticmethod
    def get_folder_by_session_id(session_id):
        """
        Retrieve the folder path for a given session ID.
        - Returns the folder path if it exists.
        - If the session folder doesn't exist, it does NOT create a new one.
        """
        session_folder = os.path.join(TMP_FILES_DIR, f"session_{session_id}")
        
        if os.path.exists(session_folder):
            return session_folder
        else:
            print(f"⚠️ Warning: Session folder does not exist for ID: {session_id}")
            return None  # Return None instead of creating a new folder
        

    @staticmethod
    def cleanup_old_sessions(expiry_hours=24):
        """
        Deletes session folders older than the given expiry time (default: 24 hours).
        """
        now = time.time()
        expiry_threshold = now - (expiry_hours * 3600)  # Convert hours to seconds

        for folder in os.listdir(TMP_FILES_DIR):
            folder_path = os.path.join(TMP_FILES_DIR, folder)
            
            if folder.startswith("session_") and os.path.isdir(folder_path):
                folder_age = os.stat(folder_path).st_mtime  # Last modified time
                
                if folder_age < expiry_threshold:
                    shutil.rmtree(folder_path, ignore_errors=True)
                    print(f"🗑 Deleted old session folder: {folder_path}")

        return "Session cleanup completed"
