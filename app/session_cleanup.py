import os
import time
import shutil
 
import threading
# Path to the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_FILES_DIR = os.path.join(ROOT_DIR, 'data/tmp')

 
class SessionCleanupManager:
    """
    Handles scheduled cleanup of session folders using Redis RQ.
    """

    @staticmethod
    def cleanup_old_sessions(expiry_hours=24):
        
        print("Deletes session started")
        """
        Deletes session folders older than the given expiry time (default: 24 hours).
        """
        now = time.time()
        expiry_threshold = now - (expiry_hours * 3600)  # Convert hours to seconds

        for folder in os.listdir(TMP_FILES_DIR):
            print(f"📁 Checking folder: {folder}")
            folder_path = os.path.join(TMP_FILES_DIR, folder)
            print(f"📂 Folder path: {folder_path}")
            if folder.startswith("session_") and os.path.isdir(folder_path):
                folder_age = os.stat(folder_path).st_mtime  # Last modified time
                
                if folder_age < expiry_threshold:
                    shutil.rmtree(folder_path, ignore_errors=True)
                    print(f"🗑 Deleted old session folder: {folder_path}")
        print("Deletes session ended")

        return "Session cleanup completed"


    @staticmethod
    def start_cleanup_scheduler():
        """
        Runs the cleanup process every 24 hours in a separate thread.
        """
        def run():
            while True:
                SessionCleanupManager.cleanup_old_sessions()  # ✅ Now calling the function correctly
                time.sleep(86400)  # 🔥 24 hours (86400 seconds)

        thread = threading.Thread(target=run, daemon=True)  # ✅ Run as a background thread
        thread.start()
        print("🕒 Session cleanup thread started.")

# ✅ Start cleanup scheduler when script runs
SessionCleanupManager.start_cleanup_scheduler()
