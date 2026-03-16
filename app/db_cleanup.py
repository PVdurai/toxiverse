"""
Database Cleanup Module
Automatically cleans up old data from the database and session folders every 24 hours
"""

from datetime import datetime, timedelta
from flask import current_app
from app.db_models import db, Dataset, Chemical, QSARModel, CVResults, Bioprofile, KDNNPrediction, Task, User, Job
import logging
import os
import shutil
import time

# Configure logging
logger = logging.getLogger(__name__)

# Path to session folders
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_FILES_DIR = os.path.join(ROOT_DIR, 'data/tmp')


def cleanup_old_data(hours=24):
    """
    Clean up database records older than specified hours

    Args:
        hours (int): Number of hours after which data should be deleted (default: 24)

    Returns:
        dict: Statistics about cleaned records
    """
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        stats = {
            'datasets': 0,
            'chemicals': 0,
            'qsar_models': 0,
            'cv_results': 0,
            'bioprofiles': 0,
            'kdnn_predictions': 0,
            'tasks': 0,
            'guest_users': 0
        }

        logger.info(f"Starting database cleanup for records older than {hours} hours (before {cutoff_time})")

        # 1. Clean up old guest users (username starts with 'guest_')
        # Only delete guest users, preserve registered users
        old_guest_users = User.query.filter(
            User.username.like('guest_%'),
            User.user_created < cutoff_time
        ).all()

        for user in old_guest_users:
            logger.info(f"Deleting guest user: {user.username} (created: {user.user_created})")
            db.session.delete(user)
            stats['guest_users'] += 1

        # 2. Clean up old completed tasks
        old_tasks = Task.query.filter(
            Task.complete == True,
            Task.time_completed < cutoff_time
        ).all()

        for task in old_tasks:
            logger.info(f"Deleting completed task: {task.name} (completed: {task.time_completed})")
            db.session.delete(task)
            stats['tasks'] += 1

        # 2.5. Clean up old jobs (Bioprofiler/QSAR job tracking)
        old_jobs = Job.query.filter(
            Job.created_at < cutoff_time
        ).all()

        stats['jobs'] = 0
        for job in old_jobs:
            logger.info(f"Deleting job: {job.job_id} (type: {job.job_type}, created: {job.created_at})")
            db.session.delete(job)
            stats['jobs'] += 1

        # 3. Clean up old datasets (and related data will cascade)
        old_datasets = Dataset.query.filter(
            Dataset.created < cutoff_time
        ).all()

        for dataset in old_datasets:
            # Count related records before deletion
            chemicals_count = Chemical.query.filter_by(dataset_id=dataset.id).count()
            qsar_count = QSARModel.query.filter_by(dataset_id=dataset.id).count()
            bio_count = Bioprofile.query.filter_by(dataset_id=dataset.id).count()

            logger.info(f"Deleting dataset: {dataset.dataset_name} (created: {dataset.created})")
            logger.info(f"  - {chemicals_count} chemicals, {qsar_count} QSAR models, {bio_count} bioprofiles")

            stats['chemicals'] += chemicals_count
            stats['qsar_models'] += qsar_count
            stats['bioprofiles'] += bio_count

            db.session.delete(dataset)
            stats['datasets'] += 1

        # 4. Clean up orphaned QSAR model results
        old_cv_results = CVResults.query.join(QSARModel).filter(
            QSARModel.created < cutoff_time
        ).all()

        for cv in old_cv_results:
            db.session.delete(cv)
            stats['cv_results'] += 1

        # 5. Clean up old k-DNN predictions
        old_kdnn = KDNNPrediction.query.filter(
            KDNNPrediction.created < cutoff_time
        ).all()

        for kdnn in old_kdnn:
            logger.info(f"Deleting k-DNN prediction: {kdnn.name} (created: {kdnn.created})")
            db.session.delete(kdnn)
            stats['kdnn_predictions'] += 1

        # Commit all deletions
        db.session.commit()

        logger.info(f"Database cleanup completed successfully")
        logger.info(f"Statistics: {stats}")

        return stats

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during database cleanup: {str(e)}")
        raise


def cleanup_all_temporary_data():
    """
    Clean up ALL temporary data from the database
    WARNING: This preserves only registered users, deletes all other data
    """
    try:
        stats = {
            'datasets': 0,
            'qsar_models': 0,
            'bioprofiles': 0,
            'kdnn_predictions': 0,
            'tasks': 0,
            'guest_users': 0
        }

        logger.info("Starting FULL database cleanup (all temporary data)")

        # 1. Delete all guest users
        guest_users = User.query.filter(User.username.like('guest_%')).all()
        for user in guest_users:
            db.session.delete(user)
            stats['guest_users'] += 1

        # 2. Delete all completed tasks
        completed_tasks = Task.query.filter_by(complete=True).all()
        for task in completed_tasks:
            db.session.delete(task)
            stats['tasks'] += 1

        # 3. Delete all datasets (cascades to chemicals, QSAR models, bioprofiles)
        all_datasets = Dataset.query.all()
        for dataset in all_datasets:
            db.session.delete(dataset)
            stats['datasets'] += 1

        # 4. Delete all k-DNN predictions
        all_kdnn = KDNNPrediction.query.all()
        for kdnn in all_kdnn:
            db.session.delete(kdnn)
            stats['kdnn_predictions'] += 1

        db.session.commit()

        logger.info(f"Full database cleanup completed")
        logger.info(f"Statistics: {stats}")

        return stats

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during full database cleanup: {str(e)}")
        raise


def cleanup_old_session_folders(hours=24):
    """
    Clean up session folders older than specified hours

    Args:
        hours (int): Number of hours after which session folders should be deleted (default: 24)

    Returns:
        dict: Statistics about cleaned session folders
    """
    try:
        if not os.path.exists(TMP_FILES_DIR):
            logger.warning(f"TMP_FILES_DIR does not exist: {TMP_FILES_DIR}")
            return {'session_folders_deleted': 0}

        now = time.time()
        expiry_threshold = now - (hours * 3600)  # Convert hours to seconds
        stats = {'session_folders_deleted': 0}

        logger.info(f"Starting session folder cleanup for folders older than {hours} hours")

        for folder in os.listdir(TMP_FILES_DIR):
            folder_path = os.path.join(TMP_FILES_DIR, folder)

            # Only process directories that start with "session_"
            if folder.startswith("session_") and os.path.isdir(folder_path):
                folder_age = os.stat(folder_path).st_mtime  # Last modified time

                if folder_age < expiry_threshold:
                    try:
                        shutil.rmtree(folder_path, ignore_errors=True)
                        logger.info(f"Deleted old session folder: {folder_path}")
                        stats['session_folders_deleted'] += 1
                    except Exception as e:
                        logger.error(f"Error deleting session folder {folder_path}: {str(e)}")

        logger.info(f"Session folder cleanup completed. Deleted {stats['session_folders_deleted']} folders")
        return stats

    except Exception as e:
        logger.error(f"Error during session folder cleanup: {str(e)}")
        return {'session_folders_deleted': 0}


def get_database_statistics():
    """
    Get current database statistics

    Returns:
        dict: Count of records in each table
    """
    stats = {
        'total_users': User.query.count(),
        'guest_users': User.query.filter(User.username.like('guest_%')).count(),
        'registered_users': User.query.filter(~User.username.like('guest_%')).count(),
        'datasets': Dataset.query.count(),
        'chemicals': Chemical.query.count(),
        'qsar_models': QSARModel.query.count(),
        'cv_results': CVResults.query.count(),
        'bioprofiles': Bioprofile.query.count(),
        'kdnn_predictions': KDNNPrediction.query.count(),
        'total_tasks': Task.query.count(),
        'completed_tasks': Task.query.filter_by(complete=True).count(),
        'pending_tasks': Task.query.filter_by(complete=False).count()
    }

    # Calculate old records (>24 hours)
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    stats['old_datasets'] = Dataset.query.filter(Dataset.created < cutoff_time).count()
    stats['old_tasks'] = Task.query.filter(
        Task.complete == True,
        Task.time_completed < cutoff_time
    ).count()
    stats['old_guest_users'] = User.query.filter(
        User.username.like('guest_%'),
        User.user_created < cutoff_time
    ).count()

    return stats


def schedule_cleanup_job(app):
    """
    Set up automatic cleanup job to run every 24 hours

    Args:
        app: Flask application instance
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    import atexit

    scheduler = BackgroundScheduler()

    # Schedule cleanup to run every 24 hours
    scheduler.add_job(
        func=lambda: cleanup_with_app_context(app),
        trigger=IntervalTrigger(hours=24),
        id='database_cleanup_job',
        name='Clean up old database records',
        replace_existing=True
    )

    scheduler.start()
    logger.info("Database cleanup scheduler started (runs every 24 hours)")

    # Shut down the scheduler when the app exits
    atexit.register(lambda: scheduler.shutdown())

    return scheduler


def cleanup_with_app_context(app):
    """
    Run cleanup with Flask application context
    Cleans both database records and session folders
    """
    with app.app_context():
        try:
            # Clean up database records
            db_stats = cleanup_old_data(hours=24)
            logger.info(f"Scheduled database cleanup completed: {db_stats}")

            # Clean up session folders
            folder_stats = cleanup_old_session_folders(hours=24)
            logger.info(f"Scheduled session folder cleanup completed: {folder_stats}")

            # Combined stats
            combined_stats = {**db_stats, **folder_stats}
            logger.info(f"Full cleanup completed: {combined_stats}")

        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {str(e)}")
