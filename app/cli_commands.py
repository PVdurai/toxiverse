"""
Flask CLI Commands for Database and Session Folder Management
"""

import click
import os
from flask.cli import with_appcontext
from app.db_cleanup import (
    cleanup_old_data,
    cleanup_all_temporary_data,
    get_database_statistics,
    cleanup_old_session_folders,
    TMP_FILES_DIR
)


@click.command('cleanup-db')
@click.option('--hours', default=24, help='Delete records older than this many hours')
@with_appcontext
def cleanup_database_command(hours):
    """Clean up old database records"""
    click.echo(f'Cleaning up records older than {hours} hours...')

    # Show statistics before cleanup
    click.echo('\n=== Before Cleanup ===')
    stats_before = get_database_statistics()
    for key, value in stats_before.items():
        click.echo(f'{key}: {value}')

    # Perform cleanup
    try:
        cleanup_stats = cleanup_old_data(hours=hours)
        click.echo(f'\n=== Cleanup Results ===')
        for key, value in cleanup_stats.items():
            click.echo(f'Deleted {value} {key}')

        # Show statistics after cleanup
        click.echo('\n=== After Cleanup ===')
        stats_after = get_database_statistics()
        for key, value in stats_after.items():
            click.echo(f'{key}: {value}')

        click.echo('\n✅ Database cleanup completed successfully!')
    except Exception as e:
        click.echo(f'\n❌ Error during cleanup: {str(e)}', err=True)


@click.command('cleanup-all')
@click.confirmation_option(prompt='Are you sure you want to delete ALL temporary data?')
@with_appcontext
def cleanup_all_command():
    """Clean up ALL temporary data (WARNING: Deletes all guest users and their data)"""
    click.echo('Cleaning up ALL temporary data...')

    try:
        cleanup_stats = cleanup_all_temporary_data()
        click.echo(f'\n=== Cleanup Results ===')
        for key, value in cleanup_stats.items():
            click.echo(f'Deleted {value} {key}')

        click.echo('\n✅ Full database cleanup completed successfully!')
    except Exception as e:
        click.echo(f'\n❌ Error during cleanup: {str(e)}', err=True)


@click.command('db-stats')
@with_appcontext
def database_stats_command():
    """Show database statistics"""
    click.echo('=== Database Statistics ===\n')

    stats = get_database_statistics()

    click.echo('Users:')
    click.echo(f'  Total: {stats["total_users"]}')
    click.echo(f'  Registered: {stats["registered_users"]}')
    click.echo(f'  Guest: {stats["guest_users"]}')
    click.echo(f'  Old Guest Users (>24h): {stats["old_guest_users"]}')

    click.echo('\nData:')
    click.echo(f'  Datasets: {stats["datasets"]} (old: {stats["old_datasets"]})')
    click.echo(f'  Chemicals: {stats["chemicals"]}')
    click.echo(f'  QSAR Models: {stats["qsar_models"]}')
    click.echo(f'  CV Results: {stats["cv_results"]}')
    click.echo(f'  Bioprofiles: {stats["bioprofiles"]}')
    click.echo(f'  k-DNN Predictions: {stats["kdnn_predictions"]}')

    click.echo('\nTasks:')
    click.echo(f'  Total: {stats["total_tasks"]}')
    click.echo(f'  Pending: {stats["pending_tasks"]}')
    click.echo(f'  Completed: {stats["completed_tasks"]}')
    click.echo(f'  Old Completed (>24h): {stats["old_tasks"]}')


@click.command('cleanup-sessions')
@click.option('--hours', default=24, help='Delete session folders older than this many hours')
@with_appcontext
def cleanup_sessions_command(hours):
    """Clean up old session folders from data/tmp"""
    click.echo(f'Cleaning up session folders older than {hours} hours...')
    click.echo(f'Scanning directory: {TMP_FILES_DIR}\n')

    # Show session folders before cleanup
    if os.path.exists(TMP_FILES_DIR):
        session_folders = [f for f in os.listdir(TMP_FILES_DIR) if f.startswith('session_') and os.path.isdir(os.path.join(TMP_FILES_DIR, f))]
        click.echo(f'=== Before Cleanup ===')
        click.echo(f'Total session folders: {len(session_folders)}')
        if session_folders:
            for folder in session_folders[:5]:  # Show first 5
                click.echo(f'  - {folder}')
            if len(session_folders) > 5:
                click.echo(f'  ... and {len(session_folders) - 5} more')
    else:
        click.echo(f'⚠️  Directory does not exist: {TMP_FILES_DIR}')

    # Perform cleanup
    try:
        cleanup_stats = cleanup_old_session_folders(hours=hours)
        click.echo(f'\n=== Cleanup Results ===')
        click.echo(f'Deleted {cleanup_stats["session_folders_deleted"]} session folders')

        # Show session folders after cleanup
        if os.path.exists(TMP_FILES_DIR):
            session_folders = [f for f in os.listdir(TMP_FILES_DIR) if f.startswith('session_') and os.path.isdir(os.path.join(TMP_FILES_DIR, f))]
            click.echo(f'\n=== After Cleanup ===')
            click.echo(f'Remaining session folders: {len(session_folders)}')

        click.echo('\n✅ Session folder cleanup completed successfully!')
    except Exception as e:
        click.echo(f'\n❌ Error during cleanup: {str(e)}', err=True)


@click.command('fix-stuck-jobs')
@click.option('--hours', default=1, help='Mark jobs as failed if stuck for this many hours')
@with_appcontext
def fix_stuck_jobs_command(hours):
    """Mark stuck jobs as failed"""
    from app.db_models import db, Job
    from datetime import datetime, timedelta

    click.echo(f'Checking for jobs stuck in "running" status for more than {hours} hour(s)...\n')

    cutoff_time = datetime.utcnow() - timedelta(hours=hours)

    # Find jobs that are stuck
    stuck_jobs = Job.query.filter(
        Job.status == 'running',
        Job.updated_at < cutoff_time
    ).all()

    if not stuck_jobs:
        click.echo('✅ No stuck jobs found!')
        return

    click.echo(f'Found {len(stuck_jobs)} stuck job(s):\n')

    for job in stuck_jobs:
        time_stuck = datetime.utcnow() - job.updated_at
        click.echo(f'  Job ID: {job.job_id}')
        click.echo(f'  Type: {job.job_type}')
        click.echo(f'  Last updated: {job.updated_at} ({time_stuck.total_seconds() / 3600:.1f} hours ago)')
        click.echo(f'  Marking as failed...')

        job.status = 'failed'
        job.updated_at = datetime.utcnow()

        click.echo('  ✅ Marked as failed\n')

    db.session.commit()
    click.echo(f'✅ Fixed {len(stuck_jobs)} stuck job(s)!')


def init_app(app):
    """Register CLI commands with the Flask app"""
    app.cli.add_command(cleanup_database_command)
    app.cli.add_command(cleanup_all_command)
    app.cli.add_command(database_stats_command)
    app.cli.add_command(cleanup_sessions_command)
    app.cli.add_command(fix_stuck_jobs_command)
