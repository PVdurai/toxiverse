import os
import uuid
from flask import Flask, session
from flask_login import LoginManager
from redis import Redis
import rq
from .db_models import db, User, migrate

# Configure TensorFlow to use CPU only (avoid CUDA errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.instance_path, 'toxpro.sqlite'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # SQLite configuration to prevent database locks and corruption
        SQLALCHEMY_ENGINE_OPTIONS={
            'connect_args': {
                'timeout': 30,  # Wait up to 30 seconds for lock
                'check_same_thread': False,  # Allow multiple threads
            },
            'pool_pre_ping': True,  # Verify connections before using
            'pool_recycle': 3600,  # Recycle connections every hour
        }
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_object('app.config.DockerConfig')
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Add the db setup
    db.init_app(app)
    migrate.init_app(app, db, render_as_batch=True)

    # Enable WAL mode for SQLite to prevent corruption
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous=NORMAL")  # Faster but still safe
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        cursor.close()

    # Set up login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.request_loader
    def load_user_from_request(request):
        # Use session_id from session storage
        session_id = session.get('session_id')
        print(f"[DEBUG] Session ID from session: {session_id}")
        
        if not session_id:
            session['session_id'] = str(uuid.uuid4())  # Create a new session_id if not present
            print(f"[DEBUG] Generated new session ID: {session['session_id']}")

        # Search for user by session_id
        user = User.query.filter_by(session_id=session['session_id']).first()
        print(f"[DEBUG] Looking up user with session ID: {session['session_id']}")
        
        if user:
            print(f"[DEBUG] User found: {user.id}")
        else:
            # Generate a unique email for the guest user by appending session_id
            unique_email = f"guest_{session['session_id']}@example.com"
            # If no user is found, create a new AnonymousUser
            user = User(session_id=session['session_id'],
                        username=f"guest_{session['session_id']}",  # Assign a temporary username
                        email=unique_email,  # You can change this as per your needs
                        password_hash="hashed_password")  # Use a default or temporary password hash (may need a better approach)
            db.session.add(user)
            try:
                db.session.commit()
                print("[DEBUG] No user found for this session ID.")
            except Exception as e:
                db.session.rollback()
                # Race condition - user was created by another request
                user = User.query.filter_by(session_id=session['session_id']).first()
                if not user:
                    # If still no user, re-raise the error
                    raise e
                print(f"[DEBUG] User created by concurrent request: {user.id}")
        
        return user

    # Register blueprints for different modules
    from . import auth
    app.register_blueprint(auth.bp)

    from . import cheminf
    app.register_blueprint(cheminf.bp)

    # Register main outline
    from . import toxpro
    app.register_blueprint(toxpro.bp)
    app.add_url_rule('/', endpoint='index')

    # Register database API module
    from . import database_api
    app.register_blueprint(database_api.bp)
    # Set up Redis for task queue (with graceful fallback if unavailable)
    # Use environment variables for flexible configuration
    # Default to 'redis' for Docker deployment, fallback to 'localhost' for local dev
    redis_host = os.getenv('REDIS_HOST', 'redis')
   
    redis_port = int(os.getenv('REDIS_PORT', 6379))

    try:
        app.redis = Redis(redis_host, redis_port, socket_connect_timeout=5)
        # Test connection
        app.redis.ping()
        app.task_queue = rq.Queue('toxpro-tasks', connection=app.redis, default_timeout=app.config.get('REDIS_JOB_TIMEOUT', 5000))
        print(f"✓ Redis connected successfully at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"⚠ Warning: Redis connection failed at {redis_host}:{redis_port}: {e}")
        print(f"⚠ Application will run in SYNCHRONOUS mode (no background tasks)")
        app.redis = None
        app.task_queue = None

    # Set up email configuration for error handling
    from .emails import mail
    mail.init_app(app)

    # Custom Jinja function to round floats
    def num_digit(flt, num):
        if flt is not None:
            return round(flt, num)

    app.jinja_env.globals.update(num_digit=num_digit)

    # Add template filter to convert UTC to US Central timezone (handles DST)
    def to_cst(utc_dt):
        """Convert UTC datetime to US Central timezone (CST/CDT with DST)."""
        if utc_dt is None:
            return ""

        try:
            import pytz
            # Use US/Central timezone which automatically handles DST
            central_tz = pytz.timezone('US/Central')

            # If datetime is naive (no timezone), assume it's UTC
            if utc_dt.tzinfo is None:
                utc_dt = pytz.utc.localize(utc_dt)

            # Convert to Central Time (automatically handles CST/CDT)
            central_dt = utc_dt.astimezone(central_tz)

            # Show CDT or CST based on DST
            tz_name = central_dt.strftime('%Z')  # Will be CST or CDT
            return central_dt.strftime(f'%Y-%m-%d %H:%M:%S {tz_name}')
        except ImportError:
            # Fallback if pytz not available (use simple offset)
            from datetime import timezone, timedelta
            cst_offset = timedelta(hours=-6)
            cst_tz = timezone(cst_offset, name='CST')

            if utc_dt.tzinfo is None:
                from datetime import timezone as tz
                utc_dt = utc_dt.replace(tzinfo=tz.utc)

            cst_dt = utc_dt.astimezone(cst_tz)
            return cst_dt.strftime('%Y-%m-%d %H:%M:%S CST')

    app.jinja_env.filters['to_cst'] = to_cst

    # Register error handlers for 404 and 500 errors
    from .errors import not_found, internal_error
    app.register_error_handler(404, not_found)
    app.register_error_handler(500, internal_error)

    # Set up logging for errors via email (only in production)
    import logging
    from logging.handlers import SMTPHandler

    if not app.debug:
        if app.config['MAIL_SERVER']:
            auth = None
            if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
                auth = (app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            secure = None
            if app.config['MAIL_USE_TLS']:
                secure = ()
            mail_handler = SMTPHandler(
                mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
                fromaddr='no-reply@' + app.config['MAIL_SERVER'],
                toaddrs=app.config['ADMINS'], subject='ToxPro Error',
                credentials=auth, secure=secure)
            mail_handler.setLevel(logging.ERROR)
            app.logger.addHandler(mail_handler)

    # Set up automatic database cleanup (runs every 24 hours)
    from .db_cleanup import schedule_cleanup_job
    schedule_cleanup_job(app)
    app.logger.info("Database cleanup scheduler initialized")

    # Register CLI commands for database management
    from . import cli_commands
    cli_commands.init_app(app)

    return app
