from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, AnonymousUserMixin, login_manager
from flask_migrate import Migrate
from werkzeug.security import check_password_hash, generate_password_hash
import click
from flask import current_app, g
from flask.cli import with_appcontext
from datetime import datetime
from sqlalchemy import MetaData
import redis
import rq
import jwt
from time import time

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Fix for unique constraints and foreign keys in Flask-migrate
naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}
db = SQLAlchemy(metadata=MetaData(naming_convention=naming_convention))
migrate = Migrate(db)


class User(db.Model, UserMixin):
    """ Main class to handle users """
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    last_login = db.Column(db.DateTime)
    user_created = db.Column(db.DateTime)
    admin = db.Column(db.Boolean, default=False)
    confirmed = db.Column(db.Boolean, default=False)
    confirmed_on = db.Column(db.DateTime)
    session_id = db.Column(db.String(128), unique=True, nullable=False)

    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic', cascade="all, delete-orphan")
    tasks = db.relationship('Task', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    kdnn_prediction = db.relationship('KDNNPrediction', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def launch_task(self, name, description, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('app.tasks.' + name, *args, **kwargs)
        task = Task(id=rq_job.get_id(), name=name, description=description, user=self)
        db.session.add(task)
        return task

    def get_tasks_in_progress(self):
        # Show last 10 tasks (both in progress and completed) for better visibility
        return Task.query.filter_by(user_id=self.id).order_by(Task.time_submitted.desc()).limit(10).all()

    def get_task_in_progress(self, name):
        return Task.query.filter_by(name=name, user=self, complete=False).first()

    def get_recent_jobs(self):
        """Get recent Job records (new model for Bioprofiler jobs)"""
        # Import Job here to avoid circular imports and use proper column reference
        # Using Job.created_at.desc() instead of string 'created_at desc' for SQLAlchemy 2.0 compatibility
        return Job.query.filter_by(user_id=self.id).order_by(Job.created_at.desc()).limit(10).all()

    def get_token(self, kind: str = 'reset_password', expires_in=600):
        return jwt.encode({kind: self.id, 'exp': time() + expires_in},
                          current_app.config['SECRET_KEY'], algorithm='HS256')

    @staticmethod
    def verify_token(token, kind):
        try:
            id = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])[kind]
        except:
            return
        return User.query.get(id)

    # ---------- Page-specific task checks ----------
    def has_running_tasks(self, names=None) -> bool:
        """
        Return True if the user has any incomplete tasks.
        If `names` is provided (str or list[str]), restrict to those task names.
        """
        q = Task.query.filter_by(user_id=self.id, complete=False)
        if names:
            if isinstance(names, str):
                names = [names]
            q = q.filter(Task.name.in_(names))
        return db.session.query(q.exists()).scalar()

    # Optional convenience wrappers for clarity
    def has_running_curator_tasks(self) -> bool:
        return self.has_running_tasks(['curate_chems'])

    def has_running_qsar_tasks(self) -> bool:
        return self.has_running_tasks(['build_qsar'])

    def has_running_import_tasks(self) -> bool:
        return self.has_running_tasks(['add_pubchem_data'])


class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False

    def is_advanced(self):
        return False


login_manager.anonymous_user = AnonymousUser


class Permission:
    GENERAL = 1
    ADVANCED = 2
    ADMIN = 4


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='user-dataset'))
    dataset_name = db.Column(db.String)
    type = db.Column(db.String)
    chemicals = db.relationship('Chemical', backref='dataset', lazy='dynamic', cascade="all, delete-orphan")
    qsar_models = db.relationship('QSARModel', backref='dataset', lazy='dynamic', cascade="all, delete-orphan")
    bioprofile = db.relationship('Bioprofile', backref='dataset', lazy='dynamic', cascade="all, delete-orphan")
    created = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'dataset_name', name='_user_dataset_uc'),)

    def get_chemicals(self):
        return Chemical.query.join(Dataset, Dataset.id == Chemical.dataset_id).filter(Dataset.id == self.id)


class Chemical(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inchi = db.Column(db.String)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    activity = db.Column(db.Float)
    compound_id = db.Column(db.String)

    def to_dict(self, structure_as_svg=False):
        result = {
            'Chemical': self.compound_id,
            'Activity': self.activity,
            'Structure': self.inchi
        }
        if structure_as_svg:
            result['Structure'] = self.get_svg()
        return result

    def get_svg(self):
        mol = Chem.MolFromInchi(self.inchi)
        d2d = rdMolDraw2D.MolDraw2DSVG(current_app.config['SVG_DISPLAY_WIDTH'],
                                       current_app.config['SVG_DISPLAY_HEIGHT'])
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()


class QSARModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='user-dataset'))
    name = db.Column(db.String)
    algorithm = db.Column(db.String)
    descriptors = db.Column(db.String)
    type = db.Column(db.String)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    sklearn_model = db.Column(db.BLOB)
    created = db.Column(db.DateTime, default=datetime.utcnow)
    cvresults = db.relationship('CVResults', backref='qsar_model', cascade="all, delete-orphan", uselist=False)


class CVResults(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    area_under_roc = db.Column(db.Float)
    cohens_kappa = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    specificity = db.Column(db.Float)
    correct_classification_rate = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    max_error = db.Column(db.Float)
    mean_squared_error = db.Column(db.Float)
    mean_absolute_percentage_error = db.Column(db.Float)
    pinball_score = db.Column(db.Float)
    qsar_model_id = db.Column(db.Integer, db.ForeignKey('qsar_model.id'))


class Bioprofile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='user-dataset'))
    name = db.Column(db.String)
    bioprofile = db.Column(db.BLOB)
    pca = db.Column(db.BLOB)
    heatmap = db.Column(db.BLOB)
    active_assay_table = db.Column(db.BLOB)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    created = db.Column(db.DateTime, default=datetime.utcnow)


class KDNNPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String)
    prediction = db.Column(db.BLOB)
    auc_score = db.Column(db.Float)
    auc_curve = db.Column(db.BLOB)
    created = db.Column(db.DateTime, default=datetime.utcnow)


class Task(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(128), index=True)
    description = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    complete = db.Column(db.Boolean, default=False)
    time_submitted = db.Column(db.DateTime, default=datetime.utcnow)
    time_completed = db.Column(db.DateTime)

    def get_rq_job(self):
        try:
            rq_job = rq.job.Job.fetch(self.id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def get_progress(self):
        job = self.get_rq_job()
        return job.meta.get('progress', 'Queued') if job is not None else 'Complete'


class Job(db.Model):
    """
    Track Bioprofiler and QSAR jobs with web-accessible status links
    """
    __tablename__ = 'job'
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(64), unique=True, nullable=False, index=True)  # Unique URL identifier
    session_id = db.Column(db.String(128), nullable=False)  # Links to session folder
    job_type = db.Column(db.String(50), nullable=False)  # 'bioprofiler' or 'qsar'
    status = db.Column(db.String(20), default='queued', nullable=False)  # 'queued', 'running', 'finished', 'failed'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = db.relationship('User', backref=db.backref('jobs', lazy='dynamic', cascade="all, delete-orphan"))

    def __repr__(self):
        return f'<Job {self.job_id} [{self.status}]>'

    def is_finished(self):
        return self.status == 'finished'

    def is_running(self):
        return self.status == 'running'

    def is_queued(self):
        return self.status == 'queued'

    def is_failed(self):
        return self.status == 'failed'


def create_db(overwrite=False):
    from . import create_app
    import os
    app = create_app()
    if overwrite and os.path.exists(os.path.join(app.instance_path, 'toxpro.sqlite')):
        os.remove(os.path.join(app.instance_path, 'toxpro.sqlite'))
    db.create_all(app=app)


@click.command('init-db')
@with_appcontext
def init_db_command():
    from . import create_app
    db.create_all(app=create_app())
    click.echo('Initialized the database.')
