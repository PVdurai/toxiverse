import os


class DockerConfig(object):
    # these should be declared in docker-environment.env
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or "smtp.gmail.com"
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    SECRET_KEY = os.environ.get('SECRET_KEY') or "dev"
    ADMINS = ['toxproemail@gmail.com', 'russodanielp@gmail.com']
    ENV = os.environ.get('FLASK_ENV') or "production"
    DEBUG = int(os.environ.get('FLASK_DEBUG', 0))

    SVG_DISPLAY_WIDTH = 150
    SVG_DISPLAY_HEIGHT = 75

    REDIS_JOB_TIMEOUT = 1200


from dotenv import load_dotenv

load_dotenv()


class Config:
    MASTER_DB_FILE = os.getenv('MASTER_DB_FILE')
    BIOASSAYS = os.getenv('BIOASSAYS')
    BIOPROFILE_DIR = os.getenv('BIOPROFILE')
    BIOASSAY_OUT = os.getenv('BIOASSAY_OUT')
    DATA_DIR = os.getenv('DATA')
