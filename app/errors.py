# module for handing error pages

from flask import render_template

def not_found(e):
    return render_template('errors/404.html'), 404

def internal_error(e):
    return render_template('errors/500.html'), 500
