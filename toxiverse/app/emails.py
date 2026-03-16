from flask_mail import Mail
from flask_mail import Message
from flask import current_app, render_template

mail = Mail()

def send_email(subject, sender, recipients, text_body, html_body):
    msg = Message(subject, sender=sender, recipients=recipients)
    msg.body = text_body
    msg.html = html_body
    mail.send(msg)

def send_password_reset_email(user):
    token = user.get_token(kind='reset_password')
    send_email('[ToxPro] Reset Your Password',
               sender=current_app.config['ADMINS'][0],
               recipients=[user.email],
               text_body=render_template('email/reset_password.txt',
                                         user=user, token=token),
               html_body=render_template('email/reset_password.html',
                                         user=user, token=token))

def send_registration_confirmation_email(user):
    token = user.get_token(kind='registration_conf', expires_in=3600)
    send_email('[ToxPro] Confirm Your Email',
               sender=current_app.config['ADMINS'][0],
               recipients=[user.email],
               text_body=render_template('email/confirm_email.txt',
                                         user=user, token=token),
               html_body=render_template('email/confirm_email.html',
                                         user=user, token=token))