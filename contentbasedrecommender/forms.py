from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired,Email,EqualTo, ValidationError
from wtforms import ValidationError


from contentbasedrecommender.models import User

class ProfileForm(FlaskForm):
    submit = SubmitField('Null everything which was liked before!')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')


class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(),Email()])
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), EqualTo('confirm_password', message='Passwords Must Match!')])
    confirm_password = PasswordField('Confirm password', validators=[DataRequired()])
    submit = SubmitField('Register!')

    def validate_email(self, field):
        # Check if not None for that user email!
        email = User.query.filter_by(email=field.data).first()
        if email:
            raise ValidationError('Your email has been registered already!')

    def validate_username(self, field):
        # Check if not None for that username!
        user = User.query.filter_by(username=field.data).first()
        if user:
            raise ValidationError('Sorry, that username is taken!')
