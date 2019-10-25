from contentbasedrecommender import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime
from sqlalchemy.sql import func

# The user_loader decorator allows flask-login to load the current user
# and grab their id.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

class User(db.Model, UserMixin):
    # Create a table in the db
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key = True)
    what_user_likes= db.Column(db.String())
    authors_user_likes= db.Column(db.String())
    user_rating= db.relationship('Rating', backref='user')
    user_recommendation= db.relationship('Recommendations', backref='user')

    def __init__(self, id ):
        self.id = id

    def check_password(self,password):
        # https://stackoverflow.com/questions/23432478/flask-generate-password-hash-not-constant-output
        return check_password_hash(self.password_hash,password)
"""
item_author = db.Table('item_author',
                       db.Column('audio_id', db.Integer, db.ForeignKey('items.audio_id')),
                       db.Column('author_id', db.Integer, db.ForeignKey('authors.author_id'))
                       )
"""
class Item(db.Model):
    __tablename__ = 'items'
    audio_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(), nullable=False)
    url = db.Column(db.String(), nullable=False)
    small_picture_url = db.Column(db.String(), nullable=False)
    big_picture_url = db.Column(db.String(), nullable=False)
    vector = db.Column(db.String(), nullable=False)
    preprocess = db.Column(db.String(), nullable=False)
    nouns = db.Column(db.String(), nullable=False)
    teaser = db.Column(db.String(), nullable=False)
    broadcast_date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float(), nullable=False)
    authors = db.Column(db.String())
    item_rating= db.relationship('Rating', backref='item', lazy='dynamic')

    def __init__(self, audio_id, title, url,small_picture_url, big_picture_url, vector, preprocess, nouns, teaser, duration,broadcast_date, authors):
        self.audio_id = audio_id
        self.title = title
        self.url = url
        self.small_picture_url = small_picture_url
        self.big_picture_url = big_picture_url
        self.vector = vector
        self.preprocess = preprocess
        self.nouns = nouns
        self.teaser = teaser
        self.broadcast_date = broadcast_date
        self.duration = duration
        self.authors = authors


    def __repr__(self):
        return f"Titel: {self.title} hat die ID {self.audio_id}"

"""
class Author(db.Model):
    __tablename__ = 'authors'
    author_id = db.Column(db.Integer, primary_key=True)
    name= db.Column(db.String(), nullable=False, unique=True)
    authors = db.relationship('Item', secondary=item_author, backref=db.backref('authors', lazy='dynamic'))

    def __init__(self, name):
        self.name= name

    def __repr__(self):
        return f"Author: {self.name} hat die ID {self.author_id}"
"""

class Rating(db.Model):
    __tablename__ = 'ratings'
    rating_id= db.Column(db.Integer, primary_key=True)
    rating=db.Column(db.Integer, nullable=False)
    rating_date = db.Column(db.DateTime(timezone=True), default=func.now())
    item_id = db.Column(db.Integer, db.ForeignKey('items.audio_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __init__(self, rating):
        self.rating= rating
"""
class UserProfile(db.Model):
    __tablename__ = 'userprofile'
    userprofile_id =  db.Column(db.Integer, primary_key=True)
    profilelist=  db.Column(db.String)
    whatuserlikes= db.Column(db.String)
    whatuserdislikes= db.Column(db.String)
    authorlist=  db.Column(db.String)
    authorsuserlikes= db.Column(db.String)
    authorsuserdislikes= db.Column(db.String)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True)

    def __init__(self, profilelist, whatuserlikes, whatuserdislikes, user_id, authorlist, authorsuserlikes, authoruserdislikes):
        self.user_id= user_id
        self.profilelist= profilelist
        self.whatuserdislikes = whatuserdislikes
        self.whatuserlikes= whatuserlikes
        self.authorlist = authorlist
        self.authorsuserlikes = authorsuserlikes
        self.authorsuserdislikes = authoruserdislikes
"""

class Recommendations(db.Model):
    __tablename__ = 'recommendations'
    recommendation_id = db.Column(db.Integer, primary_key= True)
    rating_date = db.Column(db.DateTime, default=datetime.utcnow)
    audio_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(), nullable=False)
    url = db.Column(db.String(), nullable=False)
    explanations = db.Column(db.String(), nullable=False)
    teaser = db.Column(db.String(), nullable=False)
    broadcast_date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __init__(self, audio_id, title, url,explanations, teaser, broadcast_date, duration,  user_id):
        self.audio_id = audio_id
        self.title = title
        self.url = url
        self.explanations = explanations
        self.teaser = teaser
        self.broadcast_date = broadcast_date
        self. duration = duration
        self.user_id = user_id

class Vectors(db.Model):
    __tablename__ = 'vectors'
    word = db.Column(db.String(), primary_key= True, unique=True)
    vector = db.Column(db.String(), nullable=False)

    def __init__(self,word, vector):
        self.word = word
        self.vector = vector
