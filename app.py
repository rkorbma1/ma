from contentbasedrecommender import app, db
from flask import render_template, redirect, request, url_for, flash,  jsonify
from flask_login import login_user, login_required, logout_user, current_user
from contentbasedrecommender.models import User,Item, Rating, Recommendations
from contentbasedrecommender.forms import LoginForm, RegistrationForm
from sqlalchemy import create_engine, desc
from contentbasedrecommender.code.ProfileLearner import ProfileLearner
from contentbasedrecommender.code.Recommender import Recommender
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import os
import json
import random

"""
The app.py file orchestrates the different classes and their functions. First of all the file is connected to the database 
and a few columns of the data are loaded into the RAM, because every user needs them the same way. This preloading fastens 
the Recommender-Class heavily --> From over 1 minute of calculation to 8-15 seconds. The Disadvantage of this preloading is 
the heavy use of RAM. 
"""

base_dir = os.path.abspath(os.path.join(__file__, "../contentbasedrecommender"))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data.sqlite')))

db.create_all()
profile_learner = ProfileLearner()
recommender = Recommender()
df = pd.read_sql_table('items', engine, index_col='audio_id')
df['final_vectors'] = [json.loads(w) for w in df.vector]
df['final_preprocess'] = [json.loads(w) for w in df.preprocess]
df['final_nouns'] = [json.loads(w) for w in df.nouns]
authors =[]
for values in df['authors']:
    if(',' in values):
        authors.append(values.split(', '))
    else:
        authors.append([values])

df['list_authors'] = authors

def cosine_similarity(A):
    Anorm = A / np.linalg.norm(A, axis=-1)[:, np.newaxis]
    return linear_kernel(Anorm)

vectors = df['final_vectors'].tolist()
sim_matrix = cosine_similarity(np.asarray(vectors))
#sim_matrix = np.load('sim_matrix.npy')

"""
The routes of the app.py file start here. The functions are explained in short below. 
home() --> checks whether User has Userid of 1 or not and then directs the User to the admin-page or the normal player page. 
"""

@app.route('/')
@login_required
def home():
    if(current_user.id == 1):
        users = User.query.all()
        id_list =[]
        for user in users:
            id_list.append(user.id)
        return render_template('admin.html', id_list = id_list)
    else:
        return render_template('player.html')

"""
api_get_item()--> Checks, whether a User has some Recommendations waiting for him in the DB. If this is the case 
the function returns the first Recommended Item to the User with some Information. Else it returns information about a 
random entry. 
"""

@app.route('/get_items')
@login_required
def api_get_item():
    if(Recommendations.query.filter(Recommendations.user_id == current_user.id).first() is None):
        ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
        ratings_df = ratings_df[ratings_df.user_id == current_user.id]
        not_seen_items = df[~df.index.isin(ratings_df.item_id)]
        random_entry = not_seen_items.sample(n=1)
        url = random_entry.url.values[0]
        title = random_entry.title.values[0]
        small_pic = random_entry.small_picture_url.values[0]
        big_picture = random_entry.big_picture_url.values[0]
        id = int(random_entry.index.values[0])
        duration = int(random_entry.duration)
        zufall_random = random.random()
        if (zufall_random <= 0.2):
            explanation='Dieser Beitrag wurde zufällig aus allen anderen Beiträgen ausgewählt. Bitte bewerten Sie Beiträge, damit zugeschnittene Empfehlungen generiert werden können.'
        elif(zufall_random > 0.2 and zufall_random <= 0.4):
            explanation='Hier ist mal ein zufälliger Beitrag. Vielleicht gefällt er Ihnen ja.'
        elif (zufall_random > 0.4 and zufall_random <= 0.6):
            explanation = 'Ihnen wird nun ein zufälliger Beitrag angezeigt. Um Empfehlungen zu erhalten müssen Sie mehr liken.'
        elif (zufall_random > 0.6 and zufall_random <= 0.8):
            explanation = 'Dies ist ein zufällig ausgewählter Beitrag. Liken Sie mehr Beiträge, um zu zeigen was Sie mögen.'
        elif (zufall_random > 0.8 and zufall_random <= 1):
            explanation = 'Dieser Beitrag wurde volkommen zufällig ausgewählt. Um Empfehlungen zu generieren, müssen Sie angeben, welche Beiträge Sie mögen.'
        db.session.commit()
        return jsonify({
            'url': url,
            'title': title,
            'small_pic': small_pic,
            'big_pic': big_picture,
            'id': id,
            'explanation' : explanation,
            'length' : duration
        })
    else:
        rec = Recommendations.query.filter(Recommendations.user_id == current_user.id).first()
        item = Item.query.get(int(rec.audio_id))
        db.session.delete(rec)
        url = item.url
        title = item.title
        small_pic = item.small_picture_url
        big_picture = item.big_picture_url
        id = item.audio_id
        explanation = rec.explanations
        duration = int(rec.duration)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
        return jsonify({
            'url': url,
            'title': title,
            'small_pic': small_pic,
            'big_pic': big_picture,
            'id': id,
            'explanation': explanation,
            'length': duration
        })

"""
liking() --> API-call from the user which checks, whether a Rating exists for the Item in the database and changes the 
Rating to 1 for like. 
"""
@app.route('/liking', methods=['POST'])
@login_required
def liking():
    inde = request.form['index']
    value = 1
    rating_by_user = Rating(value)
    item = Item.query.get(int(inde))
    outcome = Rating.query.filter_by(user_id=current_user.id, item_id=int(inde))
    #notfalls hier noch sachen, wie item.audio_id usw einfügen...
    if(Rating.query.filter(Rating.user_id== current_user.id).filter(Rating.item_id == int(inde)).first() is None):
        current_user.user_rating.append(rating_by_user)
        item.item_rating.append(rating_by_user)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
        return (jsonify({'error': 'Missing Data!'}))
    else:
        outcome = Rating.query.filter(Rating.user_id== current_user.id).filter(Rating.item_id == int(inde)).first()
        outcome.rating = 1
        db.session.add(outcome)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
        return(jsonify({'error': 'Missing Data!'}))

"""
remove_recommendations_byID() --> API-Call from Client, which get index of clicked Items in WordCloud and deletes all the 
Items and their corresponding Recommendations from the DB. 
"""

@app.route('/remove_recommendations_byID', methods=['POST'])
@login_required
def remove_recommendations_byID():
    ids = json.loads(request.form['indices'])
    recommendations_df = pd.read_sql_table('recommendations', engine, index_col='recommendation_id')
    current_recommendations = recommendations_df[recommendations_df.user_id == current_user.id]
    explanations = current_recommendations['explanations'].values
    indices = current_recommendations.index.tolist()
    for vals in range(0, len(explanations)):
        if (str(ids) in explanations[vals]):
            deleteable_recommendation = Recommendations.query.filter_by(recommendation_id=indices[vals]).first()
            if (deleteable_recommendation != None):
                db.session.delete(deleteable_recommendation)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
    return (jsonify({'success': 'Alles hat geklappt!'}))

"""
disliking() --> Same as liking(), but now with value -1. 
"""

@app.route('/disliking', methods=['POST'])
@login_required
def disliking():
    inde = request.form['value']
    value= -1
    rating_by_user= Rating(value)
    item = Item.query.get(int(inde))
    print('Dieses Item wird gedisliked!', item.title)
    if ((Rating.query.filter(Rating.user_id == current_user.id).filter(Rating.item_id == int(inde)).first()) is None):
        current_user.user_rating.append(rating_by_user)
        item.item_rating.append(rating_by_user)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
        return (jsonify({'success': 'Alles hat geklappt!'}))
    else:
        outcome = Rating.query.filter(Rating.user_id == current_user.id).filter(Rating.item_id == int(inde)).first()
        outcome.rating = -1
        db.session.add(outcome)
        #recommender.get_recommendations(user_id=current_user.id)
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        finally:
            db.session.close()
        return (jsonify({'success': 'Alles hat geklappt!'}))

"""
recommendation() --> API-Call from client to generated recommendations through the Recommender-Class. For more Infos read
the Recommender-Class Description in Thesis. 
"""

@app.route('/number_recs' , methods=['GET'])
@login_required
def number_recs():
    if (Recommendations.query.filter(Recommendations.user_id == current_user.id).first() is None):
        return (jsonify({'number': 0}))
    else:
        recs = Recommendations.query.filter(Recommendations.user_id == current_user.id).all()
        number= len(recs)
        return (jsonify({'number': number}))

@app.route('/recommendations' , methods=['POST'])
@login_required
def recommendations():
    ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
    ratings_df = ratings_df[ratings_df.user_id == current_user.id]
    print(ratings_df.index)
    print('Anzahl der Ratings: ', len(ratings_df.index))
    liked_ratings_indices = ratings_df[ratings_df.rating == 1]
    if(len(liked_ratings_indices) >= 1):
        liked_items = df[df.index.isin(liked_ratings_indices.item_id)]
        not_seen_items = df[~df.index.isin(ratings_df.item_id)]
        indices = list(df.index)
        #Bis hierhin funktionierts!
        rec_df = recommender.get_recommendations(liked_items= liked_items, unseen_items=not_seen_items,indices=indices, sim_matrix = sim_matrix, user_id=current_user.id)
        if(isinstance(rec_df, int)):
            return (jsonify({'error': 'Es gab keine Empfehlungen!'}))
        else:
            rec_df.drop(columns=['small_picture_url','big_picture_url','vector','preprocess','nouns','final_vectors','authors','final_preprocess','final_nouns','list_authors'], inplace=True)
            rec_df['user_id'] = current_user.id
            print('Rec_df: ',rec_df.title)
            rec_df.to_sql('recommendations', con=engine, index_label='audio_id' ,if_exists='append')
            try:
                db.session.commit()
            except:
                db.session.rollback()
                raise
            finally:
                db.session.close()
            return (jsonify({'success': 'Alles hat super geklappt!'}))
    else:
        return(jsonify({'error': 'Es gab keine Empfehlungen!'}))

"""
last_eight_items() --> API-Call by Client every time the User wants to see next item. Gets the last eight rated Items
by the User from the DB and sends it to Client. Used for the List Modification.
"""

@app.route('/last_eight_items' , methods=['GET'])
@login_required
def last_eight_items():
    all_entries = Rating.query.filter_by(user_id=current_user.id).order_by(desc(Rating.rating_date)).all()
    last_eight_entries = all_entries[:8:]
    item_id_list = []
    rating_list = []
    for values in last_eight_entries:
        item_id_list.append(values.item_id)
        rating_list.append(values.rating)
    item_title_list = list(df.loc[item_id_list, :].title)
    print(item_title_list)
    return(jsonify({'id_list': json.dumps(item_id_list),
                    'rating_list': json.dumps(rating_list),
                    'title_list': json.dumps(item_title_list)}))

"""
get_user_profile() --> API-Call by Client to get generate the new User Profile if something new was liked or change through
a click on the WordCloud. Saves the Profile in DB. 
"""

@app.route('/get_user_profile' , methods=['POST','GET'])
@login_required
def get_user_profile():
    ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
    ratings_df = ratings_df[ratings_df.user_id == current_user.id]
    liked_ratings_indices = ratings_df[ratings_df.rating == 1]
    what_user_likes = profile_learner.get_user_preferences(items_df = df, liked_ratings_indices= liked_ratings_indices)
    authors_user_likes = profile_learner.get_user_author_preferences(items_df = df, liked_ratings_indices= liked_ratings_indices)
    user = User.query.get(current_user.id)
    user.what_user_likes = json.dumps(what_user_likes)
    user.authors_user_likes = json.dumps(authors_user_likes)
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    finally:
        db.session.close()
    return (jsonify({'success': 'Alles hat geklappt!'}))

"""
get_profile_data() --> API-Client Call to Return what tags a user likes and what authors a user likes. 
"""

@app.route('/get_profile_data' , methods=['POST','GET'])
@login_required
def get_profile_data():
    user = User.query.get(current_user.id)
    what_user_likes = user.what_user_likes
    authors_user_likes = user.authors_user_likes
    try:
        db.session.commit()
    except:
        db.session.rollback()
        raise
    finally:
        db.session.close()
    return jsonify({'tags': what_user_likes,
                    'authors': authors_user_likes})

"""
remove_likes() --> API-Call by Client. Get the titles and ids from liked items from a clicked word of the WordCloud. Returns
them to display them in the Pop-Up Modal in the User Profile.
"""

@app.route('/remove_likes' , methods=['POST','GET'])
@login_required
def remove_likes():
    word = request.form['word']
    ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
    ratings_df = ratings_df[ratings_df.user_id == current_user.id]
    liked_ratings_indices = ratings_df[ratings_df.rating == 1]
    liked_items = df[df.index.isin(liked_ratings_indices.item_id)]
    prep = liked_items['final_nouns'].tolist()
    indices = liked_items.index.tolist()
    item_ids= []
    item_title=[]
    for values in range(0, len(prep)):
        if word in prep[values]:
            deleting_entry = Rating.query.filter_by(item_id=indices[values]).first()
            item_ids.append(deleting_entry.item_id)
            item_title.append(df.loc[deleting_entry.item_id].title)
    return (jsonify({'item_ids': item_ids,
                     'item_titles' : item_title}))

"""
remove_authors() --> Same as function before just with authors and not with words. 
"""

@app.route('/remove_authors' , methods=['POST','GET'])
@login_required
def remove_authors():
    author = request.form['authors']
    ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
    ratings_df = ratings_df[ratings_df.user_id == current_user.id]
    liked_ratings_indices = ratings_df[ratings_df.rating == 1]
    liked_items = df[df.index.isin(liked_ratings_indices.item_id)]
    prep = liked_items['list_authors'].values
    indices = liked_items.index.tolist()
    item_ids= []
    item_title=[]
    for values in range(0, len(prep)):
        if(author in prep[values]):
            deleting_entry = Rating.query.filter_by(item_id=indices[values]).first()
            item_ids.append(deleting_entry.item_id)
            item_title.append(df.loc[deleting_entry.item_id].title)
    return (jsonify({'item_ids': item_ids,
                     'item_titles' : item_title}))

"""
remove_likes_byID() --> Removes the Items from DB by ID. Used in the User Profile as a API-Call from Client. Removes
Ratings of Items chosen and Recommendations associated with this item.
"""

@app.route('/remove_likes_byID' , methods=['POST','GET'])
@login_required
def remove_likes_byID():
    ids = json.loads(request.form['indices'])
    recommendations_df = pd.read_sql_table('recommendations', engine, index_col='recommendation_id')
    current_recommendations = recommendations_df[recommendations_df.user_id == current_user.id]
    explanations = current_recommendations['explanations'].values
    indices = current_recommendations.index.tolist()
    for values in ids:
        deleteable_rating = Rating.query.filter_by(item_id = values).first()
        for vals in range(0,len(explanations)):
            if (str(values) in explanations[vals]):
                deleteable_recommendation = Recommendations.query.filter_by(recommendation_id=indices[vals]).first()
                if (deleteable_recommendation != None):
                    db.session.delete(deleteable_recommendation)
        if(deleteable_rating != None):
            db.session.delete(deleteable_rating)
            try:
                db.session.commit()
            except:
                db.session.rollback()
                raise
            finally:
                db.session.close()
    return (jsonify({'success': 'Alles hat geklappt!'}))

"""
profile_page() --> Onclick on Profile-Icon redirects user to profile_page.html. 
"""

@app.route('/profile_page' , methods=['POST','GET'])
@login_required
def profile_page():
        return render_template('profile_page.html')

"""
get_username() --> API-Call from Client in User Profile. Gets the username and displays it. 
"""
"""
@app.route('/get_username' , methods=['POST','GET'])
@login_required
def get_username():
    last_item = Rating.query.filter(Rating.user_id == current_user.id).all()
    last_item_id = str(last_item[-1].item_id)
    last_item_db = Item.query.filter(Item.audio_id == last_item_id).first()
    last_item_url = last_item_db.big_picture_url
    user_df = pd.read_sql_table('users', engine)
    user = user_df[user_df.id == current_user.id]
    return (jsonify({'username': user.username.values [0],
                     'big_picture_url' : last_item_url}))
"""
"""
logout() --> Logs user out of application and displays the login-screen.
"""
"""
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You logged out!")
    return render_template('logout.html', user_id = current_user.id)
"""
"""
login() --> User can login into application with this function.
"""

@app.route('/login/<userid>', methods=['GET'])
def login(userid):
    try:
        id = int(userid)
        print(id)
        user = User.query.filter_by(id=id).first()
        print(user)
        #db.session.commit()
        if (user is not None):
            db.session.commit()
            print('Hallo ich bin hier')
            login_user(user)
            print('Logged in Succesfully!')
            next = request.args.get('next')
            if next == None or not next[0] == '/':
                next = url_for('home')
            return redirect(next)
        else:
            print('Hallo ich bin hier reingekommen!')
            new_user = User(id= id)
            print('Hallo ich bin hier reingekommen!')
            db.session.add(new_user)
            db.session.commit()
            print('Hallo ich bin hier reingekommen!')
            print('Hallo ich bin hier reingekommen!')
            user = User.query.filter_by(id=id).first()
            db.session.commit()
            login_user(user)
            print('Hallo ich bin hier reingekommen!')
            print('Logged in Succesfully!')
            next = request.args.get('next')
            if next == None or not next[0] == '/':
                next = url_for('home')
            return redirect(next)
    except:
        print('Falsche ID!')
        return render_template('error_login.html')

"""
register() --> User can register with this function. 
"""
"""
@app.route('/register', methods=['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data,
                    username= form.username.data,
                    password= form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Thanks for registration!')
        return redirect(url_for('login'))
    return render_template('register.html', form = form)
"""
"""
Sets Host and debug_mode. 
"""

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')

