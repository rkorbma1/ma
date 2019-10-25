    """
    
@app.route('/profile')
@login_required
def profile():
    user_profile = UserProfile.query.filter_by(user_id= current_user.id).first()
    if user_profile is None:
        user_preferences={}
        user_dispreferences = {}
        author_preferences ={}
        author_dispreferences ={}
        cloud = {}
        cloud_authors={}
    else:
        user_preferences = json.loads(user_profile.whatuserlikes)
        user_dispreferences = json.loads(user_profile.whatuserdislikes)
        author_preferences = json.loads(user_profile.authorsuserlikes)
        author_dispreferences = json.loads(user_profile.authorsuserdislikes)
        cloud =[]
        for k, v in user_preferences.items():
            cloud_dict = {}
            cloud_dict["x"]= k
            cloud_dict["value"]= v
            cloud_dict["category"]= "Ich mag dieses Stichwort!"
            cloud_dict["custom_field"]="Du scheinst das folgende Stichwort mit einer Gewichtung von: " + str(v) +" zu mögen. Höhere Werte zeigen, dass du ein Wort mehr magst."
            cloud.append(cloud_dict)
        for k, v in user_dispreferences.items():
            cloud_dict = {}
            cloud_dict["x"]= k
            cloud_dict["value"]= abs(v)
            cloud_dict["category"]= "Ich mag dieses Stichwort nicht!"
            cloud_dict["custom_field"]="Du scheinst das folgende Stichwort mit einer Gewichtung von: " + str(v) +" zu nicht zu mögen. Niedrigere Werte zeigen, dass du ein Wort weniger magst."
            cloud.append(cloud_dict)
        cloud_authors=[]
        for k, v in author_preferences.items():
            cloud_dict = {}
            cloud_dict["x"]= k
            cloud_dict["value"]= v
            cloud_dict["category"]= "Ich mag diesen Author!"
            cloud_dict["custom_field"]="Du scheinst den folgenden Author mit einer Gewichtung von: " + str(v) +" zu mögen. Höhere Werte zeigen, dass du ein Wort mehr magst."
            cloud_authors.append(cloud_dict)
        for k, v in author_dispreferences.items():
            cloud_dict = {}
            cloud_dict["x"]= k
            cloud_dict["value"]= abs(v)
            cloud_dict["category"]= "Ich mag diesen Author nicht!"
            cloud_dict["custom_field"]="Du scheinst den folgenden Author mit einer Gewichtung von: " + str(v) +" zu nicht zu mögen. Niedrige Werte zeigen, dass du ein Wort weniger magst."
            cloud_authors.append(cloud_dict)
    return render_template('profile.html', user_preferences=user_preferences, user_dispreferences=user_dispreferences, authorsuserlikes = author_preferences, authorsuserdislikes = author_dispreferences, cloud= cloud, cloud_authors=cloud_authors)

@app.route('/nullprofile', methods=['POST'])
@login_required
def nullprofile():
    Rating.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    UserProfile.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    Recommendations.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for('profile'))

    def newer_better(self, duration1, duration2):
        try:
            diff = duration1 - duration2
            days, seconds = diff.days, diff.seconds
            hours = days * 24 + seconds // 3600
            minutes = (seconds % 3600) // 60
            total_mins = (diff.days * 1440 + diff.seconds / 60)
            calculation = 1 / (1 + math.exp(total_mins - 60 * 30 * 60))
        except:
            calculation = 0
            # Anderer Ansatz: calculation = 0 --> Dann werden ältere Titel aber gar nicht berücksichtigt.
        return calculation

    def get_recommendations(self, user_id, item_tag, item_author):
        query="SELECT profilelist, authorlist, whatuserlikes, authorsuserlikes FROM userprofile WHERE user_id =" +str(user_id)
        user_profile_df = pd.read_sql_query(query, self.engine)
        if (len(user_profile_df.index)==1):
            dictionary = json.loads(user_profile_df['profilelist'].values[0])
            user_profile_tags= np.fromiter(dictionary.values(), dtype=float)
            user_profile_tags= np.expand_dims(user_profile_tags, axis=0)
            print(user_profile_tags.shape)
            print(item_tag.shape)
            tag_probabilities = np.einsum('ji,ki->kj', user_profile_tags, item_tag)
            author_dictionary = json.loads(user_profile_df['authorlist'].values[0])
            what_user_likes = json.loads(user_profile_df['whatuserlikes'].values[0])
            what_user_likes = list(what_user_likes.keys())
            authors_user_likes = json.loads(user_profile_df['authorsuserlikes'].values[0])
            authors_user_likes = list(authors_user_likes.keys())
            user_profile_author= np.fromiter(author_dictionary.values(), dtype=float)
            user_profile_author= np.expand_dims(user_profile_author, axis=0)
            author_probabilities = np.einsum('ji,ki->kj', user_profile_author, item_author)
            print(author_probabilities)
            print(author_probabilities.sum())
            probabilities_df = pd.DataFrame(data=tag_probabilities, columns=['tags'], index=item_tag.index)
            min_max_scaler = preprocessing.MinMaxScaler()
            probabilities_df['tags'] = min_max_scaler.fit_transform(probabilities_df[["tags"]])
            probabilities_df['authors'] = author_probabilities
            print(probabilities_df['authors'])
            items_df = pd.read_sql_table('items',self.engine, index_col='audio_id')
            date_right_now = datetime.datetime.now()
            new_better = []
            items_df['broadcast_date'] = pd.to_datetime(items_df['broadcast_date'])
            for values in items_df['broadcast_date']:
                new_better.append(self.newer_better(date_right_now, values))
            items_df['new_better'] = new_better
            query="SELECT rating, item_id FROM ratings WHERE user_id = " + str(user_id)
            ratings_df = pd.read_sql(query, self.engine)
            items_df['probabilities'] = 2* probabilities_df['tags'] + probabilities_df['authors'] + items_df['new_better']
            del items_df['authors']
            final_df = pd.merge(items_df, ratings_df, left_index=True, right_on='item_id', how='left')
            final_df = pd.merge(final_df, probabilities_df, left_on='item_id', right_index=True, how='left')
            final_df.rating.fillna(0, inplace=True)
            probabilities_data = final_df[final_df.rating == 0]
            print(probabilities_data.columns)
            between1and5 = preprocessing.MinMaxScaler(feature_range=(0,5))
            probabilities_data[['tags', 'new_better','authors','probabilities']] = between1and5.fit_transform(probabilities_data[['tags', 'new_better','authors','probabilities']])
            user = probabilities_data.nlargest(10, 'probabilities')
            user = user.reset_index(drop=True)
            query = "SELECT DISTINCT user_id FROM recommendations"
            unique_users = pd.read_sql_query(query, self.engine)
            unique_userlist = unique_users['user_id'].tolist()
            user['Similarities']=''
            #for values in user['tags']:
            #    print(values)
            #between1and5 = preprocessing.MinMaxScaler(feature_range=(0,5))
            #user[['tags', 'new_better','authors','probabilities']] = between1and5.fit_transform(user[['tags', 'new_better','authors','probabilities']])
            if user_id in unique_userlist:
                query = "DELETE FROM recommendations WHERE user_id = " +str(user_id)
                self.engine.execute(query)
            for index, row in user.iterrows():
                what_the_article_is_about = re.sub(r'[^\w\s]', ' ', row['keywords'])
                what_the_article_is_about = what_the_article_is_about.split()
                what_the_article_is_about = [item.lower() for item in what_the_article_is_about]
                tag_similarity =','.join(list(set(what_the_article_is_about) & set(what_user_likes)))
                rec = Recommendations(audio_id=row['item_id'], title= row['title'], url= row['url'], teaser= row['teaser'], broadcast_date=row['broadcast_date'], duration=row['duration'], keywords=row['keywords'], user_id=user_id, probability=row['probabilities'], tag_similarity=tag_similarity, tag_probability= row['tags'], author_probability= row['authors'], newness_probability= row['new_better'])
                db.session.add(rec)
                db.session.commit()
            return user

    def SVM(self, user_id):

            from sklearn import svm
            from sklearn.feature_extraction.text import TfidfTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import confusion_matrix

            query="SELECT rating, item_id FROM ratings WHERE user_id = " + str(user_id)
            ratings_df = pd.read_sql(query, self.engine)

            item_df = pd.read_sql_table('items', self.engine)

            item_df.drop_duplicates(subset=['title'], inplace=True)

            user_item_df = item_df.merge(ratings_df, left_on='audio_id', right_on='item_id', how='left')

            user_item_df['rating'].fillna(0, inplace=True)

            date_right_now = datetime.datetime.now()
            new_better = []
            user_item_df['broadcast_date'] = pd.to_datetime(user_item_df['broadcast_date'])
            for values in user_item_df['broadcast_date']:
                new_better.append(self.newer_better(date_right_now, values))
            user_item_df['new_better'] = new_better

            query = "SELECT profilelist, authorlist, whatuserlikes, authorsuserlikes FROM userprofile WHERE user_id =" + str(
                user_id)
            user_profile_df = pd.read_sql_query(query, self.engine)

            tfidf_transformer = TfidfTransformer()
            count_vect = CountVectorizer()

            x_train = list(user_item_df.loc[user_item_df['rating'] != 0, 'keywords'])
            y_train = list(user_item_df.loc[user_item_df['rating'] != 0, 'rating'])
            ind = list(user_item_df.loc[user_item_df['rating'] != 0].index)

            x_test = list(user_item_df.loc[user_item_df['rating'] == 0, 'keywords'])
            y_test = list(user_item_df.loc[user_item_df['rating'] == 0, 'rating'])
            ind_test = list(user_item_df.loc[user_item_df['rating'] == 0].index)

            what_user_likes = ' '.join(x_train)
            what_user_likes = re.sub(r'[^\w\s]', ' ', what_user_likes)

            text_sgd = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', svm.SVC(kernel='linear', probability=True)),
            ])

            text_sgd.fit(x_train, y_train)
            predicted = text_sgd.predict(x_test)
            predicted_proba = text_sgd.predict_proba(x_test)[:, 1]

            SVM_df = pd.DataFrame(index=ind_test, data=predicted_proba, columns=['SVM'])

            value_counts = pd.value_counts(SVM_df['SVM'])

            user_item_df = pd.concat([user_item_df, SVM_df], axis=1, sort=False)

            user_item_df['probabilities'] = user_item_df['SVM'] + user_item_df['new_better']
            SVM_predictions_df = user_item_df.nlargest(50, 'SVM')
            test_diversity = user_item_df.nlargest(10, 'SVM')
            result = self.diversity(SVM_predictions_df, test_diversity, user_item_df)

            i = 0
            """
            """
            for index, row in result.iterrows():
                what_the_article_is_about = re.sub(r'[^\w\s]', ' ', row['keywords'])
                what_the_article_is_about = what_the_article_is_about.split()
                i += 1
                print(str(i) + '.Recommendation: ')
                print(index)
                print(row['title'])
                print(x_train[0])
                print(row['keywords'])
                print('Similarities: ', (list(set(what_the_article_is_about) & set(what_user_likes.split()))))
                print(row['probabilities'])

            for index, row in test_diversity.iterrows():
                what_the_article_is_about = re.sub(r'[^\w\s]', ' ', row['keywords'])
                what_the_article_is_about = what_the_article_is_about.split()
                i += 1
                print(str(i) + '.Recommendation: ')
                print(index)
                print(row['title'])
                print(x_train[0])
                print(row['keywords'])
                print('Similarities: ', (list(set(what_the_article_is_about) & set(what_user_likes.split()))))
                print(row['probabilities'])
            """
            """
    def naive_bayes(self, user_id):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import confusion_matrix

        query = "SELECT rating, item_id FROM ratings WHERE user_id = " + str(user_id)
        ratings_df = pd.read_sql(query, self.engine)

        item_df = pd.read_sql_table('items', self.engine)

        print(item_df.shape)

        item_df.drop_duplicates(subset=['title'], inplace=True)

        print(item_df.shape)

        print(item_df.columns)

        user_item_df = item_df.merge(ratings_df, left_on='audio_id', right_on='item_id', how='left')

        print(user_item_df.columns)

        user_item_df['rating'].fillna(0, inplace=True)

        for index, row in user_item_df.iterrows():
            if (row['rating'] == 1.0 or row['rating'] == -1.0):
                print(row['audio_id'])
                print(row['rating'])
                print(row['keywords'])

        tfidf_transformer = TfidfTransformer()
        count_vect = CountVectorizer()

        x_train = list(user_item_df.loc[user_item_df['rating'] != 0, 'keywords'])
        y_train = list(user_item_df.loc[user_item_df['rating'] != 0, 'rating'])
        ind = list(user_item_df.loc[user_item_df['rating'] != 0].index)

        x_test = list(user_item_df.loc[user_item_df['rating'] == 0, 'keywords'])
        y_test = list(user_item_df.loc[user_item_df['rating'] == 0, 'rating'])
        ind_test = list(user_item_df.loc[user_item_df['rating'] == 0].index)

        what_user_likes = ' '.join(x_train)
        what_user_likes = re.sub(r'[^\w\s]', ' ', what_user_likes)

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        text_clf.fit(x_train, y_train)
        predicted = text_clf.predict(x_test)
        predicted_proba = text_clf.predict_proba(x_test)[:, 1]

        NB_df = pd.DataFrame(index=ind_test, data=predicted_proba, columns=['NB'])

        value_counts = pd.value_counts(round(NB_df['NB'],4))

        print(value_counts)

        value_counts.plot.bar()

        plt.show()

        user_item_df = pd.concat([user_item_df, NB_df], axis=1, sort=False)
        NB_predictions_df = user_item_df.nlargest(10, 'NB')

        i=0

        for index, row in NB_predictions_df.iterrows():
            what_the_article_is_about = re.sub(r'[^\w\s]', ' ', row['keywords'])
            what_the_article_is_about = what_the_article_is_about.split()
            i+=1
            print(str(i) + '.Recommendation: ')
            print(index)
            print(row['title'])
            print(x_train[0])
            print(row['keywords'])
            print('Similarities: ', (list(set(what_the_article_is_about) & set(what_user_likes.split()))))
            print(row['NB'])

    def diversity(self, probabilities, test_diversity, item_df):

        tag_count_vectorizer = CountVectorizer(tokenizer=tokenize_tags)

        tags = tag_count_vectorizer.fit_transform(item_df['keywords'])

        count_vector_tags = tags.toarray()

        #print(tag_count_vectorizer.get_feature_names())

        count_vector_df_tags = pd.DataFrame(count_vector_tags, columns=tag_count_vectorizer.get_feature_names(), index=item_df.audio_id)

        diversity_recommendations = []
        index_list = list(probabilities.audio_id)

        diversity_recommendations.append(index_list[0])

        del index_list[0]

        while(len(diversity_recommendations) < 10):
            comp_list = []
            for i in range(0,len(index_list)):
                #print(int(index_list[i]))
                comp = 0
                for v in diversity_recommendations:
                    a = count_vector_df_tags.loc[int(index_list[i])]
                    b = count_vector_df_tags.loc[int(v)]
                    #print('Cosine Similarity between ',index_list[i] ,' und ', v, 'ist: ' ,np.inner(a, b) / (norm(a) * norm(b)))
                    comp += np.inner(a, b) / (norm(a) * norm(b))
                comp_list.append(comp)
            #print('Comparison List: ', comp_list)
            #print('Length Comparison List: ', len(comp_list))
            smallest_index = comp_list.index(min(comp_list))
            #print(smallest_index)
            diversity_recommendations.append(index_list[smallest_index])
            del index_list[smallest_index]

        cos=0

        for values in diversity_recommendations:
            for vals in diversity_recommendations:
                a = count_vector_df_tags.loc[int(values)]
                b = count_vector_df_tags.loc[int(vals)]
                if (vals != values):
                    cos += np.inner(a, b) / (norm(a) * norm(b))
        #print('Die komplette Diversity der verbesserten Liste ist: ', cos*(1/2))

        index_list_test = test_diversity.audio_id

        test_cos = 0

        for values in index_list_test:
            for vals in index_list_test:
                a = count_vector_df_tags.loc[int(values)]
                b = count_vector_df_tags.loc[int(vals)]
                if (vals != values):
                    test_cos += np.inner(a, b) / (norm(a) * norm(b))

        #print('Die komplette Diversity der normalen Liste ist: ', test_cos*(1/2))

        #print('Die Diversität der Liste wurde um ',(test_cos*(1/2) - cos*(1/2)),'verringert.')

        #print('Die Diversität der Liste wurde um ',(1-((cos*(1/2))/(test_cos*(1/2))))*100,'% verringert.')

        #print(item_df.columns)

        item_df.set_index('audio_id', inplace=True)

        diversity_recommendations = item_df.loc[diversity_recommendations,:]

        #print(diversity_recommendations)

        return diversity_recommendations

#TODO: Neuere Likes sind besser! In der anderen Datei!
#TODO: Neuere Beiträge sind besser!
    """
"""
profile_learner = ProfileLearner()
recommender = Recommender()
res, authors= profile_learner.create_user_item_matrix()
user= recommender.get_recommendations(1,res, authors)
"""
"""
p1 = Recommender()
start = time.time()
p1.SVM(6)
end = time.time()
print(end - start)
"""