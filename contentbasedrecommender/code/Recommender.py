import numpy as np
import os
from sqlalchemy import create_engine
from scipy import spatial
from dppy.finite_dpps import FiniteDPP
from contentbasedrecommender.models import Vectors
import json
from sklearn.metrics.pairwise import linear_kernel
import heapq
import collections
from numpy import dot
from numpy.linalg import norm
import time
import random

class Recommender:

    def __init__(self):
        base_dir = os.path.abspath(os.path.join(__file__, "../.."))
        self.engine = create_engine('sqlite:///'+ os.path.abspath(os.path.join(base_dir,'data.sqlite')))
        #self.items_df['final_nouns'] = [json.loads(w) for w in self.items_df.nouns]
        #self.model = KeyedVectors.load('word2vec_german.model', mmap='r')

    def cosine_similarity(self, A):
        Anorm = A / np.linalg.norm(A, axis=-1)[:, np.newaxis]
        return linear_kernel(Anorm)

    def get_recommendations(self, unseen_items, liked_items, indices, sim_matrix, user_id):
        zufall = random.random()
        liked_items_ind = liked_items.index
        unseen_items_ind = unseen_items.index
        liked_iloc = np.asarray([indices.index(v) for v in liked_items_ind])
        not_seen_iloc = np.asarray([indices.index(v) for v in unseen_items_ind])
        result = sim_matrix[liked_iloc.ravel()]
        max_sim = np.amax(np.asarray(result), axis=0)
        max_not_seen = max_sim[not_seen_iloc]
        print(max_not_seen.shape)
        max_not_seen_lower = max_not_seen[(max_not_seen >= 0.5)]
        candidates = max_not_seen_lower.shape[0]
        print('Candidates: ', candidates)
        if(user_id % 2 == 1 and candidates >= 5):
            max_not_seen_ind = np.argpartition(max_not_seen, -10)[-10:]
            final_indices = unseen_items_ind[max_not_seen_ind]
            # time --> 0.003999233245849609
            # final_recommendations sind die richtigen ids für die Beiträge!
            recommendation_df = unseen_items.loc[unseen_items.index.isin(final_indices)]
            recommendation_df['explanations']=''
            return recommendation_df
        if(user_id % 2 == 1 and candidates <= 5):
            return 0
        if (user_id % 2 == 0 and candidates > 15):
            print('Kandidaten 1: ', candidates)
            print('Kandidaten: ', candidates)
            # Hier kann die Länge der ausgespuckten Liste verändert werden!
            if(candidates >= 30):
                max_not_seen_ind = np.argpartition(max_not_seen, -30)[-30:]
            else:
                max_not_seen_ind = np.argpartition(max_not_seen, -candidates)[-candidates:]
            final_indices = unseen_items_ind[max_not_seen_ind]
            print(final_indices.shape)
            #time --> 0.003999233245849609
            #final_recommendations sind die richtigen ids für die Beiträge!
            recommendation_df = unseen_items.loc[unseen_items.index.isin(final_indices)]
            vectors = recommendation_df['final_vectors'].tolist()
            final_recommendations = list(recommendation_df.index)
            #Phi = np.array(vectors)
            #L = Phi.dot(Phi.T)
            L = self.cosine_similarity(np.asarray(vectors))
            #print(dist_out_diversity)
            DPP = FiniteDPP('likelihood', **{'L': L})
            if(candidates >= 30 ):
                k = 10
            else:
                k = round(candidates/3)
            DPP.flush_samples()
            #DPP.sample_exact_k_dpp(size=k)
            #for _ in range(2000):
            DPP.sample_mcmc_k_dpp(size=k,**{'nb_iter':200})
            #print(DPP.projection)
            #Die final_diversity_list speichert, ob ein vorgeschlagenes Element in den ersten 10 recommendations ist oder nicht!
            list_of_samples = DPP.list_of_samples[0]
            det_sim_list=[]
            for values in list_of_samples:
                final_diversity_list_indices = [final_recommendations[i] for i in values]
                recommendation_diversity_df = unseen_items.loc[final_diversity_list_indices, :]
                vectors = recommendation_diversity_df['final_vectors'].tolist()
                dist_out_diversity = self.cosine_similarity(np.asarray(vectors))
                det_sim_list.append(np.linalg.det(dist_out_diversity))
            final_diversity_list = DPP.list_of_samples[0][np.argmax(det_sim_list)]
            #Indices der finalen diversität recommendations.
            final_diversity_list_indices = [final_recommendations[i] for i in final_diversity_list]
            recommendation_diversity_df = unseen_items.loc[final_diversity_list_indices, :]
            # time --> 0.03799939155578613
            # Get most similar documents...
            sim_list=[]
            for values in recommendation_diversity_df.final_vectors:
                most_similar_document =[]
                for vals in liked_items.final_vectors:
                    result = 1 - spatial.distance.cosine(values, vals)
                    most_similar_document.append(result)
                sim_list.append(most_similar_document)
            sim_doc_explanations = []
            sim_doc_index = []
            for values in range(0, len(sim_list)):
                max_doc = np.argmax(sim_list[values])
                titles = liked_items['title'].tolist()
                sim_doc_index.append(liked_items.index[max_doc])
                if(zufall <= 0.2):
                    sim_doc_explanations.append(
                        'Das ähnlichste, von Ihnen gemochte Dokument zu diesem vorgeschlagenen Beitrag hat den Titel: ' + str(
                        titles[max_doc]) + ' und die Audio_id: ' + str(liked_items.index[max_doc]) + '.')
                elif (zufall >= 0.2 and zufall <= 0.4):
                    sim_doc_explanations.append(
                        'Dieser Beitrag wird Ihnen empfohlen, da Sie den Beitrag "' + str(
                            titles[max_doc]) + '" mit der Audio ID: ' + str(liked_items.index[max_doc]) + ' mögen.')
                elif (zufall >= 0.4 and zufall <= 0.6):
                    sim_doc_explanations.append(
                        'Der gerade empfohlene Beitrag wurde für Sie aufgrund des gemochten Beitrags: "' + str(
                            titles[max_doc]) + '" mit der ID : ' + str(liked_items.index[max_doc]) + ' ausgewählt.')
                else:
                    sim_doc_explanations.append(
                        'Sie könnten diesen Titel mögen, da er eine gewisse Ähnlichkeit zu dem Beitrag "' + str(
                            titles[max_doc]) + '" besitzt, den Sie mögen. Dieser besitzt die ID: ' + str(liked_items.index[max_doc]) + '.')
            for values in range(0, len(recommendation_diversity_df)):
                words_1 = recommendation_diversity_df.final_nouns.values[values]
                id = int(sim_doc_index[values])
                words_2 = liked_items.final_nouns[id]
                sim_dict = {}
                for v in set(words_1):
                    for n in set(words_2):
                        word = str(v + ' ' + n)
                        v_result = Vectors.query.filter(Vectors.word == v).first()
                        v_vector = json.loads(v_result.vector)
                        n_result = Vectors.query.filter(Vectors.word == n).first()
                        n_vector = json.loads(n_result.vector)
                        sim_dict[word]= dot(v_vector, n_vector)/(norm(v_vector)*norm(n_vector))
                top_3_sim_dict = (
                    {key: value for key, value in sim_dict.items() if value in heapq.nlargest(3, sim_dict.values())})
                sorted_dict = collections.OrderedDict(top_3_sim_dict)
                zufall_2 = random.random()
                for k, v in sorted_dict.items():
                    words = k.split()
                    if v < 1 and v >= 0.5 and zufall_2 <= 0.33:
                        sim_doc_explanations[values] = sim_doc_explanations[
                                                   values] + ' Aus dem gemochten Beitrag ist das Wort ' + (
                                                       str(words[
                                                               0].capitalize()) + ' ähnlich zu dem Wort aus dem empfohlenen Beitrag ' + str(
                                                   words[1].capitalize()) + '.')
                    elif v < 1 and v >= 0.5 and zufall_2 >= 0.33 and zufall_2 <=0.66:
                        sim_doc_explanations[values] = sim_doc_explanations[
                                                   values] + ' Die zwei Worte ' + (
                                                       str(words[
                                                               0].capitalize()) + ' und ' + str(
                                                   words[1].capitalize()) + ' werden in den Beiträgen als ähnlich angesehen.')
                    if v < 1 and v >= 0.5 and zufall_2 >= 0.66 and zufall_2 <=1:
                        sim_doc_explanations[values] = sim_doc_explanations[
                                                   values] + ' Der gemochte und der vorgeschlagene Beitrag werden sich durch die in Ihnen vorkommenden Worte ' + (
                                                       str(words[
                                                               0].capitalize()) + ' und ' + str(
                                                   words[1].capitalize()) + ' als ähnlich angesehen.')
                    else:
                        if(' Die beiden Beiträge beinhalten das identische Wort: ' + (str(words[0].capitalize()) + '.') not in sim_doc_explanations[values]):
                            sim_doc_explanations[values] = sim_doc_explanations[
                                                       values] + ' Die beiden Beiträge beinhalten das identische Wort: ' + (
                                                           str(words[
                                                                   0].capitalize()) + '.')
            #time --> 0.0010020732879638672
            recommendation_diversity_df['explanations'] = sim_doc_explanations
            return recommendation_diversity_df
        else:
            return 0
