## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re
from collections import deque
from collections import defaultdict 
import heapq

## project supplied imports
from submission_specs.SubmissionSpec12 import SubmissionSpec12


class Submission(SubmissionSpec12):


    ''' a contrived poorely performing solution for question one of this Maman '''
    
    def _estimate_emission_probabilites(self, annotated_sentences):
        words_frequency =  defaultdict(int)
        tags_frequency  = defaultdict(int)
        tag_word_frequency = defaultdict(lambda: defaultdict(int))
       
        for sentence in annotated_sentences:
            for (word, tag) in sentence:
                words_frequency[word] += 1
                tags_frequency[tag] += 1
                tag_word_frequency[tag][word] += 1

        # calculate estimate_emission_probabilites with smoothing add-delta (delta = 0.05)
        delta = 0.001
        len_words = len(words_frequency)
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        estimate_emission_probabilites = dict()
        for tag in tag_set:
            estimate_emission_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * len_words))

        for (tag, words) in tag_word_frequency.items():
            for (word, count) in words.items():
                estimate_emission_probabilites[tag][word] = (count + delta) / (tags_frequency[tag] + delta * len_words)
      
        self.estimate_emission_probabilites = estimate_emission_probabilites




    
    def _estimate_transition_probabilites(self, annotated_sentences):
        self.tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        BOS = '<s>'
        EOS = '<e>'
        tags_frequency  = defaultdict(int)
        tags_pair_frequency = defaultdict(lambda: defaultdict(int))
      
        for sentence in annotated_sentences:
            prev_tag = BOS
            tags_frequency[prev_tag] += 1
            for (word, tag) in sentence:
                tags_frequency[tag] += 1
                tags_pair_frequency[prev_tag][tag] += 1
                prev_tag = tag
           
        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.001)
        delta = 0.001
        V = len(tags_frequency)
        tag_set = [BOS] + self.tag_set 
        
        estimate_transition_probabilites = dict()

        for tag in tag_set:
             # set default value for P(tag|prev_prev_tag) for each tag
             estimate_transition_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * V))

        for (prev_tag, tags) in tags_pair_frequency.items():
             for (tag, count) in tags.items():
                estimate_transition_probabilites[prev_tag][tag] = (count + delta) / (tags_frequency[prev_tag] + delta * V)
        
        self.estimate_transition_probabilites = estimate_transition_probabilites


    def train(self, annotated_sentences):  
        #self._test()

        self.x = 1

       
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
    
        return self 


   

    def _viterbi_2(self, observations, state_graph, k=3):
        BOS = '<s>'
        EOS = '<e>'

        result = []
        len_observations = len(observations)
        viterbi =  defaultdict(lambda: defaultdict(dict))
        back_pointer = defaultdict(lambda: defaultdict(dict))
        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        # initialization step
        for state in state_graph:
            viterbi[state][0][0] = estimate_transition_probabilites[BOS][state] * estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0][0]=BOS
        

        # recursion step
        for time_step in range(1,len_observations):
            observation = observations[time_step]
            for state in state_graph:
                heap = []
                for prev_state in state_graph:
                    prob = viterbi[prev_state][time_step-1][0] * estimate_transition_probabilites[prev_state][state] * estimate_emission_probabilites[state][observation]
                    heapq.heappush(heap, (-prob, prev_state, 0))
                
                count = 0
                while len(heap) > 0 and count < k:
                    curr = heapq.heappop(heap)
                    prob = -curr[0]
                    prev_state = curr[1]
                    index = curr[2]
                    viterbi[state][time_step][count] = prob
                    back_pointer[state][time_step][count] = (prev_state, index)
                    if(index + 1 < len(viterbi[prev_state][time_step-1])):
                        prob = viterbi[prev_state][time_step-1][index + 1] * estimate_transition_probabilites[prev_state][state] * estimate_emission_probabilites[state][observation]
                        heapq.heappush(heap, (-prob, prev_state, index + 1))
                    count += 1

        # termination step
        heap = []

        for state in state_graph:
             prob = viterbi[state][len_observations-1][0] 
             heapq.heappush(heap, (-prob, state, 0))

        count = 0        
        while len(heap) > 0 and count < k:
             curr = heapq.heappop(heap)
             prob = -curr[0]
             state = curr[1]
             index = curr[2]
             best_path_prob = prob
             best_path_pointer = (state, index)

             pointer = best_path_pointer
             time_step = len_observations - 1
             path = []

             while pointer != BOS:
                 path = [pointer[0]] + path
                 pointer = back_pointer[pointer[0]][time_step][pointer[1]]
                 time_step -= 1
            
             result = result + [path]    

             if(index + 1 < len(viterbi[state][len_observations-1])):
                prob = viterbi[state][len_observations-1][index + 1]
                heapq.heappush(heap, (-prob, state, index + 1))

             count += 1

        return result
    
    
        




  
    
   


    def predict(self, sentence):
      
        prediction = []
        
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        # prediction = self._viterbi2(sentence ,tag_set)
        # assert (len(prediction) == len(sentence))
        prediction = self._viterbi_2(sentence ,tag_set)

        assert (len(prediction) == 3)

        for predict in prediction:
            assert (len(predict) == len(sentence))

        return prediction
            