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
        delta = 0.05
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
        tags_frequency  = defaultdict(int)
        tags_pair_frequency = defaultdict(lambda: defaultdict(int))
      
        for sentence in annotated_sentences:
            prev_tag = '<s>'
            tags_frequency[prev_tag] += 1
            for (word, tag) in sentence:
                tags_frequency[tag] += 1
                tags_pair_frequency[prev_tag][tag] += 1
                prev_tag = tag
            tags_pair_frequency[prev_tag]['<e>'] += 1

        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.05)
        delta = 0.05
        len_tags = len(tags_frequency)
        tag_set = '<s> ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        
        estimate_transition_probabilites = dict()
        for tag in tag_set:
             estimate_transition_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * len_tags))

        for (prev_tag, tags) in tags_pair_frequency.items():
             for (tag, count) in tags.items():
                estimate_transition_probabilites[prev_tag][tag] = (count + delta) / (tags_frequency[prev_tag] + delta * len_tags)
        
        self.estimate_transition_probabilites = estimate_transition_probabilites
                     
    def train(self, annotated_sentences):  
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
    
        return self 

    
    def _viterbi(self, observations, state_graph):
        result = []
        len_observations = len(observations)
        
        viterbi = defaultdict(dict)
        back_pointer = defaultdict(dict)

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        for state in state_graph:
            viterbi[state][0] = estimate_transition_probabilites['<s>'][state] * estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0]='<s>'
        
        for time_step in range(1,len_observations):
            observation = observations[time_step]
            _dict = {}
            for state in state_graph:
                for _state in state_graph:
                    _dict[_state] = viterbi[_state][time_step-1] * estimate_transition_probabilites[_state][state] * estimate_emission_probabilites[state][observation]
                max_prob = max(zip(_dict.values(), _dict.keys()))
                viterbi[state][time_step] = max_prob[0]
                back_pointer[state][time_step]=max_prob[1]
      
        _dict = {}
        for _state in state_graph:
            _dict[_state] = viterbi[_state][len_observations-1] 
        max_prob = max(zip(_dict.values(), _dict.keys()))
        best_path_prob = max_prob[0]
        best_path_pointer = max_prob[1]
    
        pointer = best_path_pointer
        index = len_observations - 1

        while pointer != '<s>':
            result = [pointer] + result
            pointer = back_pointer[pointer][index]
            index -= 1
        
        return result


    def predict(self, sentence):
      
        prediction = []
        
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        prediction = self._viterbi(sentence ,tag_set)

        assert (len(prediction) == len(sentence))
    
        return prediction
            