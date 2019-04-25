## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re
from collections import defaultdict 

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


        # without smoothing
        # estimate_emission_probabilites = defaultdict(lambda: defaultdict(float))
        # for (tag, words) in tag_word_frequency.items():
        #     for (word, count) in words.items():
        #         estimate_emission_probabilites[tag][word] = count / tags_frequency[tag]
        # end without smoothing

        # smoothing add-1
        self.len_words = len(words_frequency)
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        estimate_emission_probabilites = dict()
        for tag in tag_set:
            estimate_emission_probabilites[tag] = defaultdict(lambda: 1 / (tags_frequency[tag] + self.len_words))

        for (tag, words) in tag_word_frequency.items():
            for (word, count) in words.items():
                estimate_emission_probabilites[tag][word] = (count + 1) / (tags_frequency[tag] + self.len_words)
        # end smoothing add-1
        
        self.estimate_emission_probabilites = estimate_emission_probabilites

    
    def _estimate_transition_probabilites(self, annotated_sentences):
        tags_frequency  = defaultdict(int)
        tags_pair_frequency = defaultdict(lambda: defaultdict(int))
        estimate_transition_probabilites = defaultdict(lambda: defaultdict(float))
        
        for sentence in annotated_sentences:
            prev_tag = '<s>'
            tags_frequency[prev_tag] += 1
            for (word, tag) in sentence:
                tags_frequency[tag] += 1
                tags_pair_frequency[prev_tag][tag] += 1
                prev_tag = tag
            tags_pair_frequency[prev_tag]['<e>'] += 1

        for (prev_tag, tags) in tags_pair_frequency.items():
            for (tag, count) in tags.items():
                estimate_transition_probabilites[prev_tag][tag]= count / tags_frequency[prev_tag]
        
        self.estimate_transition_probabilites = estimate_transition_probabilites
                     
    def train(self, annotated_sentences):   
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
 
        return self 



    def predict(self, sentence):
        prediction = []

        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        viterbi = dict.fromkeys(tag_set, dict.fromkeys(sentence))
        back_pointer = dict.fromkeys(tag_set, dict.fromkeys(sentence))

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        first_word = sentence[0]

        for state in tag_set:
            viterbi[state][0] = estimate_transition_probabilites['<s>'][state] * estimate_emission_probabilites[state][first_word]
            back_pointer[state][0]='<s>'
        
        for i in range(1,len(sentence)):
            word = sentence[i]
            prev_word = sentence[i-1]
            _dict = {}
            for state in tag_set:
                for _state in tag_set:
                    _dict[_state] = viterbi[_state][i-1] * estimate_transition_probabilites[_state][state] * estimate_emission_probabilites[state][word]
                maximum = max(zip(_dict.values(), _dict.keys()))
                viterbi[state][i] = maximum[0]
                back_pointer[state][i]=maximum[1]

        word = sentence[len(sentence)-1]
        _dict = {}
        for _state in tag_set:
            _dict[_state] = viterbi[_state][len(sentence)-1] 
        maximum = max(zip(_dict.values(), _dict.keys()))
        best_path_prob = maximum[0]
        best_path_pointer = maximum[1]
    
        pointer = best_path_pointer
        index = len(sentence) - 1

        while pointer != '<s>':
            prediction = [pointer] + prediction
            pointer = back_pointer[pointer][index]
            index -= 1
        
        assert (len(prediction) == len(sentence))
        return prediction
            