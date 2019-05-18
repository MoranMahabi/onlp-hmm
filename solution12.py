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

    # def deleted_interpolation(self, annotated_sentences):
    #     unigrms  = defaultdict(int)
    #     bigrams =  defaultdict(int)
    #     trigrams =  defaultdict(int)

    #     for sentence in annotated_sentences:
    #         prev_prev_tag = '<s>'
    #         prev_tag = '<s>'
    #         unigrms[('<s>')] += 1
    #         bigrams[('<s>', '<s>')] += 1

    #         for (word, tag) in sentence:
    #             unigrms[(tag)] +=1
    #             bigrams[(prev_tag, tag)] += 1
    #             trigrams[(prev_prev_tag, prev_tag, tag)] += 1 
    #             prev_prev_tag = prev_tag
    #             prev_tag = tag
    #         bigrams[(prev_tag, '<e>')] += 1
    #         trigrams[(prev_prev_tag, prev_tag, '<e>')] += 1
        
    #     lambda1 = lambda2 = lambda3 = 0
    #     for (t1, t2, t3) in trigrams.items():
    #           v = trigrams[(t1, t2, t3)]
    #           if v > 0:
    #               try:
    #                   c1 = float(v-1)/(bigrams[(t1, t2)]-1)
    #                   except ZeroDivisionError:
    #             c1 = 0
    #         try:
    #             c2 = float(bigrams[(t2, t3)]-1)/(unigrms[(t2)]-1)
    #         except ZeroDivisionError:
    #             c2 = 0
    #         try:
    #             c3 = float(unigrams[(t3)]-1)/(len(unigrams)-1)
    #         except ZeroDivisionError:
    #             c3 = 0

    #         k = np.argmax([c1, c2, c3])
    #         if k == 0:
    #             lambda3 += v
    #         if k == 1:
    #             lambda2 += v
    #         if k == 2:
    #             lambda1 += v

    # weights = [lambda1, lambda2, lambda3]
    # norm_w = [float(a6)/sum(weights) for a in weights]
    # return norm_w

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
        delta = 0.1
        len_words = len(words_frequency)
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        estimate_emission_probabilites = dict()
        for tag in tag_set:
            estimate_emission_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * len_words))

        for (tag, words) in tag_word_frequency.items():
            for (word, count) in words.items():
                estimate_emission_probabilites[tag][word] = (count + delta) / (tags_frequency[tag] + delta * len_words)
      
        self.estimate_emission_probabilites = estimate_emission_probabilites



      
    # def _estimate_transition_probabilites(self, annotated_sentences):
    #     weights = self.deleted_interpolation(annotated_sentences)
    #     tags_frequency  = defaultdict(int)
    #     tags_pair_frequency = defaultdict(lambda: defaultdict(int))
    #     tags_frequency_0 = defaultdict(int)
      
    #     for sentence in annotated_sentences:
    #         prev_prev_tag = '<s>'
    #         prev_tag = '<s>'
    #         tags_frequency[(prev_prev_tag, prev_tag)] += 1
    #         for (word, tag) in sentence:
    #             tags_frequency_0[tag] += 1
    #             tags_frequency[(prev_tag, tag)] += 1
    #             tags_pair_frequency[(prev_prev_tag, prev_tag)][tag] += 1 
    #             prev_prev_tag = prev_tag
    #             prev_tag = tag
    #         tags_frequency[(prev_tag, '<e>')] += 1
    #         tags_pair_frequency[(prev_prev_tag, prev_tag)]['<e>'] += 1

    #     # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.05)
    #     delta = 0.05
    #     len_tags = len(tags_frequency_0) + 1
    #     tag_set = '<s> ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        
    #     estimate_transition_probabilites = dict()
    #     for prev_tag in tag_set: 
    #          for tag in tag_set:
    #             if(tag == '<s>' and prev_tag != '<s>'):
    #                 continue
     
    #             estimate_transition_probabilites[(prev_tag, tag)] = defaultdict(lambda: delta / (tags_frequency[(prev_tag, tag)] + delta * len_tags))
                
    #     for ((prev_prev_tag, prev_tag), tags) in tags_pair_frequency.items():
    #          for (tag, count) in tags.items():
    #             estimate_transition_probabilites[(prev_prev_tag, prev_tag)][tag] = (count + delta) / (tags_frequency[(prev_prev_tag, prev_tag)] + delta * len_tags)
                       
    #     self.estimate_transition_probabilites = estimate_transition_probabilites
    

    
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
            #tags_pair_frequency[prev_tag]['<e>'] += 1

        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.05)
        delta = 0.1
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
        #self._test()

        self.x = 1

       
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
    
        return self 

    def _test(self):
          x = [2,5,1]
          print(x)
          heapq.heapify(x)
          print(x)
          print(heapq.heappop(x))
          print(heapq.heappop(x))
          print(heapq.heappop(x))
          print(x)
         
          viterbi =  defaultdict(lambda: defaultdict(dict))
          viterbi[0][0][0] = 5.5
          viterbi[0][0][4] = 5.5
          print(len(viterbi[0][0]))


   

    def _viterbi_2(self, observations, state_graph, k=3):
        result = []
        len_observations = len(observations)
        
        viterbi =  defaultdict(lambda: defaultdict(dict))
        back_pointer = defaultdict(lambda: defaultdict(dict))

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        for state in state_graph:
            viterbi[state][0][0] = estimate_transition_probabilites['<s>'][state] * estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0][0]='<s>'
        
        for time_step in range(1,len_observations):
            observation = observations[time_step]
            for state in state_graph:
                heap = []
                for _state in state_graph:
                    prob = viterbi[_state][time_step-1][0] * estimate_transition_probabilites[_state][state] * estimate_emission_probabilites[state][observation]
                    heapq.heappush(heap, (-prob, _state, 0))
                
                index = 0
                while len(heap) > 0 and index < k:
                    curr = heapq.heappop(heap)
                    prob = -curr[0]
                    _state = curr[1]
                    _index = curr[2]
                    viterbi[state][time_step][index] = prob
                    back_pointer[state][time_step][index] = (_state, _index)
                    if(_index + 1 < len(viterbi[_state][time_step-1])):
                        prob = viterbi[_state][time_step-1][_index + 1] * estimate_transition_probabilites[_state][state] * estimate_emission_probabilites[state][observation]
                        heapq.heappush(heap, (-prob, _state, _index + 1))
                    index += 1

        
        heap = []
        for _state in state_graph:
             prob = viterbi[_state][len_observations-1][0] 
             heapq.heappush(heap, (-prob, _state, 0))

        count = 0        
        while len(heap) > 0 and count < k:
             curr = heapq.heappop(heap)
             prob = -curr[0]
             _state = curr[1]
             _index = curr[2]
             best_path_prob = prob
             best_path_pointer = (_state, _index)

             pointer = best_path_pointer
             index = len_observations - 1
             path = []

             while pointer != '<s>':
                 path = [pointer[0]] + path
                 pointer = back_pointer[pointer[0]][index][pointer[1]]
                 index -= 1
            
             result = result + [path]    

             if(_index + 1 < len(viterbi[_state][len_observations-1])):
                prob = viterbi[_state][len_observations-1][_index + 1]
                heapq.heappush(heap, (-prob, _state, _index + 1))
             count += 1

        #print(observations)
        #print(result)
        return result
    
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




    def _viterbi_3(self, observations, state_graph):

        result = []
        len_observations = len(observations)

        viterbi =  defaultdict(lambda: defaultdict(dict))
        back_pointer = defaultdict(lambda: defaultdict(dict))
        
        viterbi = defaultdict(dict)
        back_pointer = defaultdict(dict)

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        for state in state_graph:
            viterbi[('<s>', state)][0] = estimate_transition_probabilites[('<s>','<s>')][state] * estimate_emission_probabilites[state][observations[0]]
            back_pointer[('<s>', state)][0]='<s>'

        for time_step in range(1,len_observations):
            observation = observations[time_step]
           
            for state in state_graph:
                for prev_state in state_graph:
                    _dict = {}
                    state_graph_prev = state_graph
                    if(time_step == 1):
                        state_graph_prev = ['<s>']
                    for prev_prev_state in state_graph_prev:   
                        _dict[(prev_prev_state, prev_state)] = viterbi[(prev_prev_state, prev_state)][time_step-1] * estimate_transition_probabilites[(prev_prev_state, prev_state)][state] * estimate_emission_probabilites[state][observation]
                        #print(_dict[(prev_prev_state, prev_state)])
                    max_prob = max(zip(_dict.values(), _dict.keys()))
                    viterbi[(prev_state, state)][time_step] = max_prob[0]
                    back_pointer[(prev_state, state)][time_step]=max_prob[1][0]
                 
                  
    
        _dict = {}
        state_graph_prev_prev = state_graph
        state_graph_prev = state_graph

        if(len_observations == 1):
           state_graph_prev_prev = ['<s>']
            
        for prev_state in state_graph:
            for prev_prev_state in state_graph_prev_prev:
                _dict[(prev_prev_state, prev_state)] = viterbi[(prev_prev_state, prev_state)][len_observations-1] * estimate_transition_probabilites[(prev_prev_state, prev_state)]['<e>']
                
      
        max_prob = max(zip(_dict.values(), _dict.keys()))
        best_path_prob = max_prob[0]
        best_path_pointer = max_prob[1]
      
        pointer = best_path_pointer
        index = len_observations - 1

        result.append(pointer[1])
        
        if(len_observations != 1):
             result.append(pointer[0])

       
        for i, k in enumerate(range(len_observations-1-2, -1, -1)):
            result.append(back_pointer[(result[i+1], result[i])][k+2])

        result.reverse()
  
        return result


   


    def predict(self, sentence):
      
        prediction = []
        
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()

        # prediction = self._viterbi_3(sentence ,tag_set)
        # assert (len(prediction) == len(sentence))
        prediction = self._viterbi_2(sentence ,tag_set)

        assert (len(prediction) == 3)

        for predict in prediction:
            assert (len(predict) == len(sentence))

        return prediction
            