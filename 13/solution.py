'''

This source file is an empty solution file inheriting the required spec class for this assignmentself.
It also demonstrates one way of writing your parses to the required output file

See the spec that this class inherits, for the spec details

'''

from spec import Spec
from time import sleep
from collections import deque
from util.tree.builders import node_tree_from_sequence
from CFG import CFG

class Submission(Spec):

    def train(self, training_treebank_file='data/heb-ctrees.train'):
        ''' mock training function, learns nothing '''
        context_free_grammar = CFG()
       
        with open(training_treebank_file, 'r') as train_set:
             for bracketed_notation_tree in train_set:
               node_tree = node_tree_from_sequence(bracketed_notation_tree)
               q = deque()
               q.append(node_tree)
               while q:
                   current_node = q.popleft()
                   derived = list()
                   children = current_node.children
                   for child in children:
                       derived.append(child.tag)
                       q.append(child)

                   if len(children) != 0:
                      context_free_grammar.rules[current_node.tag][''] += 1
                      context_free_grammar.rules[current_node.tag][tuple(derived)] += 1
                   
        #  Binarizatio

        # example
        #         VP → VB NP PP PP (prob q)
        #         VP          → VB VB*NP-PP-PP (prob=q)
        #         VB*NP-PP-PP → NP VB-NP*PP-PP (prob=1)
        #         VB-NP*PP-PP → PP PP          (prob=1)

        rules_by_parent = context_free_grammar.rules
        for parent in rules_by_parent.copy():
            for rule in rules_by_parent[parent].copy():
                if rule == '':
                    continue
                len_rule = len(rule)
                if len_rule <= 2:
                    continue

                # insert '-' between two element in the rule
                _str = ['-'] * (len_rule * 2 - 1)
                _str[0::2] = list(rule)
                curr_parent = parent
                for i in range(len_rule -2):
                    _str1 = _str[:] 
                    _str1[2*i + 1] = '*'
                    next_parent = ''.join(_str1)
                    if i == 0:
                        context_free_grammar.rules[curr_parent][(rule[i], next_parent)] = context_free_grammar.rules[parent][rule]
                        context_free_grammar.rules[curr_parent][''] += context_free_grammar.rules[parent][rule]
                        del context_free_grammar.rules[parent][rule]
                    else:
                        context_free_grammar.rules[curr_parent][(rule[i], next_parent)] += 1
                        context_free_grammar.rules[curr_parent][''] += 1
  
                    curr_parent = next_parent
                
                # if i == 0:
                #     context_free_grammar.rules[curr_parent][(rule[len_rule-2], rule[len_rule-1])] = context_free_grammar.rules[parent][rule]
                #     context_free_grammar.rules[curr_parent][''] += context_free_grammar.rules[parent][rule]
                #     del context_free_grammar.rules[parent][rule]
                # else:
                context_free_grammar.rules[curr_parent][(rule[len_rule-2], rule[len_rule-1])] += 1
                context_free_grammar.rules[curr_parent][''] += 1
                
                

               
                

        
        #test probabilities:
       
        for tag in context_free_grammar.rules:
            _sum = 0

            for rule in context_free_grammar.rules[tag]:
                if rule != '':
                  _sum += context_free_grammar.rules[tag][rule]

            if _sum != context_free_grammar.rules[tag]['']:
                print('-------------failed----------------')
                print(_sum)
                print(context_free_grammar.rules[tag][''])
                print(tag)
                print(context_free_grammar.rules[tag])
                

            
        #  Percolatio





                    

             

       
        



              
    def parse(self, sentence):
        ''' mock parsing function, returns a constant parse unrelated to the input sentence '''
        return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'
    
    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        ''' function writing the parse to the output file '''
        with open(output_treebank_file, 'w') as f:
            for sentence in sentences:
                f.write(self.parse(sentence))
                f.write('\n')
                