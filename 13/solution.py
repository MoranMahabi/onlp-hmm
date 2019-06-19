from collections import deque, defaultdict

from cfg import CFG
from spec import Spec
from util.tree.builders import node_tree_from_sequence
from util.transliteration import to_trans



TOTAL_MARK = ''


class Submission(Spec):
    cfg = CFG()

    def train(self, training_treebank_file='data/heb-ctrees.train'):
        with open(training_treebank_file, 'r') as train_set:
            i = 0
            for bracketed_notation_tree in train_set:
                # i += 1
                # if i > 5:  # short-pass for debug
                #      break
                q = deque()
                node = node_tree_from_sequence(bracketed_notation_tree)
                q.append(node)
                while q:
                    node = q.popleft()
                    children = node.children
                    if len(children):
                        q.extend(children)
                        derived = tuple(c.tag for c in children)
                        if len(children) == 1 and len(children[0].children) == 0:
                            self.cfg.add(node.tag, derived, True)
                        else:
                            self.cfg.add(node.tag, derived, False)

        

        self.cfg.binarize()

        self.cfg.percolate()

        self.cfg.validate()


    def parse(self, sentence):
        ''' mock parsing function, returns a constant parse unrelated to the input sentence '''
        print(sentence)


        CKY = defaultdict(lambda: defaultdict(lambda : defaultdict(float)))
        BP = defaultdict(lambda: defaultdict(lambda : defaultdict()))

        k = 0
        print(len(sentence))
        

        # initialization - lex rules
        for i in range(0,len(sentence)):
            found = False
            for parent_tag, lst in self.cfg.rules.items():
              for rule, count in lst[self.cfg.TERMINAL_RULES].items():
                  if(rule[0] == sentence[i]):
                     CKY[i][i+1][parent_tag] = count / lst[self.cfg.TOTAL_MARK]
                     found = True
            if found:
                k = k + 1
        

        print(k)
        
        # algorithm - gram rules
        for length in range(2, len(sentence)):
            for i in range(0, len(sentence)-length+1):
                j = i + length
                _max = 0
                _argmax = None
                for parent_tag, lst in self.cfg.rules.items():
                    for rule, count in lst[self.cfg.NON_TERMINAL_RULES].items():
                        for s in range(i+1, j):
                            curr = count / lst[self.cfg.TOTAL_MARK] * CKY[i][s][rule[0]] * CKY[s+1][j][rule[1]]
                            if(curr > _max):
                                _max = curr
                                _argmax = (parent_tag, rule)
                    CKY[i][j][parent_tag] = _max
                    BP[i][j][parent_tag] = _argmax
        
     

        return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'





        # for j in range(len(sentence)):
        #   for parent_tag, lst in self.cfg.rules.items():
        #     for rule, count in lst[self.cfg.TERMINAL_RULES].items():
        #         if()
        #         table[j][j+1][parent_tag] = count / lst[self.cfg.TOTAL_MARK]
        
        #   for i in range(j-1,-1,-1):
        #       for k in range(i+1, j):
        #          for parent_tag, lst in self.cfg.rules.items():
        #               for rule, count in lst[self.cfg.NON_TERMINAL_RULES].items():




        

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        ''' function writing the parse to the output file '''
        with open(output_treebank_file, 'w') as f:
            for sentence in sentences:
                f.write(self.parse(sentence))
                f.write('\n')
