from collections import deque

from cfg import CFG
from spec import Spec
from util.tree.builders import node_tree_from_sequence

TOTAL_MARK = ''


class Submission(Spec):
    cfg = CFG()

    def train(self, training_treebank_file='data/heb-ctrees.train'):
        with open(training_treebank_file, 'r') as train_set:
            i = 0
            for bracketed_notation_tree in train_set:
                i += 1
                if i > 100:  # short-pass for debug
                    break
                q = deque()
                node = node_tree_from_sequence(bracketed_notation_tree)
                q.append(node)
                while q:
                    node = q.popleft()
                    children = node.children
                    if len(children):
                        q.extend(children)
                        derived = tuple(c.tag for c in children)
                        self.cfg.add(node.tag, derived)

        self.cfg.binarize()

        self.cfg.validate()

        #  Percolation

    def parse(self, sentence):
        ''' mock parsing function, returns a constant parse unrelated to the input sentence '''
        return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        ''' function writing the parse to the output file '''
        with open(output_treebank_file, 'w') as f:
            for sentence in sentences:
                f.write(self.parse(sentence))
                f.write('\n')
