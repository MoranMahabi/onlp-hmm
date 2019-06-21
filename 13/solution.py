import pickle
from collections import deque, defaultdict

from pcfg import PCFG
from spec import Spec
from util import transliteration
from util.tree.builders import node_tree_from_sequence

DEBUG = False


class Submission(Spec):
    pcfg = PCFG()

    def train(self, training_treebank_file='data/heb-ctrees.train'):
        if DEBUG:
            self.pcfg = pickle.load(open("./pcfg.p", "rb"))
            return

        with open(training_treebank_file, 'r') as train_set:
            i = 0
            for bracketed_notation_tree in train_set:
                # i += 1
                # if i > 100:  # short-pass for debug
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
                            self.pcfg.add(node.tag, derived, True)
                        else:
                            self.pcfg.add(node.tag, derived, False)

        self.pcfg.smooth_unknowns()

        self.pcfg.binarize()

        self.pcfg.percolate()

        self.pcfg.validate()

        pickle.dump(self.pcfg, open("./pcfg.p", "wb"))

    def parse(self, sentence):
        ''' mock parsing function, returns a constant parse unrelated to the input sentence '''

        # if len(sentence) > 10:
        #    return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'

        cky = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        bp = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        # initialization - lex rules
        for i in range(1, len(sentence) + 1):
            for parent_tag, lst in self.pcfg.rules.items():
                count = next((count for rule, count in lst[self.pcfg.TERMINAL_RULES].items() if rule[0] == sentence[i - 1]), None)
                if count is None:  # not found, used smoothed value
                    count = next((count for rule, count in lst[self.pcfg.TERMINAL_RULES].items() if rule[0] == self.pcfg.UNKNOWN), None)
                if count is not None:
                    cky[i][i][parent_tag] = count / lst[self.pcfg.TOTAL_MARK]

        # algorithm - gram rules
        for length in range(1, len(sentence)):
            for i in range(1, len(sentence) - length + 1):
                max_p = 0
                argmax = None
                for parent_tag, lst in self.pcfg.rules.items():
                    for rule, count in lst[self.pcfg.NON_TERMINAL_RULES].items():
                        for s in range(i, i + length):
                            if lst[self.pcfg.TOTAL_MARK] != 0:
                                curr = (count / lst[self.pcfg.TOTAL_MARK]) * cky[i][s][rule[0]] * cky[s + 1][i + length][rule[1]]
                                if curr > max_p:
                                    max_p = curr
                                    argmax = (parent_tag, rule, s)
                    cky[i][i + length][parent_tag] = max_p
                    bp[i][i + length][parent_tag] = argmax

        print(len(sentence))
        print(bp[1][len(sentence)]['TOP'])
        print("--------------------------")

        def tree_to_str(start, end, tag):
            if start == end:
                return tag
            if tag not in bp[start][end]:
                return 'XX-' + tag
            tag, children_tags, s = bp[start][end][tag]
            left_tag, right_tag = children_tags
            left_str = tree_to_str(start, s, left_tag)
            right_str = tree_to_str(s + 1, end, right_tag)
            if '*' in tag:
                tag = tag.split("*")[-1].replace("-", " ")
            return f"{tag} ({left_str} {right_str})"

        ret = "(" + tree_to_str(1, len(sentence), "TOP") + ")"

        return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        ''' function writing the parse to the output file '''
        with open(output_treebank_file, 'w') as f:
            for sentence in sentences:
                f.write(self.parse(sentence))
                f.write('\n')

    def to_heb(self, sentence):
        " ".join([''.join([transliteration.to_heb(c) for c in w]) for w in sentence if not w.startswith('yy')])
