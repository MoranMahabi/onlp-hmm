from collections import deque, defaultdict

import dill

from pcfg import PCFG
from spec import Spec
from util import transliteration
from util.tree.builders import node_tree_from_sequence

DEBUG = False


class Submission(Spec):
    pcfg = PCFG()

    def train(self, training_treebank_file='data/heb-ctrees.mini'):
        if DEBUG:
            with open("./pcfg.p", "rb") as f:
                self.pcfg = dill.load(f)
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

        # self.pcfg.smooth_unknowns()

        self.pcfg.binarize()

        self.pcfg.percolate()

        self.pcfg.validate()

        with open("./pcfg.p", "wb") as f:
            dill.dump(self.pcfg, f)

    def parse(self, sentence):
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
                j = i + length
                for parent_tag, lst in self.pcfg.rules.items():
                    for rule, count in lst[self.pcfg.NON_TERMINAL_RULES].items():
                        for s in range(i, j):
                            curr = (count / lst[self.pcfg.TOTAL_MARK]) * cky[i][s][rule[0]] * cky[s + 1][j][rule[1]]
                            if curr > max_p:
                                max_p = curr
                                argmax = (parent_tag, rule, s)
                    cky[i][j][parent_tag] = max_p
                    bp[i][j][parent_tag] = argmax

        print(len(sentence))
        print(bp[1][len(sentence)]['TOP'])
        print("--------------------------")

        return self.tree_to_str(bp, sentence, 1, len(sentence), "TOP")

    def tree_to_str(self, bp, sentence, start, end, tag):
        if start == end:
            return f"({tag} {sentence[start - 1]})"
        if tag not in bp[start][end]:
            return 'XX-' + tag
        tag, children_tags, s = bp[start][end][tag]
        left_tag, right_tag = children_tags
        left_str = self.tree_to_str(bp, sentence, start, s, left_tag)
        right_str = self.tree_to_str(bp, sentence, s + 1, end, right_tag)
        if '*' in tag:
            return f"{left_str} {right_str}"
        return f"({tag} {left_str} {right_str})"

    def write_parse(self, sentences, output_treebank_file='output/predicted.txt'):
        ''' function writing the parse to the output file '''
        with open(output_treebank_file, 'w') as f:
            for sentence in sentences:
                f.write(self.parse(sentence))
                f.write('\n')
                f.flush()

    def to_heb(self, sentence):
        " ".join([''.join([transliteration.to_heb(c) for c in w]) for w in sentence if not w.startswith('yy')])
