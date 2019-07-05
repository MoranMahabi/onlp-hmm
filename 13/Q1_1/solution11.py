from collections import defaultdict

from solution import Submission


class Submission11(Submission):

    def train(self, training_treebank_file='./data/heb-ctrees.train', percolate=True):
        super().train(training_treebank_file, percolate=True)

    def parse(self, sentence):
        cky = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        bp = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        # initialization - lex rules
        for i in range(1, len(sentence) + 1):
            possible_parents = self.pcfg.reverse_rules[self.pcfg.TERMINAL_RULES][sentence[i - 1]]
            for parent_tag, rule, rule_prob in possible_parents:
                cky[i][i][parent_tag] = rule_prob
            if not possible_parents:
                cky[i][i].update(self.pcfg.unknown_rules)

        # algorithm - gram rules
        for length in range(1, len(sentence)):
            for i in range(1, len(sentence) - length + 1):
                for s in range(i, i + length):
                    possible_left_children = [k for k, v in cky[i][s].items() if v > 0]
                    for plc in possible_left_children:
                        possible_left_parents = self.pcfg.reverse_rules[self.pcfg.NON_TERMINAL_RULES][plc]
                        for parent_tag, rule, rule_prob in possible_left_parents:
                            assert len(rule) == 2
                            assert plc == rule[0]  # rule is of left child
                            assert cky[i][s].get(rule[0], 0) != 0
                            if rule_prob and cky[s + 1][i + length].get(rule[1], 0) != 0:
                                prob = rule_prob * cky[i][s][rule[0]] * cky[s + 1][i + length][rule[1]]
                                if prob > cky[i][i + length].get(parent_tag, 0):
                                    cky[i][i + length][parent_tag] = prob
                                    bp[i][i + length][parent_tag] = (parent_tag, rule, s)

        if 'TOP' not in bp[1][len(sentence)]:
            print("Un-parsable sentence")
            return ''

        ret = self.tree_to_str(bp, sentence, 1, len(sentence), "TOP")
        return ret

    def tree_to_str(self, bp, sentence, start, end, tag):
        if start == end and tag not in bp[start][end]:
            return f"({tag} {sentence[start - 1]})"
        if tag not in bp[start][end]:  # Shouldn't happen
            return 'XX-' + tag
        tag, children_tags, s = bp[start][end][tag]
        left_tag, right_tag = children_tags
        left_str = self.tree_to_str(bp, sentence, start, s, left_tag)
        right_str = self.tree_to_str(bp, sentence, s + 1, end, right_tag)
        if '*' in tag:
            return f"{left_str} {right_str}"
        return f"({tag} {left_str} {right_str})"
