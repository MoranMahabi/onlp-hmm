from collections import defaultdict

from solution import Submission


class Submission12(Submission):

    def train(self, training_treebank_file='data/heb-ctrees.train', percolate=False, parent_encoding=False):
        super().train(training_treebank_file, percolate=percolate, parent_encoding=parent_encoding)

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

            # handle unaries
            added = True
            while added:
                added = False
                possible_left_children = [k for k, v in cky[i][i].items() if v > 0]
                for plc in possible_left_children:
                    possible_left_parents = self.pcfg.reverse_rules[self.pcfg.NON_TERMINAL_RULES][plc]
                    for parent_tag, rule, rule_prob in possible_left_parents:
                        if len(rule) != 1:
                            continue
                        assert plc == rule[0]  # rule is of left child
                        assert cky[i][i].get(plc, 0) != 0
                        prob = rule_prob * cky[i][i][plc]
                        if prob > cky[i][i].get(parent_tag, 0):
                            cky[i][i][parent_tag] = prob
                            bp[i][i][parent_tag] = (parent_tag, (plc,))
                            added = True

        # algorithm - gram rules
        for length in range(1, len(sentence)):
            for i in range(1, len(sentence) - length + 1):
                for s in range(i, i + length):
                    possible_left_children = [k for k, v in cky[i][s].items() if v > 0]
                    for plc in possible_left_children:
                        possible_left_parents = self.pcfg.reverse_rules[self.pcfg.NON_TERMINAL_RULES][plc]
                        for parent_tag, rule, rule_prob in possible_left_parents:
                            if len(rule) != 2:
                                continue
                            assert plc == rule[0]  # rule is of left child
                            assert cky[i][s].get(rule[0], 0) != 0
                            if rule_prob and cky[s + 1][i + length].get(rule[1], 0) != 0:
                                prob = rule_prob * cky[i][s][rule[0]] * cky[s + 1][i + length][rule[1]]
                                if prob > cky[i][i + length].get(parent_tag, 0):
                                    cky[i][i + length][parent_tag] = prob
                                    bp[i][i + length][parent_tag] = (parent_tag, rule, s)

                # handle unaries
                added = True
                while added:
                    added = False
                    possible_left_children = [k for k, v in cky[i][i + length].items() if v > 0]
                    for plc in possible_left_children:
                        possible_left_parents = self.pcfg.reverse_rules[self.pcfg.NON_TERMINAL_RULES][plc]
                        for parent_tag, rule, rule_prob in possible_left_parents:
                            if len(rule) != 1:
                                continue
                            assert plc == rule[0]  # rule is of left child
                            assert cky[i][i + length].get(plc, 0) != 0
                            prob = rule_prob * cky[i][i + length][plc]
                            if prob > cky[i][i + length].get(parent_tag, 0):
                                cky[i][i + length][parent_tag] = prob
                                bp[i][i + length][parent_tag] = (parent_tag, rule)
                                added = True

        return self.get_bp_result(bp, sentence)

    def tree_to_str(self, bp, sentence, start, end, tag):
        if start == end and tag not in bp[start][end]:
            return f"({tag} {sentence[start - 1]})"
        if len(bp[start][end][tag]) == 2:  # without split, unary chain
            tag, children_tags = bp[start][end][tag]
            chain = self.tree_to_str(bp, sentence, start, end, children_tags[0])
            return f"({tag} {chain})"

        tag, children_tags, s = bp[start][end][tag]
        left_tag, right_tag = children_tags
        left_str = self.tree_to_str(bp, sentence, start, s, left_tag)
        right_str = self.tree_to_str(bp, sentence, s + 1, end, right_tag)
        if '*' in tag:
            return f"{left_str} {right_str}"
        return f"({tag} {left_str} {right_str})"
