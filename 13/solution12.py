from collections import defaultdict

from solution import Submission


class Submission12(Submission):

    def train(self, training_treebank_file='data/heb-ctrees.mini', percolate=False):
        super().train(training_treebank_file, percolate=False)

    def parse(self, sentence):
        print(len(sentence))

        cky = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        bp = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        # initialization - lex rules
        for i in range(1, len(sentence) + 1):
            found = False
            for parent_tag, lst in self.pcfg.rules.items():
                count = next((count for rule, count in lst[self.pcfg.TERMINAL_RULES].items() if rule[0] == sentence[i - 1]), None)
                if count is not None:
                    found = True
                    cky[i][i][parent_tag] = count / lst[self.pcfg.TOTAL_MARK]

            # handle unaries
            added = True
            while added:
                added = False
                for parent_tag, lst in self.pcfg.rules.items():
                    for rule, count in lst[self.pcfg.NON_TERMINAL_RULES].items():
                        if len(rule) != 1:
                            continue
                        if cky[i][i][rule[0]] > 0:
                            prob = count / lst[self.pcfg.TOTAL_MARK] * cky[i][i][rule[0]]
                            if prob > cky[i][i][parent_tag]:
                                cky[i][i][parent_tag] = prob
                                bp[i][i][parent_tag] = (parent_tag, rule)
                                added = True

        # algorithm - gram rules
        for length in range(1, len(sentence)):
            for i in range(1, len(sentence) - length + 1):
                for s in range(i, i + length):
                    for parent_tag, lst in self.pcfg.rules.items():
                        for rule, count in lst[self.pcfg.NON_TERMINAL_RULES].items():
                            if len(rule) != 2:
                                continue
                            if lst[self.pcfg.TOTAL_MARK] != 0 and cky[i][s][rule[0]] != 0 and cky[s + 1][i + length][rule[1]] != 0:
                                prob = (count / lst[self.pcfg.TOTAL_MARK]) * cky[i][s][rule[0]] * cky[s + 1][i + length][rule[1]]
                                if prob > cky[i][i + length][parent_tag]:
                                    cky[i][i + length][parent_tag] = prob
                                    bp[i][i + length][parent_tag] = (parent_tag, rule, s)

                    # handle unaries
                    added = True
                    while added:
                        added = False
                        for parent_tag, lst in self.pcfg.rules.items():
                            for rule, count in lst[self.pcfg.NON_TERMINAL_RULES].items():
                                if len(rule) != 1:
                                    continue
                                if cky[i][i + length][rule[0]] > 0:
                                    prob = count / lst[self.pcfg.TOTAL_MARK] * cky[i][i + length][rule[0]]
                                    if prob > cky[i][i + length][parent_tag]:
                                        cky[i][i + length][parent_tag] = prob
                                        bp[i][i + length][parent_tag] = (parent_tag, rule)
                                        added = True

        if 'TOP' not in bp[1][len(sentence)]:
            print("Un-parsable sentence")
            return ''

        print(bp[1][len(sentence)]['TOP'])
        print("--------------------------")

        ret = self.tree_to_str(bp, sentence, 1, len(sentence), "TOP")
        print(ret)

        return ret

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
