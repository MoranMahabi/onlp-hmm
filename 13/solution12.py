from collections import defaultdict

from solution import Submission


class Submission12(Submission):

    def parse(self, sentence):

        ''' mock parsing function, returns a constant parse unrelated to the input sentence '''

        # if len(sentence) > 10:
        #     return '(TOP (S (VP (VB TM)) (NP (NNT MSE) (NP (H H) (NN HLWWIIH))) (yyDOT yyDOT)))'
        print(len(sentence))

        cky = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        bp = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        # initialization - lex rules
        for i in range(1, len(sentence) + 1):
            found = False
            for parent_tag, lst in self.pcfg.rules.items():
                for rule, count in lst[self.pcfg.TERMINAL_RULES].items():
                    if rule[0] == sentence[i - 1]:
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

        print(bp[1][len(sentence)]['TOP'])
        print("--------------------------")

        print("------tree_to_str-------")

        ret = self.tree_to_str(bp, sentence, 1, len(sentence), "TOP")

        print("---ret----")
        print(ret)

        return ret
