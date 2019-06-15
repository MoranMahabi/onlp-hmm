from collections import defaultdict


class CFG:
    TOTAL_MARK = ''
    rules = defaultdict(lambda: defaultdict(int))

    def add(self, tag, derived, count=1):
        self.rules[tag][self.TOTAL_MARK] += count
        self.rules[tag][derived] += count

    def remove(self, parent, derived):
        del self.rules[parent][derived]

    def binarize(self):
        # example
        #         VP → VB NP PP PP (prob q)
        #         VP          → VB VB*NP-PP-PP (prob=q)
        #         VB*NP-PP-PP → NP VB-NP*PP-PP (prob=1)
        #         VB-NP*PP-PP → PP PP          (prob=1)

        for parent_tag, rules in self.rules.copy().items():
            for rule in rules.copy():
                if rule == self.TOTAL_MARK:
                    continue
                rule_len = len(rule)
                if rule_len <= 2:  # Already binary, or unary
                    continue

                # insert '-' between two element in the rule
                dashed = ['-'] * (rule_len * 2 - 1)
                dashed[0::2] = list(rule)
                curr_parent_tag = parent_tag
                for i in range(rule_len - 2):
                    new_derived = dashed[:]
                    new_derived[2 * i + 1] = '*'
                    next_parent_tag = ''.join(new_derived)
                    derived = (rule[i], next_parent_tag)
                    if i == 0:
                        self.add(curr_parent_tag, derived, self.rules[parent_tag][rule])
                        self.remove(parent_tag, rule)
                    else:
                        self.add(curr_parent_tag, derived)

                    curr_parent_tag = next_parent_tag

                # if i == 0:
                #     context_free_grammar.rules[curr_parent][(rule[len_rule-2], rule[len_rule-1])] = context_free_grammar.rules[parent][rule]
                #     context_free_grammar.rules[curr_parent][TOTAL_MARK] += context_free_grammar.rules[parent][rule]
                #     del context_free_grammar.rules[parent][rule]
                # else:

                self.add(curr_parent_tag, (rule[rule_len - 2], rule[rule_len - 1]))

    def validate(self):
        # test probabilities:
        res = True
        for tag, tag_rules in self.rules.items():
            _sum = sum(v for rule, v in tag_rules.items() if rule != self.TOTAL_MARK)

            if _sum != tag_rules[self.TOTAL_MARK]:
                print('------------- Failed ----------------')
                print(f'Calculated sum {_sum}, Saved sum {tag_rules[self.TOTAL_MARK]}, {tag}')
                print(tag_rules)

        assert res
