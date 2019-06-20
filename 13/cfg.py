import copy
from collections import defaultdict, deque
from math import isclose


class CFG:
    UNKNOWN=-1
    TERMINAL_RULES = 0
    NON_TERMINAL_RULES = 1
    TOTAL_MARK = 2
    rules = defaultdict(lambda: [defaultdict(float), defaultdict(float), 0.0])

    def add(self, tag, derived, is_terminal_rule, count=1.0, fix_total=True):
        if not count:
            return
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        self.rules[tag][rule_index][derived] += count
        if fix_total:
            self.rules[tag][self.TOTAL_MARK] += count

    def remove(self, parent, derived, is_terminal_rule, fix_total=True):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        if fix_total:
            self.rules[parent][self.TOTAL_MARK] -= self.rules[parent][rule_index][derived]
        del self.rules[parent][rule_index][derived]

    def unknownSmoothing(self):
         words_frequency = defaultdict(int)

         for parent_tag, lst in self.rules.items():
            for rule in lst[self.TERMINAL_RULES].keys():
                words_frequency[rule[0]] += 1
        
         min_frequency = words_frequency[min(words_frequency.keys(), key=(lambda k: words_frequency[k]))]

         for parent_tag, lst in copy.deepcopy(list(self.rules.items())):
            for rule, count in lst[self.TERMINAL_RULES].items():
                if(words_frequency[rule[0]] == min_frequency):
                    del self.rules[parent_tag][self.TERMINAL_RULES][rule]
                    self.rules[parent_tag][self.TERMINAL_RULES][(self.UNKNOWN,)] += count
                  
    def percolate(self):
        worklist = deque()
        done = set()
        for parent_tag, lst in copy.deepcopy(list(self.rules.items())):
            for rule in lst[self.NON_TERMINAL_RULES].keys():
                if len(rule) == 1:
                    worklist.append((parent_tag, rule[0]))

        while worklist:
            a, b = worklist.popleft()
            for index in [self.NON_TERMINAL_RULES, self.TERMINAL_RULES]:
                is_terminal = (index == self.TERMINAL_RULES)

                a_to_b_count = self.rules[a][self.NON_TERMINAL_RULES][(b,)]
                total_b = self.rules[b][self.TOTAL_MARK]
                a_from_b_ratio = a_to_b_count / total_b

                for b_rule, count in copy.deepcopy(list(self.rules[b][index].items())):
                    self.add(a, b_rule, is_terminal, count=count * a_from_b_ratio, fix_total=False)
                    if not is_terminal and len(b_rule) == 1:
                        if (a, b_rule[0]) not in done and (a, b_rule[0]) not in worklist:
                            worklist.append((a, b_rule[0]))
                        else:
                            self.remove(a, b_rule, False, fix_total=True)

            done.add((a, b))
            self.remove(a, (b,), False, fix_total=False)
            print(len(worklist))

    def binarize(self):
        # example
        #         VP → VB NP PP PP (prob q)
        #         VP          → VB VB*NP-PP-PP (prob=q)
        #         VB*NP-PP-PP → NP VB-NP*PP-PP (prob=1)
        #         VB-NP*PP-PP → PP PP          (prob=1)

        for parent_tag, lst in copy.deepcopy(list(self.rules.items())):
            for rule, count in lst[self.NON_TERMINAL_RULES].items():
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
                        self.add(curr_parent_tag, derived, False, count)
                        self.remove(parent_tag, rule, False)
                    else:
                        self.add(curr_parent_tag, derived, False)

                    curr_parent_tag = next_parent_tag

                self.add(curr_parent_tag, (rule[rule_len - 2], rule[rule_len - 1]), False)

    def validate(self):
        # test probabilities:
        print('----validate-----')
        res = True
        for tag, lst in self.rules.items():
            _sum = 0
            for index in [self.NON_TERMINAL_RULES, self.TERMINAL_RULES]:
                _sum += sum(v for rule, v in lst[index].items())

            for r in lst[self.NON_TERMINAL_RULES].keys():
                if (len(r) != 2):
                    print(tag)
                    print(r)

            terminal_len_correct = all(len(r) == 1 for r in lst[self.TERMINAL_RULES].keys())
            non_terminal_len_correct = all(len(r) == 2 for r in lst[self.NON_TERMINAL_RULES].keys())
            # print(non_terminal_len_correct)
            assert terminal_len_correct, "Terminal rule length isn't 1"
            assert non_terminal_len_correct, "Non Terminal rule length isn't 2"

            if not isclose(_sum, lst[self.TOTAL_MARK], abs_tol=0.0001):
                print('------------- Failed ----------------')
                print(_sum)
                print(lst[self.TOTAL_MARK])

                # print(f'Calculated sum {_sum}, Saved sum {tag_rules[self.TOTAL_MARK]}, {tag}')
                # print(tag_rules)

        assert res
