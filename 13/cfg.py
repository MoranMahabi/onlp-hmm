import copy
from collections import defaultdict, deque


class CFG:
    TERMINAL_RULES = 0
    NON_TERMINAL_RULES = 1
    TOTAL_MARK = 2
    rules = defaultdict(lambda: [defaultdict(float), defaultdict(float), 0.0])

    def add(self, tag, derived, is_terminal_rule, count=1):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        self.rules[tag][rule_index][derived] += count
        self.rules[tag][self.TOTAL_MARK] += count
    
    def add1(self, tag, derived, is_terminal_rule, count=1):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        self.rules[tag][rule_index][derived] += count

    def remove1(self, parent, derived, is_terminal_rule):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        del self.rules[parent][rule_index][derived]

    def remove(self, parent, derived, is_terminal_rule):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        self.rules[parent][self.TOTAL_MARK] -= self.rules[parent][rule_index][derived]
        del self.rules[parent][rule_index][derived]

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
                for b_rule, count in copy.deepcopy(list(self.rules[b][index].items())):
                    if (is_terminal) or (not is_terminal and len(b_rule) >= 2):
                       self.add1(a, b_rule, is_terminal, count=(count / self.rules[b][self.TOTAL_MARK]) * self.rules[a][self.NON_TERMINAL_RULES][(b,)])
                    if not is_terminal and len(b_rule) == 1:
                       self.add1(a, b_rule, is_terminal, count=(count / self.rules[b][self.TOTAL_MARK]) * self.rules[a][self.NON_TERMINAL_RULES][(b,)])
                       if (a, b_rule[0]) not in done and (a, b_rule[0]) not in worklist:
                            worklist.append((a, b_rule[0]))
                       else:
                            self.remove(a, b_rule, is_terminal)
                       
            self.remove1(a, (b,), False)
            done.add((a, b))
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
                if(len(r) != 2):
                    print(tag)
                    print(r)
            

            terminal_len_correct = all(len(r) == 1 for r in lst[self.TERMINAL_RULES].keys())
            non_terminal_len_correct = all(len(r) == 2 for r in lst[self.NON_TERMINAL_RULES].keys())
            #print(non_terminal_len_correct)
            assert terminal_len_correct, "Terminal rule length isn't 1"
            assert non_terminal_len_correct, "Non Terminal rule length isn't 2"
           

            if _sum != lst[self.TOTAL_MARK]:
                print('------------- Failed ----------------')
                print(_sum)
                print(lst[self.TOTAL_MARK])

                #print(f'Calculated sum {_sum}, Saved sum {tag_rules[self.TOTAL_MARK]}, {tag}')
                # print(tag_rules)

        assert res
