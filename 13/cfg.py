from collections import defaultdict
import copy

class CFG:

    TERMINAL_RULES = 0
    NON_TERMINAL_RULES = 1
    TOTAL_MARK = 2
    rules = defaultdict(lambda: [defaultdict(int), defaultdict(int), 0])

    def add(self, tag, derived, is_terminal_rule, count=1):
        if is_terminal_rule:
             self.rules[tag][self.TERMINAL_RULES][derived] += count
        else:
             self.rules[tag][self.NON_TERMINAL_RULES][derived] += count
        
        self.rules[tag][self.TOTAL_MARK] += count

    def remove(self, parent, derived, is_terminal_rule):
       if is_terminal_rule:
           self.rules[parent][self.TOTAL_MARK] = self.rules[parent][self.TOTAL_MARK] - self.rules[parent][self.TERMINAL_RULES][derived]
           del self.rules[parent][self.TERMINAL_RULES][derived]
       else:
           self.rules[parent][self.TOTAL_MARK] = self.rules[parent][self.TOTAL_MARK] - self.rules[parent][self.NON_TERMINAL_RULES][derived]
           del self.rules[parent][self.NON_TERMINAL_RULES][derived]

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
                        self.add(curr_parent_tag, False, derived)

                    curr_parent_tag = next_parent_tag

                self.add(curr_parent_tag, (rule[rule_len - 2], rule[rule_len - 1]), False)

    def validate(self):
        # test probabilities:
        print('----validate-----')
        res = True
        for tag, lst in self.rules.items():
            _sum = sum(v for rule, v in lst[self.TERMINAL_RULES].items())
            _sum = _sum + sum(v for rule, v in lst[self.NON_TERMINAL_RULES].items())
           
            if _sum != lst[self.TOTAL_MARK]:
                print('------------- Failed ----------------')
                #print(f'Calculated sum {_sum}, Saved sum {tag_rules[self.TOTAL_MARK]}, {tag}')
                #print(tag_rules)

        #assert res
