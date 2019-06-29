import copy
from collections import defaultdict, deque

from math import isclose

from util.tree.builders import node_tree_from_sequence


class PCFG:
    TERMINAL_RULES = "T"
    NON_TERMINAL_RULES = "N"
    TOTAL_MARK = "Total"

    UNKNOWN = -1

    percolated = False

    def __init__(self, training_treebank_file, parent_encoding):
        self.unknown_rules = defaultdict(float)
        self.rules = defaultdict(
            lambda: {
                self.TERMINAL_RULES: defaultdict(float),
                self.NON_TERMINAL_RULES: defaultdict(float),
                self.TOTAL_MARK: 0.0  # relevant only for unnormalized grammar
            })
        self.reverse_rules = {
            self.TERMINAL_RULES: defaultdict(list),
            self.NON_TERMINAL_RULES: defaultdict(list),
        }
        self.terminals = set()

        if parent_encoding:
            self.update_rules_with_parent_encoding(training_treebank_file)
        else:
            self.update_rules(training_treebank_file)

    def update_rules_with_parent_encoding(self, training_treebank_file):
        with open(training_treebank_file, 'r') as train_set:
            for bracketed_notation_tree in train_set:
                q = deque()
                node = node_tree_from_sequence(bracketed_notation_tree)
                q.append((node, None))
                while q:
                    node, parent_node_tag = q.popleft()
                    children = node.children
                    if len(children):
                        q.extend([(c, node.tag) for c in children])
                        prarent_tag = node.tag if parent_node_tag is None else f"{node.tag}@{parent_node_tag}"
                        if len(children) == 1 and len(children[0].children) == 0:
                            derived = tuple(c.tag for c in children)
                            self._add(prarent_tag, derived, True)
                            self.terminals.add(derived)
                        else:
                            derived = tuple(f"{c.tag}@{node.tag}" for c in children)
                            self._add(prarent_tag, derived, False)

    def update_rules(self, training_treebank_file):
        with open(training_treebank_file, 'r') as train_set:
            for bracketed_notation_tree in train_set:
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
                            self._add(node.tag, derived, True)
                            self.terminals.add(derived)
                        else:
                            self._add(node.tag, derived, False)

    def normalize_and_smooth(self):
        V = len(self.terminals)
        delta = 0.1

        for parent_tag, lst in self.rules.items():
            total = lst[self.TOTAL_MARK]
            total_none_terminals = 0

            for rule, count in lst[self.NON_TERMINAL_RULES].items():
                self.rules[parent_tag][self.NON_TERMINAL_RULES][rule] = count / total
                total_none_terminals += count

            total_terminals = total - total_none_terminals
            for rule, count in lst[self.TERMINAL_RULES].items():
                self.rules[parent_tag][self.TERMINAL_RULES][rule] = (count + delta) / (total + (total / total_terminals) * delta * V)

            if total_terminals != 0:
                self.unknown_rules[parent_tag] = delta / (total + (total / total_terminals) * delta * V)
            else:
                self.unknown_rules[parent_tag] = 0

    def percolate(self):
        self.percolated = True

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
                a_to_b_prob = self.rules[a][self.NON_TERMINAL_RULES][(b,)]

                for b_rule, prob in copy.deepcopy(list(self.rules[b][index].items())):
                    self._add(a, b_rule, is_terminal, count=prob * a_to_b_prob, fix_total=False)
                    if not is_terminal and len(b_rule) == 1:
                        if (a, b_rule[0]) not in done and (a, b_rule[0]) not in worklist:
                            worklist.append((a, b_rule[0]))
                        else:
                            self._remove(a, b_rule, False, fix_total=True)  # TO DO

            done.add((a, b))
            self._remove(a, (b,), False, fix_total=False)

    def binarize(self):
        # example
        #         VP → VB NP PP PP (prob q)
        #         VP          → VB VB*NP-PP-PP (prob=q)
        #         VB*NP-PP-PP → NP VB-NP*PP-PP (prob=1)
        #         VB-NP*PP-PP → PP PP          (prob=1)

        for parent_tag, lst in copy.deepcopy(list(self.rules.items())):
            for rule, prob in lst[self.NON_TERMINAL_RULES].items():
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
                        self.rules[curr_parent_tag][self.NON_TERMINAL_RULES][derived] = prob
                        del self.rules[parent_tag][self.NON_TERMINAL_RULES][rule]
                    else:
                        self.rules[curr_parent_tag][self.NON_TERMINAL_RULES][derived] = 1

                    curr_parent_tag = next_parent_tag

                self.rules[curr_parent_tag][self.NON_TERMINAL_RULES][(rule[rule_len - 2], rule[rule_len - 1])] = 1

    def reverse(self):
        for index in [self.TERMINAL_RULES, self.NON_TERMINAL_RULES]:
            for parent_tag, lst in self.rules.items():
                for rule, rule_prob in lst[index].items():
                    assert rule_prob
                    self.reverse_rules[index][rule[0]].append((parent_tag, rule, rule_prob))

    def validate(self):
        # test probabilities:
        print('----validate-----')
        res = True
        for tag, lst in self.rules.items():
            _sum = 0
            for index in [self.NON_TERMINAL_RULES, self.TERMINAL_RULES]:
                _sum += sum(v for rule, v in lst[index].items())

            if _sum < 1:
                print(tag)
                print(_sum)

            # for r in lst[self.NON_TERMINAL_RULES].keys():
            #     if len(r) != 2:
            #         print(tag)
            #         print(r)

            terminal_len_correct = all(len(r) == 1 for r in lst[self.TERMINAL_RULES].keys())
            if self.percolated:
                non_terminal_len_correct = all(len(r) == 2 for r in lst[self.NON_TERMINAL_RULES].keys())
            else:
                non_terminal_len_correct = all(1 <= len(r) <= 2 for r in lst[self.NON_TERMINAL_RULES].keys())
            # print(non_terminal_len_correct)
            assert terminal_len_correct, "Terminal rule length isn't 1"
            assert non_terminal_len_correct, "Non Terminal rule length isn't 2"

            # if not isclose(_sum, lst[self.TOTAL_MARK], abs_tol=0.0001):
            # print('------------- Failed ----------------')
            # print(_sum)
            # print(lst[self.TOTAL_MARK])

            # print(f'Calculated sum {_sum}, Saved sum {tag_rules[self.TOTAL_MARK]}, {tag}')
            # print(tag_rules)

        assert res

    def _add(self, tag, derived, is_terminal_rule, count=1.0, fix_total=True):
        if not count:
            return
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        self.rules[tag][rule_index][derived] += count
        if fix_total:
            self.rules[tag][self.TOTAL_MARK] += count

    def _remove(self, parent, derived, is_terminal_rule, fix_total=True):
        rule_index = self.TERMINAL_RULES if is_terminal_rule else self.NON_TERMINAL_RULES
        if fix_total:
            self.rules[parent][self.TOTAL_MARK] -= self.rules[parent][rule_index][derived]
        del self.rules[parent][rule_index][derived]
