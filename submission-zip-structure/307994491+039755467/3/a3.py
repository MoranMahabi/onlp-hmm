from collections import defaultdict

import a1


class Submission(a1.Submission):

    def _estimate_transition_probabilites(self, annotated_sentences):
        tags_frequency = defaultdict(int)
        tags_pair_frequency = defaultdict(lambda: defaultdict(int))
        tags_frequency_0 = defaultdict(int)

        for sentence in annotated_sentences:
            prev_prev_tag = '<s>'
            prev_tag = '<s>'
            tags_frequency[(prev_prev_tag, prev_tag)] += 1
            for (word, tag) in sentence:
                tags_frequency_0[tag] += 1
                tags_frequency[(prev_tag, tag)] += 1
                tags_pair_frequency[(prev_prev_tag, prev_tag)][tag] += 1
                prev_prev_tag = prev_tag
                prev_tag = tag
            tags_frequency[(prev_tag, '<e>')] += 1
            tags_pair_frequency[(prev_prev_tag, prev_tag)]['<e>'] += 1

        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.05)
        delta = 0.05
        len_tags = len(tags_frequency_0) + 1
        tag_set = ['<s>'] + self.tag_set

        estimate_transition_probabilites = dict()
        for prev_tag in tag_set:
            for tag in tag_set:
                if tag == '<s>' and prev_tag != '<s>':
                    continue

                estimate_transition_probabilites[(prev_tag, tag)] = defaultdict(
                    lambda: delta / (tags_frequency[(prev_tag, tag)] + delta * len_tags))

        for ((prev_prev_tag, prev_tag), tags) in tags_pair_frequency.items():
            for (tag, count) in tags.items():
                estimate_transition_probabilites[(prev_prev_tag, prev_tag)][tag] = (count + delta) / (
                        tags_frequency[(prev_prev_tag, prev_tag)] + delta * len_tags)

        self.estimate_transition_probabilites = estimate_transition_probabilites

    def _viterbi(self, observations, state_graph):
        result = []
        len_observations = len(observations)

        viterbi = defaultdict(dict)
        back_pointer = defaultdict(dict)

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        for state in state_graph:
            viterbi[('<s>', state)][0] = estimate_transition_probabilites[('<s>', '<s>')][state] * \
                                         estimate_emission_probabilites[state][observations[0]]
            back_pointer[('<s>', state)][0] = '<s>'

        for time_step in range(1, len_observations):
            observation = observations[time_step]

            for state in state_graph:
                for prev_state in state_graph:
                    _dict = {}
                    state_graph_prev = state_graph
                    if (time_step == 1):
                        state_graph_prev = ['<s>']
                    for prev_prev_state in state_graph_prev:
                        _dict[(prev_prev_state, prev_state)] = viterbi[(prev_prev_state, prev_state)][time_step - 1] * \
                                                               estimate_transition_probabilites[
                                                                   (prev_prev_state, prev_state)][state] * \
                                                               estimate_emission_probabilites[state][observation]
                    max_prob = max(zip(_dict.values(), _dict.keys()))
                    viterbi[(prev_state, state)][time_step] = max_prob[0]
                    back_pointer[(prev_state, state)][time_step] = max_prob[1][0]

        _dict = {}
        state_graph_prev_prev = state_graph
        state_graph_prev = state_graph

        if len_observations == 1:
            state_graph_prev_prev = ['<s>']

        for prev_state in state_graph:
            for prev_prev_state in state_graph_prev_prev:
                _dict[(prev_prev_state, prev_state)] = viterbi[(prev_prev_state, prev_state)][len_observations - 1] * \
                                                       estimate_transition_probabilites[(prev_prev_state, prev_state)][
                                                           '<e>']

        max_prob = max(zip(_dict.values(), _dict.keys()))
        best_path_prob = max_prob[0]
        best_path_pointer = max_prob[1]

        pointer = best_path_pointer
        index = len_observations - 1

        result.append(pointer[1])

        if len_observations != 1:
            result.append(pointer[0])

        for i, k in enumerate(range(len_observations - 1 - 2, -1, -1)):
            result.append(back_pointer[(result[i + 1], result[i])][k + 2])

        result.reverse()
        return result
