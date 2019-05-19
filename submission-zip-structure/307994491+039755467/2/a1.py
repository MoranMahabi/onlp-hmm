from collections import defaultdict

from submission_specs.SubmissionSpec12 import SubmissionSpec12


class Submission(SubmissionSpec12):
    tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
    BOS = '<s>'
    EOS = '<e>'

    def train(self, annotated_sentences):
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))

        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def predict(self, sentence):
        prediction = self._viterbi(sentence, self.tag_set)
        assert (len(prediction) == len(sentence))

        return prediction

    def _estimate_emission_probabilites(self, annotated_sentences):
        words_frequency = defaultdict(int)
        tags_frequency = defaultdict(int)
        tag_word_frequency = defaultdict(lambda: defaultdict(int))

        for sentence in annotated_sentences:
            for (word, tag) in sentence:
                words_frequency[word] += 1
                tags_frequency[tag] += 1
                tag_word_frequency[tag][word] += 1

        # calculate estimate_emission_probabilites with smoothing add-delta (delta = 0.001)
        delta = 0.001
        V = len(words_frequency)

        estimate_emission_probabilites = dict()

        for tag in self.tag_set:
            # set default value for P(tag|word) for each tag
            estimate_emission_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * V))

        for (tag, words) in tag_word_frequency.items():
            for (word, count) in words.items():
                estimate_emission_probabilites[tag][word] = (count + delta) / (tags_frequency[tag] + delta * V)

        self.estimate_emission_probabilites = estimate_emission_probabilites

    def _estimate_transition_probabilites(self, annotated_sentences):
        tags_frequency = defaultdict(int)
        tags_pair_frequency = defaultdict(lambda: defaultdict(int))

        for sentence in annotated_sentences:
            prev_tag = self.BOS
            tags_frequency[prev_tag] += 1
            for (word, tag) in sentence:
                tags_frequency[tag] += 1
                tags_pair_frequency[prev_tag][tag] += 1
                prev_tag = tag

        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.001)
        delta = 0.001
        V = len(tags_frequency)
        tag_set = [self.BOS] + self.tag_set

        estimate_transition_probabilites = dict()

        for tag in tag_set:
            # set default value for P(tag|prev_prev_tag) for each tag
            estimate_transition_probabilites[tag] = defaultdict(lambda: delta / (tags_frequency[tag] + delta * V))

        for (prev_tag, tags) in tags_pair_frequency.items():
            for (tag, count) in tags.items():
                estimate_transition_probabilites[prev_tag][tag] = (count + delta) / \
                                                                  (tags_frequency[prev_tag] + delta * V)

        self.estimate_transition_probabilites = estimate_transition_probabilites

    def _viterbi(self, observations, state_graph):
        result = []
        len_observations = len(observations)
        viterbi = defaultdict(dict)
        back_pointer = defaultdict(dict)
        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        # initialization step
        for state in state_graph:
            viterbi[state][0] = estimate_transition_probabilites[self.BOS][state] * \
                                estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0] = self.BOS

        # recursion step
        for time_step in range(1, len_observations):
            observation = observations[time_step]
            for state in state_graph:
                max_prob = float('-inf')
                for prev_state in state_graph:
                    curr_prob = viterbi[prev_state][time_step - 1] * estimate_transition_probabilites[prev_state][
                        state] * estimate_emission_probabilites[state][observation]
                    if (curr_prob > max_prob):
                        max_prob = curr_prob
                        viterbi[state][time_step] = curr_prob
                        back_pointer[state][time_step] = prev_state

        # termination step       
        max_prob = float('-inf')
        for prev_state in state_graph:
            curr_prob = viterbi[prev_state][len_observations - 1]
            if (curr_prob > max_prob):
                max_prob = curr_prob
                best_path_prob = curr_prob
                best_path_pointer = prev_state

        pointer = best_path_pointer
        time_step = len_observations - 1

        while pointer != self.BOS:
            result = [pointer] + result
            pointer = back_pointer[pointer][time_step]
            time_step -= 1

        return result
