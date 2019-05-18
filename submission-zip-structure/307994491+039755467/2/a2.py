import heapq
from collections import defaultdict

import a1


class Submission(a1.Submission):

    def predict(self, sentence, n=3):
        prediction = self._viterbi(sentence, self.tag_set)

        assert (len(prediction) == n)

        for predict in prediction:
            assert (len(predict) == len(sentence))

        return prediction

    def _viterbi(self, observations, state_graph, n=3):
        result = []
        len_observations = len(observations)

        viterbi = defaultdict(lambda: defaultdict(dict))
        back_pointer = defaultdict(lambda: defaultdict(dict))

        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites

        for state in state_graph:
            viterbi[state][0][0] = estimate_transition_probabilites['<s>'][state] * \
                                   estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0][0] = '<s>'

        for time_step in range(1, len_observations):
            observation = observations[time_step]
            for state in state_graph:
                heap = []
                for _state in state_graph:
                    prob = viterbi[_state][time_step - 1][0] * estimate_transition_probabilites[_state][state] * \
                           estimate_emission_probabilites[state][observation]
                    heapq.heappush(heap, (-prob, _state, 0))

                index = 0
                while len(heap) > 0 and index < n:
                    curr = heapq.heappop(heap)
                    prob = -curr[0]
                    _state = curr[1]
                    _index = curr[2]
                    viterbi[state][time_step][index] = prob
                    back_pointer[state][time_step][index] = (_state, _index)
                    if (_index + 1 < len(viterbi[_state][time_step - 1])):
                        prob = viterbi[_state][time_step - 1][_index + 1] * estimate_transition_probabilites[_state][
                            state] * estimate_emission_probabilites[state][observation]
                        heapq.heappush(heap, (-prob, _state, _index + 1))
                    index += 1

        heap = []
        for _state in state_graph:
            prob = viterbi[_state][len_observations - 1][0]
            heapq.heappush(heap, (-prob, _state, 0))

        count = 0
        while len(heap) > 0 and count < n:
            curr = heapq.heappop(heap)
            prob = -curr[0]
            _state = curr[1]
            _index = curr[2]
            best_path_prob = prob
            best_path_pointer = (_state, _index)

            pointer = best_path_pointer
            index = len_observations - 1
            path = []

            while pointer != '<s>':
                path = [pointer[0]] + path
                pointer = back_pointer[pointer[0]][index][pointer[1]]
                index -= 1

            result = result + [path]

            if (_index + 1 < len(viterbi[_state][len_observations - 1])):
                prob = viterbi[_state][len_observations - 1][_index + 1]
                heapq.heappush(heap, (-prob, _state, _index + 1))
            count += 1

        return result
