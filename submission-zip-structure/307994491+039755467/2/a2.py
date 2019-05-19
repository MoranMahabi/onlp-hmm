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

        # initialization step
        for state in state_graph:
            viterbi[state][0][0] = estimate_transition_probabilites[self.BOS][state] * \
                                   estimate_emission_probabilites[state][observations[0]]
            back_pointer[state][0][0] = self.BOS

        # recursion step
        for time_step in range(1, len_observations):
            observation = observations[time_step]
            for state in state_graph:
                heap = []
                for prev_state in state_graph:
                    prob = viterbi[prev_state][time_step - 1][0] * estimate_transition_probabilites[prev_state][state] * \
                           estimate_emission_probabilites[state][observation]
                    heapq.heappush(heap, (-prob, prev_state, 0))

                count = 0
                while len(heap) > 0 and count < n:
                    curr = heapq.heappop(heap)
                    prob = -curr[0]
                    prev_state = curr[1]
                    index = curr[2]
                    viterbi[state][time_step][count] = prob
                    back_pointer[state][time_step][count] = (prev_state, index)
                    if (index + 1 < len(viterbi[prev_state][time_step - 1])):
                        prob = viterbi[prev_state][time_step - 1][index + 1] * \
                               estimate_transition_probabilites[prev_state][state] * \
                               estimate_emission_probabilites[state][observation]
                        heapq.heappush(heap, (-prob, prev_state, index + 1))
                    count += 1

        # termination step
        heap = []

        for state in state_graph:
            prob = viterbi[state][len_observations - 1][0]
            heapq.heappush(heap, (-prob, state, 0))

        count = 0
        while len(heap) > 0 and count < n:
            curr = heapq.heappop(heap)
            prob = -curr[0]
            state = curr[1]
            index = curr[2]
            best_path_prob = prob
            best_path_pointer = (state, index)

            pointer = best_path_pointer
            time_step = len_observations - 1
            path = []

            while pointer != self.BOS:
                path = [pointer[0]] + path
                pointer = back_pointer[pointer[0]][time_step][pointer[1]]
                time_step -= 1

            result = result + [path]

            if index + 1 < len(viterbi[state][len_observations - 1]):
                prob = viterbi[state][len_observations - 1][index + 1]
                heapq.heappush(heap, (-prob, state, index + 1))

            count += 1

        return result
