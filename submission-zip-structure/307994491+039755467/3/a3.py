from collections import defaultdict

import a1


class Submission(a1.Submission):

    def _estimate_transition_probabilites(self, annotated_sentences):
        BOS = '<s>'
        EOS = '<e>'
        tags_frequency = defaultdict(int)
        bigrams_frequency = defaultdict(int)
        bigrams_tags_frequency = defaultdict(lambda: defaultdict(int))
       
        for sentence in annotated_sentences:
            prev_prev_tag = BOS
            prev_tag = BOS
            bigrams_frequency[(prev_prev_tag, prev_tag)] += 1
            for (word, tag) in sentence:
                tags_frequency[tag] += 1
                bigrams_frequency[(prev_tag, tag)] += 1
                bigrams_tags_frequency[(prev_prev_tag, prev_tag)][tag] += 1
                prev_prev_tag = prev_tag
                prev_tag = tag
            bigrams_frequency[(prev_tag, EOS)] += 1
            bigrams_tags_frequency[(prev_prev_tag, prev_tag)][EOS] += 1

        # calculate estimate_transition_probabilites with smoothing add-delta (delta = 0.001)
        delta = 0.001
        V = len(tags_frequency) + 1  # number of unique tags, include EOS
        tag_set = [BOS] + self.tag_set

        estimate_transition_probabilites = dict()

        for prev_prev_tag in tag_set:
            for prev_tag in tag_set:
                if prev_tag == BOS and prev_prev_tag != BOS:
                    continue

                # set default value for P(tag|prev_prev_tag, prev_tag) for each tag
                estimate_transition_probabilites[(prev_prev_tag, prev_tag)] = defaultdict(lambda: delta / (bigrams_frequency[(prev_prev_tag, prev_tag)] + delta * V))

        for ((prev_prev_tag, prev_tag), tags) in bigrams_tags_frequency.items():
            for (tag, count) in tags.items():
                estimate_transition_probabilites[(prev_prev_tag, prev_tag)][tag] = (count + delta) / (bigrams_frequency[(prev_prev_tag, prev_tag)] + delta * V)

        self.estimate_transition_probabilites = estimate_transition_probabilites

     def _viterbi(self, observations, state_graph):
        BOS = '<s>'
        EOS = '<e>'

        result = []
        states_by_time_step = {}
        len_observations = len(observations)
        viterbi = defaultdict(dict)
        back_pointer = defaultdict(dict)
        estimate_transition_probabilites = self.estimate_transition_probabilites
        estimate_emission_probabilites = self.estimate_emission_probabilites
    
        states_by_time_step.update(dict.fromkeys([-2, -1], [BOS]))
        states_by_time_step.update(dict.fromkeys(range(len_observations), state_graph))

        # initialization step - the best probabiltie that we are in -1 time step and (prev_state, curr_state) = (BOS, BOS) is 1
        viterbi[(BOS, BOS)][-1] = 1

        # recursion step
        for time_step in range(len_observations):      
            observation = observations[time_step]
            for state in states_by_time_step[time_step]:
                for prev_state in states_by_time_step[time_step-1]:
                    max_prob = float('-inf')
                    for prev_prev_state in states_by_time_step[time_step-2]:
                        curr_prob = viterbi[(prev_prev_state, prev_state)][time_step - 1] * estimate_transition_probabilites[(prev_prev_state, prev_state)][state] *  estimate_emission_probabilites[state][observation]
                        if(curr_prob > max_prob):
                            max_prob = curr_prob
                            viterbi[(prev_state, state)][time_step] = curr_prob
                            back_pointer[(prev_state, state)][time_step] = prev_prev_state

        # termination step
        max_prob = float('-inf')
        for prev_state in states_by_time_step[len_observations-1]:
            for prev_prev_state in states_by_time_step[len_observations-1-1]:
                curr_prob = viterbi[(prev_prev_state, prev_state)][len_observations - 1] *  estimate_transition_probabilites[(prev_prev_state, prev_state)][EOS]
                if(curr_prob > max_prob):
                    max_prob= curr_prob
                    best_path_pointer = (prev_prev_state, prev_state)
                    
        
        result = [best_path_pointer[0], best_path_pointer[1]] + result
        time_step = len_observations - 1
       
        while(result[0] != BOS):
             result = [back_pointer[(result[0], result[1])][time_step]] + result
             time_step -= 1
           
        result = [tag for tag in result if tag != BOS]
      
        return result