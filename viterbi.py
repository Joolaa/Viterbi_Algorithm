import numpy as np

#Implementation of the Viterbi algorithm.
#
#Given a sequence of observations (events)
#and a Hidden Markov Model, such that the
#given observations are in the set of events
#emitted by the HMM, it returns the most probable
#sequence of hidden states that could have
#emitted the events given as the argument.
def viterbi(events, transit_probs,
            emission_probs, init_probs):
    probs_table = np.zeros((len(events), len(transit_probs)))
    nodes_table = np.zeros((len(events), len(transit_probs)),
                           dtype=np.dtype(int))
    probs_table[0] = np.multiply(emission_probs, init_probs)[events[0]]
    for i in xrange(1, len(events)):
        for j in xrange(len(transit_probs)):
            max_val = -1
            max_ix = -1
            for k in xrange(len(transit_probs)):
                cur_val = probs_table[i - 1, k] * transit_probs[j, k] * (
                    emission_probs[events[i], j])
                if cur_val > max_val:
                    max_val = cur_val
                    max_ix = k
            probs_table[i, j] = max_val
            nodes_table[i, j] = max_ix
    result_indices = []
    previous_index = np.argmax(probs_table[len(probs_table) - 1])
    result_indices.append(previous_index)
    for i in range(len(probs_table) - 1, 0, -1):
        current_index = nodes_table[i, previous_index]
        result_indices.append(current_index)
        previous_index = current_index
    result_indices.reverse()
    return result_indices


#test_transits = np.array([[1.0/3.0, 1.0/3.0, 1.0/3.0, 0],
#                          [1.0/3.0, 1.0/3.0, 0, 1.0/3.0],
#                          [1.0/3.0, 0, 1.0/3.0, 1.0/3.0],
#                          [0, 1.0/3.0, 1.0/3.0, 1.0/3.0]])
#
#test_emit_X = np.array([[0.1, 0.2, 0.7],
#                        [0.3, 0.4, 0.3],
#                        [0.3, 0.4, 0.3],
#                        [0.7, 0.2, 0.1]])
#
#test_emit_Y = np.array([[0.7, 0.2, 0.1],
#                        [0.3, 0.4, 0.3],
#                        [0.2, 0.3, 0.5],
#                        [0.1, 0.2, 0.7]])
#
#def combine(arr1, arr2):
#    result = []
#    for i in xrange(len(arr1)):
#        for j in xrange(len(arr2)):
#            result.append(arr1[i] * arr2[j])
#    return result
#
#test_emissions = np.array([combine(test_emit_X[i], test_emit_Y[i]) for i in
#                           xrange(len(test_emit_X))]).transpose()
#
#test_inits = np.array([0.2, 0.3, 0.3, 0.2])
#
#test_obs = np.array([8, 4, 2, 1, 7, 4])
#############################
#viterbi(test_obs, test_transits, test_emissions, test_inits)
#=> [2, 2, 3, 1, 0, 1]
