import numpy as np

def convert_to_labels(indices, labels):
    result = []
    for index in indices:
        result.append(labels[index])
    return result

#Implementation of the Viterbi algorithm.
#
#Given a sequence of observations (events)
#and a Hidden Markov Model, such that the
#given observations are in the set of events
#emitted by the HMM, it returns the most probable
#sequence of hidden states that could have
#emitted the events given as the argument.
def viterbi(events, transit_probs,
            emission_probs, init_probs,
            event_labels):
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
    result_probs = []
    result_probs.append(np.amax(probs_table[len(probs_table) - 1]))
    previous_index = np.argmax(probs_table[len(probs_table) - 1])
    result_indices.append(previous_index)
    for i in range(len(probs_table) - 1, 0, -1):
        current_index = nodes_table[i, previous_index]
        result_indices.append(current_index)
        result_probs.append(probs_table[i, previous_index])
        previous_index = current_index
    result_indices.reverse()
    result_probs.reverse()
    if event_labels:
        result_indices = convert_to_labels(result_indices, event_labels)
    return result_indices, result_probs


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
#test_labels = ["A", "B", "C", "D"]
#############################
#viterbi(test_obs, test_transits, test_emissions, test_inits, test_labels)
#=> (['C', 'C', 'D', 'B', 'A', 'B'],
#    [0.0018,
#     0.00029399999999999994,
#     1.1759999999999996e-05,
#     5.4879999999999962e-07,
#     2.9269333333333318e-08,
#     2.9269333333333318e-08])

