
def VITERBI_Lists(state_transition_probmat, initial_state_prob):



    viterbi_mat = []
    backpointer = []
    viterbi_1stLayer = []
    for i in range(len(initial_state_prob)):
        viterbi_1stLayer.append(float(initial_state_prob[i]))
    viterbi_mat.append(viterbi_1stLayer)

    for time_step in range(len(state_transition_probmat)):
        viterbi_layer = []
        backpointer_layer = []
        for state in range(len(state_transition_probmat[time_step])):
            iteration_vec = [viterbi_mat[time_step][i]*state_transition_probmat[time_step][state][i] for i in range(len(viterbi_mat[time_step]))]

            maxval = max(iteration_vec)
            maxind = iteration_vec.index(maxval)
            viterbi_layer.append(maxval)
            # max_index = max(range(len(state_vec)), key=lambda i: state_vec[i])
            backpointer_layer.append(maxind)
          
        viterbi_mat.append(viterbi_layer)
        backpointer.append(backpointer_layer)
    
    best_path_prob = max(viterbi_mat[-1])
    # max_index = max(range(len(viterbi_mat[-1])), key = lambda i: viterbi_mat[-1][i])
    max_index = viterbi_mat[-1].index(best_path_prob)
    best_backpointer = max_index
    best_path = [best_backpointer]
    j = 0
    for i in reversed(range(len(state_transition_probmat))):
        best_path.append(backpointer[i][best_path[j]])
        j += 1
    best_path = best_path[::-1]
    return best_path, viterbi_mat
