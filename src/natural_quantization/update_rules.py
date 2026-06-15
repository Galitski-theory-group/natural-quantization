def gradient_descent(
    hidden_layer_n, weights, bias, η, minibatch_pool, a_errors, b_errors
):
    for l_i in range(0, hidden_layer_n + 1):
        weights[l_i] -= η / minibatch_pool * a_errors[l_i]
        bias[l_i] -= η / minibatch_pool * b_errors[l_i]
