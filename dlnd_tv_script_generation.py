import numpy as np
import tensorflow as tf


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int, int_to_vocab = {}, {}

    for i, word in enumerate(set(text)):
        vocab_to_int[word] = i
        int_to_vocab[i] = word

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    return {
        '.': '||Period||',
        ',': '||Comma||',
        '"': '||Quotation_Mark||',
        ';': '||Semicolon||',
        '!': '||Exclamation_Mark||',
        '?': '||Question_Mark||',
        '(': '||Left_Parentheses||',
        ')': '||Right_Parentheses||',
        '--': '||Dash||',
        '\n': '||Return||',
    }


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input_ = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
    learning_rate = tf.placeholder(tf.int32, shape=None, name='learning_rate')

    return input_, targets, learning_rate


def get_init_cell(batch_size, rnn_size, lstm_layers=2):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
    # Getting an initial state of all zeros
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')

    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))

    return tf.nn.embedding_lookup(embedding, input_data)


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    outputs, final_state = build_rnn(cell, inputs=get_embed(input_data,
                                                            vocab_size=vocab_size,
                                                            embed_dim=embed_dim))

    logits = tf.contrib.layers.fully_connected(outputs,
                                               num_outputs=vocab_size,
                                               activation_fn=None)

    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    batch_chars = batch_size * seq_length
    n_batches = len(int_text) // batch_chars

    # only full batches
    int_text = int_text[: n_batches * batch_chars]

    x = np.array(int_text)
    # shift first batch by one
    y = np.array(int_text[1:] + [int_text[0]])

    x_batches = np.split(x.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(y.reshape(batch_size, -1), n_batches, 1)

    batches = [(x_batch, y_batch) for x_batch, y_batch in zip(x_batches, y_batches)]

    return np.array(batches)


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_ = loaded_graph.get_tensor_by_name("input:0")
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")

    return input_, initial_state, final_state, probs


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return int_to_vocab[np.argmax(probabilities)]


def main():
    import problem_unittests as tests

    tests.test_get_init_cell(get_init_cell)
    tests.test_get_embed(get_embed)
    tests.test_build_rnn(build_rnn)
    tests.test_build_nn(build_nn)
    tests.test_get_batches(get_batches)
    tests.test_get_tensors(get_tensors)
    tests.test_pick_word(pick_word)

    print(get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      batch_size=3,
                      seq_length=2))


if __name__ == '__main__':
    main()
