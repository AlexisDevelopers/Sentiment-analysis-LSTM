This project consists of a sentiment analysis using a LSTM (Long Short-Term Memory) neural network to classify whether a text has a positive or negative connotation.

First, the necessary libraries are imported, such as Tensorflow and tools for loading and pre-processing data from the IMDB database, which contains movie reviews labeled as positive or negative. The maximum dimensions of the vocabulary and the sequences of words that will be used are established.

The training and test data is then loaded using the imdb.load_data() function. Each comment is represented as a sequence of word indexes, where each index represents a vocabulary word. The full vocabulary is also printed using imdb.get_word_index().

A reverse dictionary is generated that allows a sequence of word indexes to be decoded into readable text. The training and testing sequences are preprocessed by keras.preprocessing.sequence.pad_sequences() to have the maximum length set.

From here begins the code to build the model. A sequential model is created (Sequential()) and layers are added to it. First, the Embedding() layer is used to transform each word index into a dense vector of fixed length. Two LSTM() layers of 64 and 32 units respectively are then added, which allow the neural network to capture long-term dependencies on the word sequences. Finally, a Dense() layer is added with a unit and a sigmoid activation function, to produce the binary output indicating whether the comment is positive or negative.