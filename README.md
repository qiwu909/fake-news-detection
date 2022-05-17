# fake-news-detection
In this report, we will implement public dataset from GitHub to try automatically predict fake news from the dataset. We compared several preprocessing methods and classifiers and finally come up a model that reached above 90% accuracy.

 RNN vs GRU vs LSTM: 
 Recurrent Neural Networks are designed to work with sequential data. RNN feeds the words one by one in the sentence into the network. This process goes until all words in the sentence are given input. However, RNN has a disadvantage about the memory. As the number of sentences becomes bigger, it suffers from vanishing gradient. Why is this happening? The gradient is used to update weights in the network. If the effect of the previous layer on the current layer is small then the gradient value will be small and vice-versa. If the gradient of the previous layer is smaller than the gradient of the current layer will be even smaller. As a result, when we proceed the process, the gradient of previous layer shrinks down. 
 

Fig. 1. RNN basic architecture.

In order to solve this problem, another two specialized versions of RNN were introduced. The first one is called GRU (Gated Recurrent Unit). Fig.1 was shown to explain the architecture of it. The workflow of GRU is the same as RNN but the difference is the operations inside the GRU unit. There are two gates inside GRU. Update gate is designed to decide if the cell state should be updated with the candidate state. Reset gate is used to measure whether the previous cell state is important. If reset close to 0, we can ignore previous hidden state and allow the model to drop previous irrelevant data. If gamma (update gate) close to 1, then we can remember information in that unit. In a word, gates are capable of learning which inputs in the sequence are important and capable of storing the data in the memory unit. 
 
Fig. 2. GRU basic architecture.

Additional to GRU here there are two more gates in LSTM. As shown in the Fig.3, LSTM works with all three gates (input gate, output gate, and forget gate). 
 
Fig. 3. LSTM basic architecture.

Word and Sentence Embedding
	As we know in part 1), GRU were adopted to collect the annotations’ contextual information. A bidirectional GRU contains a forward GRU and a backward GRU. Forward GRU Gf reads the ith sentence from word x1i to xmi, while backward GRU Gb reads the ith sentence from word xmi to xli.

O_f=GRU(x_t^i ),t∈{1,…,m_1}
O_b=GRU(x_t^i ),t∈{m_i,…,1}

The same approach used for word encoding is used to encode sentences. The RNN with GRU units is utilized to encode each sentence of news. The annotated word vector Okt is used to learn the sentence representation Sk by using the bidirectional GRU units. It can be shown mathematically as follows:
 
S_f^k=GRU(O_t^k ),k∈{1,2…,N}
S_b^k=GRU(O_t^k ),k∈{1,2…,N}

Both forward and backward annotations were concatenated to obtain the sentence annotationS_k=<S_f^k,S_b^k> The notation Sk apprehends the contextual information from the sentences that lie in the locality of sentence k.
Comments Embedding
	User comments are equally vital in this scenario. Since people reflect differently on different styles of articles, user comments are valuable samples to learn about news content. With the help of bidirectional GRU, we can collect the comments, reactions and skeptical opinions then find out clues to discriminate between real and fake news.
C. Vectorization
There are basically two methods of vectorization. Their major difference is whether all features should be assumed independent. There are three main vectorization tools: a Count Vectorizer (Bag of Words), a Term Frequency-Inverse Document Frequency Vectorizer (TF-IDF), and One-Hot representation into an embedding layer. The bag of words vectorizer collects unique words from the text and the numerical instances of each word. The TF-IDF vectorizer operates similarly, except words that are common throughout all of the data have their weights, or relative importance, decreased. This algorithm produces an arguably more accurate representation of the words in the text and their relative importance. The one-hot encoding into an embedding layer essentially creates, for each article, a multi-dimensional vector representing a sequence of words. This method of encoding takes into account the sequential nature of writing: in the one-hot representation, the sentences "live to eat" and "eat to live" are distinct, whereas, in the previous two methods, they can’t recognize.
D. Classification
As we discussed in Subsection 3, there are two kinds of classifiers in cooperate with two vectorization method. Multinomial Naïve Bayes model implements Bayes theorem and conditional probabilities to calculate a probability for the article belonging in each class. It has an advantage of a fast, simple model because it assumes that all features are independent. The Passive-Aggressive classifier, rather than using the dataset as a whole, takes in one piece at a time, adjusting the weights of its model based on each entry's results. However, the LSTM model studies human interpretation, takes articles as a sequence of words. With one-hot encoding, the order of the text is preserved. So, it is ideal to use the LSTM model. 
