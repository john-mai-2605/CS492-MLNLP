# Korean NLP

In this task, we use a Korean dataset named Naver Sentiment Movie Corpus (NSMC).
We are going to solve a binary sentiment classification problem again.
However, if we run an RNN model that we built last time, you can observe a serious overfitting problem.
To handle this overfitting problem, I suggest you to try sub-word models.
Here, we are going to use a Byte Pair Encoder and a Wordpiece model.
You have to implement some parts of a Byte Pair Encoder and a bidirectional RNN model.
See how the performance is changed according to the encoding methods.

