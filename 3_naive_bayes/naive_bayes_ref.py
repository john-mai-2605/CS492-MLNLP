import csv
import random
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

"""
# Naive Bayes Classifier

In this task, you will implement Naive Bayes classifier for binary sentiment classification.

Implement four methods:
- `MyNaiveBayes.fit(self, bows, labels) -> None`
- `MyNaiveBayes.get_prior(self) -> prior`
- `MyNaiveBayes.get_likelihood_with_smoothing(self) -> likelihood`
- `MyNaiveBayes.predict(self, bows) -> labels`

## Instruction

* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.
* Do not use scikit-learn Naive Bayes.
* For your information, TA's code got 0.8415 validation accuracy in 13s with TA's laptop.

Useful numpy methods for efficient computation:
- https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
- https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
- https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html
"""


def _download_dataset(size=10000):
    import sys
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
        print("Download: {}".format(file_path))
    else:
        print("Already exist: {}".format(file_path))


def _get_review_data(path, num_samples, train_test_ratio=0.8):
    """Do not modify the code in this function."""
    _download_dataset()
    print("Load Data at {}".format(path))
    reviews, sentiments = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            reviews.append(line["review"])
            sentiments.append(int(line["sentiment"]))

    # Data shuffle
    random.seed(42)
    zipped = list(zip(reviews, sentiments))
    random.shuffle(zipped)
    reviews, sentiments = zip(*(zipped[:num_samples]))
    reviews, sentiments = np.asarray(reviews), np.asarray(sentiments)

    # Train/test split
    num_data, num_train = len(sentiments), int(len(sentiments) * train_test_ratio)
    return (reviews[:num_train], sentiments[:num_train]), (reviews[num_train:], sentiments[num_train:])


def _create_bow(sentences, vectorizer=None, msg_prefix="\n"):
    """Make the Bag-of-Words model from the sentences, return (vectorizer, sentence_vectors)

    :param sentences: (array_like): array_like objects of strings
    :param vectorizer: (CountVectorizer, optional)
    :param msg_prefix: (str)
    :return: Tuple[CountVectorizer, array_like]
    """
    print("{} Bow construction".format(msg_prefix))
    if vectorizer is None:
        vectorizer = CountVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
    else:
        sentence_vectors = vectorizer.transform(sentences)
    return vectorizer, sentence_vectors.toarray()


class MyNaiveBayes:

    def __init__(self, num_vocab, num_classes):
        self.num_classes = num_classes
        self.num_vocab = num_vocab

        self.class_to_num_sentences = np.zeros(self.num_classes)
        self.class_and_word_to_counts = np.zeros((self.num_classes, self.num_vocab))

        self.log_prior = None
        self.log_likelihood = None

    def fit(self, bows, labels):
        """Compute self.class_to_num_sentences and self.class_and_word_to_counts,
            - class_to_num_sentences[c]: number of sentences the class of which is 'c'.
            - class_and_word_to_counts[c, w]: number of the word 'w' appeared in sentences of class 'c'.

        e.g., if bows are [[42, 2, 0], [2, 0, 1]] and labels are [0, 1],
            class_to_num_sentences[0] will be number of the sentences the class of which is 0, i.e., 1.
            class_and_word_to_counts[0, 0] will be number of the word 0 appeared in sentences of class 0, i.e., 42.

        And get log_prior and log_likelihood with these.

        :param bows: ndarray, the shape of which is (num_batches, num_vocab)
        :param labels: ndarray, the shape of which is (num_batches,)
        """
        # Compute self.class_to_num_sentences and self.class_and_word_to_counts
        for c in range(self.num_classes):
            self.class_to_num_sentences[c] = np.sum(labels == c)

        for bow, label in tqdm(zip(bows, labels), total=len(bows)):
            self.class_and_word_to_counts[label] += bow

        # Get log_prior and log_likelihood with these. (Do not modify below three lines.)
        self.log_prior = np.log(self.get_prior())
        self.log_likelihood = np.log(self.get_likelihood_with_smoothing())
        self._check()

    def get_prior(self):
        """Get prior, P(c)

        e.g., if there are 48 negative (0) labels and 52 positive (1) labels,
        P[0] = 48 / 100 = 0.48.

        :return ndarray P, the shape of which is (num_classes,)
            where P[c] is the prior of class c.
        """
        return self.class_to_num_sentences / np.sum(self.class_to_num_sentences)

    def get_likelihood_with_smoothing(self):
        """Get likelihood, P(w|c).

        i.e., p(w|c) = [# of the word 'w' appeared in the sentences the class of which is 'c']
                        / [# of the total tokens in the sentences the class of which is 'c']
        Note that this example did not apply smoothing.

        :return ndarray P, the shape of which is (num_classes, num_vocab)
            where P[c, w] is the likelihood of word w and given class c.
        """
        likelihood_list = []
        for c in range(self.num_classes):
            likelihood = (self.class_and_word_to_counts[c] + 1) \
                         / (np.sum(self.class_and_word_to_counts[c]) + self.num_vocab)
            likelihood_list.append(likelihood)
        return np.asarray(likelihood_list)

    def predict(self, bows):
        """Predict labels (0 or 1) by posterior, p(c_k|w_1, ..., w_n) ~ p(c_k) \prod_{i=1}^{n} p(w_i|c_k)

        Use log-probabilities (self.log_prior and self.log_likelihood) instead of vanilla probabilities.
        This works and is rather elegant for the following reasons.
        - A product of probabilities (i.e., \prod_{i=1}^{n} p(w_i|c_k)) is usually a small float point number,
        which a computer cannot handle. The computer easily regards small numbers as zero.
        - log(.) is a monotonically increasing function, so if a < b, then log a < log b. This property
        makes comparing log-probabilities be equivalent to comparing probabilities (e.g., argmax).
        - We can easily transform a product of numbers to a sum of log-numbers.

        i.e., log_posterior of c_k where (w_1, ..., w_n) are given
            = log p(c_k|w_1, ..., w_n)
            = log p(c_k) \prod_{i=1}^{n} p(w_i|c_k)
            = log p(c_k) + log \prod_{i=1}^{n} p(w_i|c_k)
            = log p(c_k) + \sum_{i=1}^{n} log p(w_i|c_k)
            = log_prior[c_k] + \sum_{i=1}^{n} log_likelihood[c_k, w_i]
        Note that 'n' is not self.num_vocab, it is the number of total tokens in the sentence.

        :param bows: ndarray, the shape of which is (num_batches, num_vocab)
        :return ndarray L, the shape of which is (num_batches,)
            where L[i] is the label of the ith sample.
        """
        labels = []
        for bow in tqdm(bows):
            log_posterior = self.log_prior + np.sum(self.log_likelihood * bow, axis=1)
            labels.append(np.argmax(log_posterior))
        return np.asarray(labels)

    def _check(self):
        """Do not modify the code in this function."""
        assert self.log_prior is not None and self.log_likelihood is not None
        assert self.class_to_num_sentences.sum() > 0 and self.class_and_word_to_counts.sum() > 0
        assert self.log_prior.shape[0] == self.num_classes and len(self.log_prior.shape) == 1
        assert self.log_likelihood.shape[0] == self.num_classes and self.log_likelihood.shape[1] == self.num_vocab


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/review_10k.csv", num_samples=num_samples)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))

    # Create bow representation of train set
    count_vectorizer, train_bows = _create_bow(train_xs, msg_prefix="\n[Train]")
    counted = len(count_vectorizer.get_feature_names())
    if verbose:
        print("\n[Vocab]: {} words".format(counted))

    clf = MyNaiveBayes(num_vocab=counted, num_classes=2)
    clf.fit(train_bows, train_ys)
    if verbose:
        print("\n[MyNaiveBayes] Training Complete")

    # Create bow representation of validation set
    _, val_bows = _create_bow(val_xs, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")

    # Evaluation
    val_preds = clf.predict(val_bows)
    val_accuracy = accuracy_score(val_ys, val_preds)
    if verbose:
        print("\n[Validation] Accuracy: {}".format(val_accuracy))

    # Grading: Do not modify below lines.
    if test_xs is not None:
        _, test_bows = _create_bow(test_xs, vectorizer=count_vectorizer, msg_prefix="\n[Test]")
        test_preds = clf.predict(test_bows)
        return {"clf": clf, "val_accuracy": val_accuracy, "test_accuracy": accuracy_score(test_ys, test_preds)}
    else:
        return {"clf": clf}


if __name__ == '__main__':
    run()
