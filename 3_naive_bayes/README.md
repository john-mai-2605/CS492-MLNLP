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
