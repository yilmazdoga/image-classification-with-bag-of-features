from sklearn import svm
from sklearn.preprocessing import LabelEncoder


class Classifier:
    def __init__(self, histograms, labels):
        self.classifier = svm.SVC()
        self.label_encoder = LabelEncoder()
        self.histograms = histograms
        self.labels = labels
        self.labels_as_int = self.label_encoder.fit_transform(self.labels)

    def train(self):
        self.classifier.fit(self.histograms, self.labels_as_int)

    def predict(self, histograms):
        return self.label_encoder.inverse_transform(self.classifier.predict(histograms))
