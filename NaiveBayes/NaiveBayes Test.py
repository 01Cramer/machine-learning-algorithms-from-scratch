from NaiveBayes import NaiveBayes

X = [["żywe", "tak", "4", "nie"], ["żywe", "tak", 4, "nie"], ["jajo", "nie", 2, "tak"]]
y = ["ssaki", "ssaki", "ptaki"]
clasifier = NaiveBayes()
clasifier.fit(X, y)
clasifier._predict([["żywe", "tak", 2, "nie"]])
