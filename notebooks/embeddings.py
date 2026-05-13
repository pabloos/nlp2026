import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

model = api.load("word2vec-google-news-300")
# dirección género
gender = model["man"] - model["woman"]

def projection(word):
    return np.dot(model[word], gender)

words = ["king", "queen", "prince", "princess", "man", "woman"]

for w in words:
    print(w, projection(w))

words = ["king", "queen", "man", "woman", "dog", "cat", "car", "plane"]
X = np.array([model[w] for w in words])

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

for i, word in enumerate(words):
    plt.scatter(X_2d[i,0], X_2d[i,1])
    plt.text(X_2d[i,0], X_2d[i,1], word)

plt.show()