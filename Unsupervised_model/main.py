import numpy as np
from PCA import PCA as MyPCA
from SVD import SVD as MySVD

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request
from PIL import Image

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

x = mnist.data.astype('float32')
y = mnist.target.astype('int64')

x, y = x[:10000], y[:10000]

def model(img, method):
    if method == "PCA":
        # Dimensionality reduction using PCA
        pca = MyPCA(n_components=50)
        pca.fit(x)
        x_train_model = pca.fit_transform(x)
        x_train_model = np.real(x_train_model)
        x_train, x_test, y_train, y_test = train_test_split(x_train_model, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(penalty='none', random_state=0, max_iter=1000)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)

        img_reduce = pca.fit_transform(img)
        img_reduce = np.real(img_reduce)
        predicted_cluster = clf.predict(img_reduce)
        return accuracy, predicted_cluster

    elif method == "SVD":
        # Dimensionality reduction using SVD
        svd = MySVD(n_components=70)
        svd.fit(x)
        x_train_model = svd.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_train_model, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(penalty='none', random_state=0, max_iter=1000)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)

        img_reduce = svd.fit_transform(img)
        img_reduce = np.real(img_reduce)
        predicted_cluster = clf.predict(img_reduce)
        return accuracy, predicted_cluster
    else:
        raise ValueError("Invalid method selected.")

def convert_image(image_file):
    img = Image.open(image_file)
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    image_reshape = img_array.reshape(1, -1)
    return image_reshape

app = Flask(__name__)

@app.route('/digit', methods=['POST'])
def convert_image_to_array():
    # get the image file from the request
    image_file = request.files['image']
    method = request.values['data']

    image_array = convert_image(image_file)

    accuracy, predict = model(image_array, method)

    # return the array as a JSON response
    return {'accuracy': accuracy, 'predict': int(predict[0])}

if __name__ == '__main__':
    app.run()