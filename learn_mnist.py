#!/bin/env python
# -*- coding: UTF-8 -*-

def main():
    from datetime import datetime
    start_datetime = datetime.now()
    print('start: %s' % (start_datetime,))

    mnist_data_train, mnist_data_test, mnist_label_train, mnist_label_test = load_mnist()

    # confirm data on console
    # print(mnist_data_train)
    # print(mnist_data_test)
    # print(mnist_label_train)
    # print(mnist_label_test)

    estimator = create_estimator()
    estimator.fit(mnist_data_train, mnist_label_train)

    mnist_label_predict = estimator.predict(mnist_data_test)

    # metrics
    from sklearn.metrics import precision_score, recall_score

    precision_micro = precision_score(mnist_label_test, mnist_label_predict, average = 'micro')
    recall_micro = recall_score(mnist_label_test, mnist_label_predict, average = 'micro')
    print('precision_micro = %f, recall_micro = %f' % (precision_micro, recall_micro))

    precision_macro = precision_score(mnist_label_test, mnist_label_predict, average = 'macro')
    recall_macro = recall_score(mnist_label_test, mnist_label_predict, average = 'macro')
    print('precision_macro = %f, recall_macro = %f' % (precision_macro, recall_macro))

    precision_None = precision_score(mnist_label_test, mnist_label_predict, average = None)
    recall_None = recall_score(mnist_label_test, mnist_label_predict, average = None)
    print('precision_None = %s, recall_None = %s' % (precision_None, recall_None))

    end_datetime = datetime.now()
    print('end: %s' % (end_datetime,))
    elapsed = end_datetime - start_datetime
    print('elapsed: %s' % (elapsed,))

def load_mnist():
    import numpy as np
    from sklearn.datasets import fetch_mldata

    # original data load
    mnist = fetch_mldata('MNIST original')

    import pandas as pd

    # transform data
    from sklearn.model_selection import train_test_split
    mnist_data = pd.DataFrame(mnist.data, dtype = 'float64')
    mnist_label = pd.Series(mnist.target)

    # clear variable "mnist" for memory capacity
    mnist = None

    return train_test_split(mnist_data, mnist_label, test_size = 10000, random_state = 10)

def create_estimator():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    return Pipeline([('scaler', StandardScaler()), ('classifier', SVC(gamma = 'auto', kernel = 'rbf'))])



if __name__ == '__main__':
    main()
