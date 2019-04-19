import warnings

from numpy.ma import average
from sklearn import svm
from sklearn.model_selection import cross_validate

from Exercise_2a.DataSet import DataSet

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    trainSet: DataSet = DataSet('train.csv')
    testSet: DataSet = DataSet('test.csv')

    def run_with_kernel(kernel):
        classifier = svm.SVC(kernel=kernel)
        classifier.fit(trainSet.data, trainSet.labels)

        # Three fold cross validation
        cross_validation = cross_validate(classifier, testSet.data, testSet.labels, return_estimator=True)
        estimators = cross_validation['estimator']
        score = cross_validation['test_score']

        print("Kernel: " + kernel)
        print("Average Cross Validation Score: " + str(average(score)))
        print("C: " + str(average([estimator.C for estimator in estimators])))
        print("Gamma: " + str(average([estimator._gamma for estimator in estimators])))
        print()


    run_with_kernel('linear')
    run_with_kernel('rbf')






