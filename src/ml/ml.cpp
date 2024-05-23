#include "ml.h"
#include <memory>
#include <vector>

// KNearest
CvStatus KNearest_Create(KNearest *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::KNearest>(cv::ml::KNearest::create())};
    END_WRAP
}

void KNearest_Close(KNearest *knn) {
    CVD_FREE(knn)
}

CvStatus KNearest_Train(KNearest knn, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*knn.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus KNearest_FindNearest(KNearest knn, Mat samples, int k, Mat results, Mat neighborResponses, Mat dists) {
    BEGIN_WRAP
    (*knn.ptr)->findNearest(*samples.ptr, k, *results.ptr, *neighborResponses.ptr, *dists.ptr);
    END_WRAP
}

// SVM
CvStatus SVM_Create(SVM *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::SVM>(cv::ml::SVM::create())};
    END_WRAP
}

void SVM_Close(SVM *svm) {
    CVD_FREE(svm)
}

CvStatus SVM_Train(SVM svm, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*svm.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus SVM_Predict(SVM svm, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*svm.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// DecisionTree
CvStatus DTrees_Create(DTrees *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::DTrees>(cv::ml::DTrees::create())};
    END_WRAP
}

void DTrees_Close(DTrees *dtree) {
    CVD_FREE(dtree)
}

CvStatus DTrees_Train(DTrees dtree, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*dtree.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus DTrees_Predict(DTrees dtree, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*dtree.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// RandomForest
CvStatus RTrees_Create(RTrees *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::RTrees>(cv::ml::RTrees::create())};
    END_WRAP
}

void RTrees_Close(RTrees *rtree) {
    CVD_FREE(rtree)
}

CvStatus RTrees_Train(RTrees rtree, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*rtree.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus RTrees_Predict(RTrees rtree, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*rtree.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// Boost
CvStatus Boost_Create(Boost *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::Boost>(cv::ml::Boost::create())};
    END_WRAP
}

void Boost_Close(Boost *boost) {
    CVD_FREE(boost)
}

CvStatus Boost_Train(Boost boost, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*boost.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus Boost_Predict(Boost boost, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*boost.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// NeuralNetwork
CvStatus ANN_MLP_Create(ANN_MLP *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::ANN_MLP>(cv::ml::ANN_MLP::create())};
    END_WRAP
}

void ANN_MLP_Close(ANN_MLP *ann) {
    CVD_FREE(ann)
}

CvStatus ANN_MLP_Train(ANN_MLP ann, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*ann.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus ANN_MLP_Predict(ANN_MLP ann, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*ann.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// LogisticRegression
CvStatus LogisticRegression_Create(LogisticRegression *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::LogisticRegression>(cv::ml::LogisticRegression::create())};
    END_WRAP
}

void LogisticRegression_Close(LogisticRegression *logreg) {
    CVD_FREE(logreg)
}

CvStatus LogisticRegression_Train(LogisticRegression logreg, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*logreg.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus LogisticRegression_Predict(LogisticRegression logreg, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*logreg.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}

// NormalBayesClassifier
CvStatus NormalBayesClassifier_Create(NormalBayesClassifier *rval) {
    BEGIN_WRAP
    *rval = {new cv::Ptr<cv::ml::NormalBayesClassifier>(cv::ml::NormalBayesClassifier::create())};
    END_WRAP
}

void NormalBayesClassifier_Close(NormalBayesClassifier *nbayes) {
    CVD_FREE(nbayes)
}

CvStatus NormalBayesClassifier_Train(NormalBayesClassifier nbayes, Mat samples, int layout, Mat responses) {
    BEGIN_WRAP
    (*nbayes.ptr)->train(*samples.ptr, layout, *responses.ptr);
    END_WRAP
}

CvStatus NormalBayesClassifier_Predict(NormalBayesClassifier nbayes, Mat samples, Mat results, int flags) {
    BEGIN_WRAP
    (*nbayes.ptr)->predict(*samples.ptr, *results.ptr, flags);
    END_WRAP
}
