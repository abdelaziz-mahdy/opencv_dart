#ifndef _OPENCV3_ML_H_
#define _OPENCV3_ML_H_

#include "core/core.h"
#include <stdbool.h>

#ifdef __cplusplus
#include <opencv2/ml.hpp>
extern "C" {
#endif

#ifdef __cplusplus
CVD_TYPEDEF(cv::Ptr<cv::ml::NormalBayesClassifier>, NormalBayesClassifier)
CVD_TYPEDEF(cv::Ptr<cv::ml::KNearest>, KNearest)
CVD_TYPEDEF(cv::Ptr<cv::ml::SVM>, SVM)
CVD_TYPEDEF(cv::Ptr<cv::ml::EM>, EM)
CVD_TYPEDEF(cv::Ptr<cv::ml::DTrees>, DTrees)
CVD_TYPEDEF(cv::Ptr<cv::ml::RTrees>, RTrees)
CVD_TYPEDEF(cv::Ptr<cv::ml::Boost>, Boost)
CVD_TYPEDEF(cv::Ptr<cv::ml::ANN_MLP>, ANN_MLP)
CVD_TYPEDEF(cv::Ptr<cv::ml::LogisticRegression>, LogisticRegression)
CVD_TYPEDEF(cv::Ptr<cv::ml::SVMSGD>, SVMSGD)
#else
CVD_TYPEDEF(void, NormalBayesClassifier)
CVD_TYPEDEF(void, KNearest)
CVD_TYPEDEF(void, SVM)
CVD_TYPEDEF(void, EM)
CVD_TYPEDEF(void, DTrees)
CVD_TYPEDEF(void, RTrees)
CVD_TYPEDEF(void, Boost)
CVD_TYPEDEF(void, ANN_MLP)
CVD_TYPEDEF(void, LogisticRegression)
CVD_TYPEDEF(void, SVMSGD)
#endif

CVD_TYPEDEF_PTR(NormalBayesClassifier)
CVD_TYPEDEF_PTR(KNearest)
CVD_TYPEDEF_PTR(SVM)
CVD_TYPEDEF_PTR(EM)
CVD_TYPEDEF_PTR(DTrees)
CVD_TYPEDEF_PTR(RTrees)
CVD_TYPEDEF_PTR(Boost)
CVD_TYPEDEF_PTR(ANN_MLP)
CVD_TYPEDEF_PTR(LogisticRegression)
CVD_TYPEDEF_PTR(SVMSGD)

CvStatus NormalBayesClassifier_Create(NormalBayesClassifier* rval);
CvStatus KNearest_Create(KNearest* rval);
CvStatus SVM_Create(SVM* rval);
CvStatus EM_Create(EM* rval);
CvStatus DTrees_Create(DTrees* rval);
CvStatus RTrees_Create(RTrees* rval);
CvStatus Boost_Create(Boost* rval);
CvStatus ANN_MLP_Create(ANN_MLP* rval);
CvStatus LogisticRegression_Create(LogisticRegression* rval);
CvStatus SVMSGD_Create(SVMSGD* rval);

void NormalBayesClassifier_Close(NormalBayesClassifier* self);
void KNearest_Close(KNearest* self);
void SVM_Close(SVM* self);
void EM_Close(EM* self);
void DTrees_Close(DTrees* self);
void RTrees_Close(RTrees* self);
void Boost_Close(Boost* self);
void ANN_MLP_Close(ANN_MLP* self);
void LogisticRegression_Close(LogisticRegression* self);
void SVMSGD_Close(SVMSGD* self);

CvStatus NormalBayesClassifier_Train(NormalBayesClassifier self, Mat samples, int layout, Mat responses);
CvStatus KNearest_Train(KNearest self, Mat samples, int layout, Mat responses);
CvStatus SVM_Train(SVM self, Mat samples, int layout, Mat responses);
CvStatus EM_Train(EM self, Mat samples, int layout, Mat responses);
CvStatus DTrees_Train(DTrees self, Mat samples, int layout, Mat responses);
CvStatus RTrees_Train(RTrees self, Mat samples, int layout, Mat responses);
CvStatus Boost_Train(Boost self, Mat samples, int layout, Mat responses);
CvStatus ANN_MLP_Train(ANN_MLP self, Mat samples, int layout, Mat responses);
CvStatus LogisticRegression_Train(LogisticRegression self, Mat samples, int layout, Mat responses);
CvStatus SVMSGD_Train(SVMSGD self, Mat samples, int layout, Mat responses);

CvStatus NormalBayesClassifier_Predict(NormalBayesClassifier self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus KNearest_Predict(KNearest self, Mat samples, CVD_OUT Mat* results, int k, float* rval);
CvStatus SVM_Predict(SVM self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus EM_Predict(EM self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus DTrees_Predict(DTrees self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus RTrees_Predict(RTrees self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus Boost_Predict(Boost self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus ANN_MLP_Predict(ANN_MLP self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus LogisticRegression_Predict(LogisticRegression self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);
CvStatus SVMSGD_Predict(SVMSGD self, Mat samples, CVD_OUT Mat* results, int flags, float* rval);

CvStatus KNearest_FindNearest(KNearest self, Mat samples, int k, CVD_OUT Mat* results, CVD_OUT Mat* neighborResponses, CVD_OUT Mat* dist, float* rval);

#ifdef __cplusplus
}
#endif

#endif //_OPENCV3_ML_H_
