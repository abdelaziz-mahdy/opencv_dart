/* Created by Rainyl. Licensed: Apache 2.0 license. Copyright (c) 2024 Rainyl. */
#ifndef _OPENCV3_ML_H_
#define _OPENCV3_ML_H_

#include "core/core.h"
#include <stdbool.h>

#ifdef __cplusplus
#include <opencv2/ml.hpp>
extern "C" {
#endif

#ifdef __cplusplus
CVD_TYPEDEF(cv::ml::ParamGrid, ParamGrid)
CVD_TYPEDEF(cv::Ptr<cv::ml::ParamGrid>, PtrParamGrid)
CVD_TYPEDEF(cv::ml::TrainData, TrainData)
CVD_TYPEDEF(cv::Ptr<cv::ml::TrainData>, PtrTrainData)
CVD_TYPEDEF(cv::Ptr<cv::ml::StatModel>, PtrStatModel)
CVD_TYPEDEF(cv::Ptr<cv::ml::NormalBayesClassifier>, PtrNormalBayesClassifier)
CVD_TYPEDEF(cv::Ptr<cv::ml::KNearest>, PtrKNearest)
CVD_TYPEDEF(cv::Ptr<cv::ml::SVM>, PtrSVM)
CVD_TYPEDEF(cv::Ptr<cv::ml::EM>, PtrEM)
CVD_TYPEDEF(cv::Ptr<cv::ml::DTrees>, PtrDTrees)
CVD_TYPEDEF(cv::Ptr<cv::ml::RTrees>, PtrRTrees)
CVD_TYPEDEF(cv::Ptr<cv::ml::Boost>, PtrBoost)
CVD_TYPEDEF(cv::Ptr<cv::ml::ANN_MLP>, PtrANN_MLP)
CVD_TYPEDEF(cv::Ptr<cv::ml::LogisticRegression>, PtrLogisticRegression)
CVD_TYPEDEF(cv::Ptr<cv::ml::SVMSGD>, PtrSVMSGD)
typedef cv::InputArray CVInputArray;
typedef cv::OutputArray CVOutputArray;
typedef cv::String CVString;
typedef cv::Ptr<cv::ml::SVM::Kernel> PtrSVMKernel;
#else
CVD_TYPEDEF(void, ParamGrid)
CVD_TYPEDEF(void, TrainData)
CVD_TYPEDEF(void *, PtrParamGrid)
CVD_TYPEDEF(void *, PtrTrainData)
CVD_TYPEDEF(void *, PtrStatModel)
CVD_TYPEDEF(void *, PtrNormalBayesClassifier)
CVD_TYPEDEF(void *, PtrKNearest)
CVD_TYPEDEF(void *, PtrSVM)
CVD_TYPEDEF(void *, PtrEM)
CVD_TYPEDEF(void *, PtrDTrees)
CVD_TYPEDEF(void *, PtrRTrees)
CVD_TYPEDEF(void *, PtrBoost)
CVD_TYPEDEF(void *, PtrANN_MLP)
CVD_TYPEDEF(void *, PtrLogisticRegression)
CVD_TYPEDEF(void *, PtrSVMSGD)
typedef void *CVInputArray;
typedef void *CVOutputArray;
typedef void *CVString;
typedef void *PtrSVMKernel;
#endif

CVD_TYPEDEF_PTR(ParamGrid)
CVD_TYPEDEF_PTR(PtrParamGrid)
CVD_TYPEDEF_PTR(PtrTrainData)
CVD_TYPEDEF_PTR(PtrStatModel)
CVD_TYPEDEF_PTR(PtrNormalBayesClassifier)
CVD_TYPEDEF_PTR(PtrKNearest)
CVD_TYPEDEF_PTR(PtrSVM)
CVD_TYPEDEF_PTR(PtrEM)
CVD_TYPEDEF_PTR(PtrDTrees)
CVD_TYPEDEF_PTR(PtrRTrees)
CVD_TYPEDEF_PTR(PtrBoost)
CVD_TYPEDEF_PTR(PtrANN_MLP)
CVD_TYPEDEF_PTR(PtrLogisticRegression)
CVD_TYPEDEF_PTR(PtrSVMSGD)

CvStatus ParamGrid_New(ParamGrid *rval);
CvStatus ParamGrid_NewWithParams(double _minVal, double _maxVal, double _logStep, ParamGrid *rval);
CvStatus ParamGrid_Create(double minVal, double maxVal, double logstep, PtrParamGrid *rval);

CvStatus TrainData_MissingValue(float *rval);
CvStatus TrainData_Close(PtrTrainData *cs);
CvStatus TrainData_GetLayout(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetNTrainSamples(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetNTestSamples(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetNSamples(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetNVars(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetNAllVars(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetSample(const PtrTrainData *cs, CVInputArray varIdx, int sidx, float* buf);
CvStatus TrainData_GetSamples(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetMissing(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainSamples(const PtrTrainData *cs, int layout, bool compressSamples, bool compressVars, Mat *rval);
CvStatus TrainData_GetTrainResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainNormCatResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTestResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTestNormCatResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetNormCatResponses(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetSampleWeights(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainSampleWeights(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTestSampleWeights(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetVarIdx(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetVarType(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetVarSymbolFlags(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetResponseType(const PtrTrainData *cs, int *rval);
CvStatus TrainData_GetTrainSampleIdx(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetTestSampleIdx(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetValues(const PtrTrainData *cs, int vi, CVInputArray sidx, float* values);
CvStatus TrainData_GetNormCatValues(const PtrTrainData *cs, int vi, CVInputArray sidx, int* values);
CvStatus TrainData_GetDefaultSubstValues(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetCatCount(const PtrTrainData *cs, int vi, int *rval);
CvStatus TrainData_GetClassLabels(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetCatOfs(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetCatMap(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_SetTrainTestSplit(PtrTrainData *cs, int count, bool shuffle);
CvStatus TrainData_SetTrainTestSplitRatio(PtrTrainData *cs, double ratio, bool shuffle);
CvStatus TrainData_ShuffleTrainTest(PtrTrainData *cs);
CvStatus TrainData_GetTestSamples(const PtrTrainData *cs, Mat *rval);
CvStatus TrainData_GetNames(const PtrTrainData *cs, std::vector<CVString>& names);
CvStatus TrainData_GetSubVector(const Mat& vec, const Mat& idx, Mat *rval);
CvStatus TrainData_GetSubMatrix(const Mat& matrix, const Mat& idx, int layout, Mat *rval);
CvStatus TrainData_LoadFromCSV(const CVString& filename, int headerLineCount, int responseStartIdx, int responseEndIdx, const CVString& varTypeSpec, char delimiter, char missch, PtrTrainData *rval);
CvStatus TrainData_Create(CVInputArray samples, int layout, CVInputArray responses, CVInputArray varIdx, CVInputArray sampleIdx, CVInputArray sampleWeights, CVInputArray varType, PtrTrainData *rval);

CvStatus StatModel_GetVarCount(const PtrStatModel *cs, int *rval);
CvStatus StatModel_Empty(const PtrStatModel *cs, bool *rval);
CvStatus StatModel_IsTrained(const PtrStatModel *cs, bool *rval);
CvStatus StatModel_IsClassifier(const PtrStatModel *cs, bool *rval);
CvStatus StatModel_Train(PtrStatModel *cs, const PtrTrainData& trainData, int flags, bool *rval);
CvStatus StatModel_TrainWithData(PtrStatModel *cs, CVInputArray samples, int layout, CVInputArray responses, bool *rval);
CvStatus StatModel_CalcError(const PtrStatModel *cs, const PtrTrainData& data, bool test, CVOutputArray resp, float *rval);
CvStatus StatModel_Predict(const PtrStatModel *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval);

CvStatus NormalBayesClassifier_PredictProb(const PtrNormalBayesClassifier *cs, CVInputArray inputs, CVOutputArray outputs, CVOutputArray outputProbs, int flags, float *rval);
CvStatus NormalBayesClassifier_Create(PtrNormalBayesClassifier *rval);
CvStatus NormalBayesClassifier_Load(const CVString& filepath, const CVString& nodeName, PtrNormalBayesClassifier *rval);

CvStatus KNearest_GetDefaultK(const PtrKNearest *cs, int *rval);
CvStatus KNearest_SetDefaultK(PtrKNearest *cs, int val);
CvStatus KNearest_GetIsClassifier(const PtrKNearest *cs, bool *rval);
CvStatus KNearest_SetIsClassifier(PtrKNearest *cs, bool val);
CvStatus KNearest_GetEmax(const PtrKNearest *cs, int *rval);
CvStatus KNearest_SetEmax(PtrKNearest *cs, int val);
CvStatus KNearest_GetAlgorithmType(const PtrKNearest *cs, int *rval);
CvStatus KNearest_SetAlgorithmType(PtrKNearest *cs, int val);
CvStatus KNearest_FindNearest(const PtrKNearest *cs, CVInputArray samples, int k, CVOutputArray results, CVOutputArray neighborResponses, CVOutputArray dist, float *rval);
CvStatus KNearest_Create(PtrKNearest *rval);
CvStatus KNearest_Load(const CVString& filepath, PtrKNearest *rval);

CvStatus SVM_GetType(const PtrSVM *cs, int *rval);
CvStatus SVM_SetType(PtrSVM *cs, int val);
CvStatus SVM_GetGamma(const PtrSVM *cs, double *rval);
CvStatus SVM_SetGamma(PtrSVM *cs, double val);
CvStatus SVM_GetCoef0(const PtrSVM *cs, double *rval);
CvStatus SVM_SetCoef0(PtrSVM *cs, double val);
CvStatus SVM_GetDegree(const PtrSVM *cs, double *rval);
CvStatus SVM_SetDegree(PtrSVM *cs, double val);
CvStatus SVM_GetC(const PtrSVM *cs, double *rval);
CvStatus SVM_SetC(PtrSVM *cs, double val);
CvStatus SVM_GetNu(const PtrSVM *cs, double *rval);
CvStatus SVM_SetNu(PtrSVM *cs, double val);
CvStatus SVM_GetP(const PtrSVM *cs, double *rval);
CvStatus SVM_SetP(PtrSVM *cs, double val);
CvStatus SVM_GetClassWeights(const PtrSVM *cs, Mat *rval);
CvStatus SVM_SetClassWeights(PtrSVM *cs, const Mat &val);
CvStatus SVM_GetTermCriteria(const PtrSVM *cs, TermCriteria *rval);
CvStatus SVM_SetTermCriteria(PtrSVM *cs, const TermCriteria &val);
CvStatus SVM_GetKernelType(const PtrSVM *cs, int *rval);
CvStatus SVM_SetKernel(PtrSVM *cs, int kernelType);
CvStatus SVM_SetCustomKernel(PtrSVM *cs, const PtrSVMKernel &_kernel);
CvStatus SVM_TrainAuto(PtrSVM *cs, const PtrTrainData& data, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, bool balanced, bool *rval);
CvStatus SVM_TrainAutoWithData(PtrSVM *cs, CVInputArray samples, int layout, CVInputArray responses, int kFold, PtrParamGrid Cgrid, PtrParamGrid gammaGrid, PtrParamGrid pGrid, PtrParamGrid nuGrid, PtrParamGrid coeffGrid, PtrParamGrid degreeGrid, bool balanced, bool *rval);
CvStatus SVM_GetSupportVectors(const PtrSVM *cs, Mat *rval);
CvStatus SVM_GetUncompressedSupportVectors(const PtrSVM *cs, Mat *rval);
CvStatus SVM_GetDecisionFunction(const PtrSVM *cs, int i, CVOutputArray alpha, CVOutputArray svidx, double *rval);
CvStatus SVM_GetDefaultGrid(int param_id, ParamGrid *rval);
CvStatus SVM_GetDefaultGridPtr(int param_id, PtrParamGrid *rval);
CvStatus SVM_Create(PtrSVM *rval);
CvStatus SVM_Load(const CVString& filepath, PtrSVM *rval);

CvStatus EM_GetClustersNumber(const PtrEM *cs, int *rval);
CvStatus EM_SetClustersNumber(PtrEM *cs, int val);
CvStatus EM_GetCovarianceMatrixType(const PtrEM *cs, int *rval);
CvStatus EM_SetCovarianceMatrixType(PtrEM *cs, int val);
CvStatus EM_GetTermCriteria(const PtrEM *cs, TermCriteria *rval);
CvStatus EM_SetTermCriteria(PtrEM *cs, const TermCriteria &val);
CvStatus EM_GetWeights(const PtrEM *cs, Mat *rval);
CvStatus EM_GetMeans(const PtrEM *cs, Mat *rval);
CvStatus EM_GetCovs(const PtrEM *cs, std::vector<Mat>& covs);
CvStatus EM_Predict(const PtrEM *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval);
CvStatus EM_Predict2(const PtrEM *cs, CVInputArray sample, CVOutputArray probs, Vec2d *rval);
CvStatus EM_TrainEM(PtrEM *cs, CVInputArray samples, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval);
CvStatus EM_TrainE(PtrEM *cs, CVInputArray samples, CVInputArray means0, CVInputArray covs0, CVInputArray weights0, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval);
CvStatus EM_TrainM(PtrEM *cs, CVInputArray samples, CVInputArray probs0, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval);
CvStatus EM_Create(PtrEM *rval);
CvStatus EM_Load(const CVString& filepath, const CVString& nodeName, PtrEM *rval);

CvStatus DTrees_GetMaxCategories(const PtrDTrees *cs, int *rval);
CvStatus DTrees_SetMaxCategories(PtrDTrees *cs, int val);
CvStatus DTrees_GetMaxDepth(const PtrDTrees *cs, int *rval);
CvStatus DTrees_SetMaxDepth(PtrDTrees *cs, int val);
CvStatus DTrees_GetMinSampleCount(const PtrDTrees *cs, int *rval);
CvStatus DTrees_SetMinSampleCount(PtrDTrees *cs, int val);
CvStatus DTrees_GetCVFolds(const PtrDTrees *cs, int *rval);
CvStatus DTrees_SetCVFolds(PtrDTrees *cs, int val);
CvStatus DTrees_GetUseSurrogates(const PtrDTrees *cs, bool *rval);
CvStatus DTrees_SetUseSurrogates(PtrDTrees *cs, bool val);
CvStatus DTrees_GetUse1SERule(const PtrDTrees *cs, bool *rval);
CvStatus DTrees_SetUse1SERule(PtrDTrees *cs, bool val);
CvStatus DTrees_GetTruncatePrunedTree(const PtrDTrees *cs, bool *rval);
CvStatus DTrees_SetTruncatePrunedTree(PtrDTrees *cs, bool val);
CvStatus DTrees_GetRegressionAccuracy(const PtrDTrees *cs, float *rval);
CvStatus DTrees_SetRegressionAccuracy(PtrDTrees *cs, float val);
CvStatus DTrees_GetPriors(const PtrDTrees *cs, Mat *rval);
CvStatus DTrees_SetPriors(PtrDTrees *cs, const Mat &val);
CvStatus DTrees_GetRoots(const PtrDTrees *cs, const std::vector<int>** rval);
CvStatus DTrees_GetNodes(const PtrDTrees *cs, const std::vector<cv::ml::DTrees::Node>** rval);
CvStatus DTrees_GetSplits(const PtrDTrees *cs, const std::vector<cv::ml::DTrees::Split>** rval);
CvStatus DTrees_GetSubsets(const PtrDTrees *cs, const std::vector<int>** rval);
CvStatus DTrees_Create(PtrDTrees *rval);
CvStatus DTrees_Load(const CVString& filepath, const CVString& nodeName, PtrDTrees *rval);

CvStatus RTrees_GetCalculateVarImportance(const PtrRTrees *cs, bool *rval);
CvStatus RTrees_SetCalculateVarImportance(PtrRTrees *cs, bool val);
CvStatus RTrees_GetActiveVarCount(const PtrRTrees *cs, int *rval);
CvStatus RTrees_SetActiveVarCount(PtrRTrees *cs, int val);
CvStatus RTrees_GetTermCriteria(const PtrRTrees *cs, TermCriteria *rval);
CvStatus RTrees_SetTermCriteria(PtrRTrees *cs, const TermCriteria &val);
CvStatus RTrees_GetVarImportance(const PtrRTrees *cs, Mat *rval);
CvStatus RTrees_GetVotes(const PtrRTrees *cs, CVInputArray samples, CVOutputArray results, int flags);
CvStatus RTrees_GetOOBError(const PtrRTrees *cs, double *rval);
CvStatus RTrees_Create(PtrRTrees *rval);
CvStatus RTrees_Load(const CVString& filepath, const CVString& nodeName, PtrRTrees *rval);

CvStatus Boost_GetBoostType(const PtrBoost *cs, int *rval);
CvStatus Boost_SetBoostType(PtrBoost *cs, int val);
CvStatus Boost_GetWeakCount(const PtrBoost *cs, int *rval);
CvStatus Boost_SetWeakCount(PtrBoost *cs, int val);
CvStatus Boost_GetWeightTrimRate(const PtrBoost *cs, double *rval);
CvStatus Boost_SetWeightTrimRate(PtrBoost *cs, double val);
CvStatus Boost_Create(PtrBoost *rval);
CvStatus Boost_Load(const CVString& filepath, const CVString& nodeName, PtrBoost *rval);

CvStatus ANN_MLP_GetTrainMethod(const PtrANN_MLP *cs, int *rval);
CvStatus ANN_MLP_SetTrainMethod(PtrANN_MLP *cs, int method, double param1, double param2);
CvStatus ANN_MLP_SetActivationFunction(PtrANN_MLP *cs, int type, double param1, double param2);
CvStatus ANN_MLP_SetLayerSizes(PtrANN_MLP *cs, CVInputArray _layer_sizes);
CvStatus ANN_MLP_GetLayerSizes(const PtrANN_MLP *cs, cv::Mat *rval);
CvStatus ANN_MLP_GetTermCriteria(const PtrANN_MLP *cs, TermCriteria *rval);
CvStatus ANN_MLP_SetTermCriteria(PtrANN_MLP *cs, TermCriteria val);
CvStatus ANN_MLP_GetBackpropWeightScale(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetBackpropWeightScale(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetBackpropMomentumScale(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetBackpropMomentumScale(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDW0(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDW0(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWPlus(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWPlus(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMinus(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMinus(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMin(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMin(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMax(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMax(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealInitialT(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealInitialT(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealFinalT(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealFinalT(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealCoolingRatio(const PtrANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealCoolingRatio(PtrANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealItePerStep(const PtrANN_MLP *cs, int *rval);
CvStatus ANN_MLP_SetAnnealItePerStep(PtrANN_MLP *cs, int val);
CvStatus ANN_MLP_GetWeights(const PtrANN_MLP *cs, int layerIdx, Mat *rval);
CvStatus ANN_MLP_Create(PtrANN_MLP *rval);
CvStatus ANN_MLP_Load(const CVString& filepath, PtrANN_MLP *rval);

CvStatus LogisticRegression_GetLearningRate(const PtrLogisticRegression *cs, double *rval);
CvStatus LogisticRegression_SetLearningRate(PtrLogisticRegression *cs, double val);
CvStatus LogisticRegression_GetIterations(const PtrLogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetIterations(PtrLogisticRegression *cs, int val);
CvStatus LogisticRegression_GetRegularization(const PtrLogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetRegularization(PtrLogisticRegression *cs, int val);
CvStatus LogisticRegression_GetTrainMethod(const PtrLogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetTrainMethod(PtrLogisticRegression *cs, int val);
CvStatus LogisticRegression_GetMiniBatchSize(const PtrLogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetMiniBatchSize(PtrLogisticRegression *cs, int val);
CvStatus LogisticRegression_GetTermCriteria(const PtrLogisticRegression *cs, TermCriteria *rval);
CvStatus LogisticRegression_SetTermCriteria(PtrLogisticRegression *cs, TermCriteria val);
CvStatus LogisticRegression_Predict(const PtrLogisticRegression *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval);
CvStatus LogisticRegression_GetLearntThetas(const PtrLogisticRegression *cs, Mat *rval);
CvStatus LogisticRegression_Create(PtrLogisticRegression *rval);
CvStatus LogisticRegression_Load(const CVString& filepath, const CVString& nodeName, PtrLogisticRegression *rval);

CvStatus SVMSGD_GetWeights(PtrSVMSGD *cs, Mat *rval);
CvStatus SVMSGD_GetShift(PtrSVMSGD *cs, float *rval);
CvStatus SVMSGD_Create(PtrSVMSGD *rval);
CvStatus SVMSGD_Load(const CVString& filepath, const CVString& nodeName, PtrSVMSGD *rval);
CvStatus SVMSGD_SetOptimalParameters(PtrSVMSGD *cs, int svmsgdType, int marginType);
CvStatus SVMSGD_GetSvmsgdType(const PtrSVMSGD *cs, int *rval);
CvStatus SVMSGD_SetSvmsgdType(PtrSVMSGD *cs, int svmsgdType);
CvStatus SVMSGD_GetMarginType(const PtrSVMSGD *cs, int *rval);
CvStatus SVMSGD_SetMarginType(PtrSVMSGD *cs, int marginType);
CvStatus SVMSGD_GetMarginRegularization(const PtrSVMSGD *cs, float *rval);
CvStatus SVMSGD_SetMarginRegularization(PtrSVMSGD *cs, float marginRegularization);
CvStatus SVMSGD_GetInitialStepSize(const PtrSVMSGD *cs, float *rval);
CvStatus SVMSGD_SetInitialStepSize(PtrSVMSGD *cs, float InitialStepSize);
CvStatus SVMSGD_GetStepDecreasingPower(const PtrSVMSGD *cs, float *rval);
CvStatus SVMSGD_SetStepDecreasingPower(PtrSVMSGD *cs, float stepDecreasingPower);
CvStatus SVMSGD_GetTermCriteria(const PtrSVMSGD *cs, TermCriteria *rval);
CvStatus SVMSGD_SetTermCriteria(PtrSVMSGD *cs, const TermCriteria &val);

CvStatus RandMVNormal(CVInputArray mean, CVInputArray cov, int nsamples, CVOutputArray samples);
CvStatus CreateConcentricSpheresTestSet(int nsamples, int nfeatures, int nclasses, CVOutputArray samples, CVOutputArray responses);

#ifdef __cplusplus
} // extern "C"

// Template functions cannot be in the extern "C" block
template<class SimulatedAnnealingSolverSystem>
CvStatus SimulatedAnnealingSolver(SimulatedAnnealingSolverSystem& solverSystem, double initialTemperature, double finalTemperature, double coolingRatio, size_t iterationsPerStep, double* lastTemperature, cv::RNG& rngEnergy, int *rval);

#endif

#endif //_OPENCV3_ML_H_
