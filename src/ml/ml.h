

#ifndef _OPENCV3_ML_H_
#define _OPENCV3_ML_H_
#include "core/core.h"
#include <stdbool.h>
#ifdef __cplusplus
#include <opencv2/ml.hpp>
extern "C" {
#endif

#ifdef __cplusplus

#include <float.h>
#include <map>
#include <iostream>

namespace cv
{

namespace ml
{



CvStatus ParamGrid_New(ParamGrid *rval);
CvStatus ParamGrid_NewWithParams(double _minVal, double _maxVal, double _logStep, ParamGrid *rval);
CvStatus ParamGrid_Create(double minVal, double maxVal, double logstep, Ptr<ParamGrid> *rval);

CvStatus TrainData_MissingValue(float *rval);
CvStatus TrainData_Close(TrainData *cs);
CvStatus TrainData_GetLayout(const TrainData *cs, int *rval);
CvStatus TrainData_GetNTrainSamples(const TrainData *cs, int *rval);
CvStatus TrainData_GetNTestSamples(const TrainData *cs, int *rval);
CvStatus TrainData_GetNSamples(const TrainData *cs, int *rval);
CvStatus TrainData_GetNVars(const TrainData *cs, int *rval);
CvStatus TrainData_GetNAllVars(const TrainData *cs, int *rval);
CvStatus TrainData_GetSample(const TrainData *cs, InputArray varIdx, int sidx, float* buf);
CvStatus TrainData_GetSamples(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetMissing(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainSamples(const TrainData *cs, int layout, bool compressSamples, bool compressVars, Mat *rval);
CvStatus TrainData_GetTrainResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainNormCatResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTestResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTestNormCatResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetNormCatResponses(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetSampleWeights(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTrainSampleWeights(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTestSampleWeights(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetVarIdx(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetVarType(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetVarSymbolFlags(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetResponseType(const TrainData *cs, int *rval);
CvStatus TrainData_GetTrainSampleIdx(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetTestSampleIdx(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetValues(const TrainData *cs, int vi, InputArray sidx, float* values);
CvStatus TrainData_GetNormCatValues(const TrainData *cs, int vi, InputArray sidx, int* values);
CvStatus TrainData_GetDefaultSubstValues(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetCatCount(const TrainData *cs, int vi, int *rval);
CvStatus TrainData_GetClassLabels(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetCatOfs(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetCatMap(const TrainData *cs, Mat *rval);
CvStatus TrainData_SetTrainTestSplit(TrainData *cs, int count, bool shuffle);
CvStatus TrainData_SetTrainTestSplitRatio(TrainData *cs, double ratio, bool shuffle);
CvStatus TrainData_ShuffleTrainTest(TrainData *cs);
CvStatus TrainData_GetTestSamples(const TrainData *cs, Mat *rval);
CvStatus TrainData_GetNames(const TrainData *cs, std::vector<String>& names);
CvStatus TrainData_GetSubVector(const Mat& vec, const Mat& idx, Mat *rval);
CvStatus TrainData_GetSubMatrix(const Mat& matrix, const Mat& idx, int layout, Mat *rval);
CvStatus TrainData_LoadFromCSV(const String& filename, int headerLineCount, int responseStartIdx, int responseEndIdx, const String& varTypeSpec, char delimiter, char missch, Ptr<TrainData> *rval);
CvStatus TrainData_Create(InputArray samples, int layout, InputArray responses, InputArray varIdx, InputArray sampleIdx, InputArray sampleWeights, InputArray varType, Ptr<TrainData> *rval);

CvStatus StatModel_GetVarCount(const StatModel *cs, int *rval);
CvStatus StatModel_Empty(const StatModel *cs, bool *rval);
CvStatus StatModel_IsTrained(const StatModel *cs, bool *rval);
CvStatus StatModel_IsClassifier(const StatModel *cs, bool *rval);
CvStatus StatModel_Train(StatModel *cs, const Ptr<TrainData>& trainData, int flags, bool *rval);
CvStatus StatModel_TrainWithData(StatModel *cs, InputArray samples, int layout, InputArray responses, bool *rval);
CvStatus StatModel_CalcError(const StatModel *cs, const Ptr<TrainData>& data, bool test, OutputArray resp, float *rval);
CvStatus StatModel_Predict(const StatModel *cs, InputArray samples, OutputArray results, int flags, float *rval);

CvStatus NormalBayesClassifier_PredictProb(const NormalBayesClassifier *cs, InputArray inputs, OutputArray outputs, OutputArray outputProbs, int flags, float *rval);
CvStatus NormalBayesClassifier_Create(Ptr<NormalBayesClassifier> *rval);
CvStatus NormalBayesClassifier_Load(const String& filepath, const String& nodeName, Ptr<NormalBayesClassifier> *rval);

CvStatus KNearest_GetDefaultK(const KNearest *cs, int *rval);
CvStatus KNearest_SetDefaultK(KNearest *cs, int val);
CvStatus KNearest_GetIsClassifier(const KNearest *cs, bool *rval);
CvStatus KNearest_SetIsClassifier(KNearest *cs, bool val);
CvStatus KNearest_GetEmax(const KNearest *cs, int *rval);
CvStatus KNearest_SetEmax(KNearest *cs, int val);
CvStatus KNearest_GetAlgorithmType(const KNearest *cs, int *rval);
CvStatus KNearest_SetAlgorithmType(KNearest *cs, int val);
CvStatus KNearest_FindNearest(const KNearest *cs, InputArray samples, int k, OutputArray results, OutputArray neighborResponses, OutputArray dist, float *rval);
CvStatus KNearest_Create(Ptr<KNearest> *rval);
CvStatus KNearest_Load(const String& filepath, Ptr<KNearest> *rval);

CvStatus SVM_GetType(const SVM *cs, int *rval);
CvStatus SVM_SetType(SVM *cs, int val);
CvStatus SVM_GetGamma(const SVM *cs, double *rval);
CvStatus SVM_SetGamma(SVM *cs, double val);
CvStatus SVM_GetCoef0(const SVM *cs, double *rval);
CvStatus SVM_SetCoef0(SVM *cs, double val);
CvStatus SVM_GetDegree(const SVM *cs, double *rval);
CvStatus SVM_SetDegree(SVM *cs, double val);
CvStatus SVM_GetC(const SVM *cs, double *rval);
CvStatus SVM_SetC(SVM *cs, double val);
CvStatus SVM_GetNu(const SVM *cs, double *rval);
CvStatus SVM_SetNu(SVM *cs, double val);
CvStatus SVM_GetP(const SVM *cs, double *rval);
CvStatus SVM_SetP(SVM *cs, double val);
CvStatus SVM_GetClassWeights(const SVM *cs, Mat *rval);
CvStatus SVM_SetClassWeights(SVM *cs, const Mat &val);
CvStatus SVM_GetTermCriteria(const SVM *cs, TermCriteria *rval);
CvStatus SVM_SetTermCriteria(SVM *cs, const TermCriteria &val);
CvStatus SVM_GetKernelType(const SVM *cs, int *rval);
CvStatus SVM_SetKernel(SVM *cs, int kernelType);
CvStatus SVM_SetCustomKernel(SVM *cs, const Ptr<SVM::Kernel> &_kernel);
CvStatus SVM_TrainAuto(SVM *cs, const Ptr<TrainData>& data, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, bool balanced, bool *rval);
CvStatus SVM_TrainAutoWithData(SVM *cs, InputArray samples, int layout, InputArray responses, int kFold, Ptr<ParamGrid> Cgrid, Ptr<ParamGrid> gammaGrid, Ptr<ParamGrid> pGrid, Ptr<ParamGrid> nuGrid, Ptr<ParamGrid> coeffGrid, Ptr<ParamGrid> degreeGrid, bool balanced, bool *rval);
CvStatus SVM_GetSupportVectors(const SVM *cs, Mat *rval);
CvStatus SVM_GetUncompressedSupportVectors(const SVM *cs, Mat *rval);
CvStatus SVM_GetDecisionFunction(const SVM *cs, int i, OutputArray alpha, OutputArray svidx, double *rval);
CvStatus SVM_GetDefaultGrid(int param_id, ParamGrid *rval);
CvStatus SVM_GetDefaultGridPtr(int param_id, Ptr<ParamGrid> *rval);
CvStatus SVM_Create(Ptr<SVM> *rval);
CvStatus SVM_Load(const String& filepath, Ptr<SVM> *rval);

CvStatus EM_GetClustersNumber(const EM *cs, int *rval);
CvStatus EM_SetClustersNumber(EM *cs, int val);
CvStatus EM_GetCovarianceMatrixType(const EM *cs, int *rval);
CvStatus EM_SetCovarianceMatrixType(EM *cs, int val);
CvStatus EM_GetTermCriteria(const EM *cs, TermCriteria *rval);
CvStatus EM_SetTermCriteria(EM *cs, const TermCriteria &val);
CvStatus EM_GetWeights(const EM *cs, Mat *rval);
CvStatus EM_GetMeans(const EM *cs, Mat *rval);
CvStatus EM_GetCovs(const EM *cs, std::vector<Mat>& covs);
CvStatus EM_Predict(const EM *cs, InputArray samples, OutputArray results, int flags, float *rval);
CvStatus EM_Predict2(const EM *cs, InputArray sample, OutputArray probs, Vec2d *rval);
CvStatus EM_TrainEM(EM *cs, InputArray samples, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval);
CvStatus EM_TrainE(EM *cs, InputArray samples, InputArray means0, InputArray covs0, InputArray weights0, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval);
CvStatus EM_TrainM(EM *cs, InputArray samples, InputArray probs0, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval);
CvStatus EM_Create(Ptr<EM> *rval);
CvStatus EM_Load(const String& filepath, const String& nodeName, Ptr<EM> *rval);

CvStatus DTrees_GetMaxCategories(const DTrees *cs, int *rval);
CvStatus DTrees_SetMaxCategories(DTrees *cs, int val);
CvStatus DTrees_GetMaxDepth(const DTrees *cs, int *rval);
CvStatus DTrees_SetMaxDepth(DTrees *cs, int val);
CvStatus DTrees_GetMinSampleCount(const DTrees *cs, int *rval);
CvStatus DTrees_SetMinSampleCount(DTrees *cs, int val);
CvStatus DTrees_GetCVFolds(const DTrees *cs, int *rval);
CvStatus DTrees_SetCVFolds(DTrees *cs, int val);
CvStatus DTrees_GetUseSurrogates(const DTrees *cs, bool *rval);
CvStatus DTrees_SetUseSurrogates(DTrees *cs, bool val);
CvStatus DTrees_GetUse1SERule(const DTrees *cs, bool *rval);
CvStatus DTrees_SetUse1SERule(DTrees *cs, bool val);
CvStatus DTrees_GetTruncatePrunedTree(const DTrees *cs, bool *rval);
CvStatus DTrees_SetTruncatePrunedTree(DTrees *cs, bool val);
CvStatus DTrees_GetRegressionAccuracy(const DTrees *cs, float *rval);
CvStatus DTrees_SetRegressionAccuracy(DTrees *cs, float val);
CvStatus DTrees_GetPriors(const DTrees *cs, Mat *rval);
CvStatus DTrees_SetPriors(DTrees *cs, const Mat &val);
CvStatus DTrees_GetRoots(const DTrees *cs, const std::vector<int>** rval);
CvStatus DTrees_GetNodes(const DTrees *cs, const std::vector<DTrees::Node>** rval);
CvStatus DTrees_GetSplits(const DTrees *cs, const std::vector<DTrees::Split>** rval);
CvStatus DTrees_GetSubsets(const DTrees *cs, const std::vector<int>** rval);
CvStatus DTrees_Create(Ptr<DTrees> *rval);
CvStatus DTrees_Load(const String& filepath, const String& nodeName, Ptr<DTrees> *rval);

CvStatus RTrees_GetCalculateVarImportance(const RTrees *cs, bool *rval);
CvStatus RTrees_SetCalculateVarImportance(RTrees *cs, bool val);
CvStatus RTrees_GetActiveVarCount(const RTrees *cs, int *rval);
CvStatus RTrees_SetActiveVarCount(RTrees *cs, int val);
CvStatus RTrees_GetTermCriteria(const RTrees *cs, TermCriteria *rval);
CvStatus RTrees_SetTermCriteria(RTrees *cs, const TermCriteria &val);
CvStatus RTrees_GetVarImportance(const RTrees *cs, Mat *rval);
CvStatus RTrees_GetVotes(const RTrees *cs, InputArray samples, OutputArray results, int flags);
CvStatus RTrees_GetOOBError(const RTrees *cs, double *rval);
CvStatus RTrees_Create(Ptr<RTrees> *rval);
CvStatus RTrees_Load(const String& filepath, const String& nodeName, Ptr<RTrees> *rval);

CvStatus Boost_GetBoostType(const Boost *cs, int *rval);
CvStatus Boost_SetBoostType(Boost *cs, int val);
CvStatus Boost_GetWeakCount(const Boost *cs, int *rval);
CvStatus Boost_SetWeakCount(Boost *cs, int val);
CvStatus Boost_GetWeightTrimRate(const Boost *cs, double *rval);
CvStatus Boost_SetWeightTrimRate(Boost *cs, double val);
CvStatus Boost_Create(Ptr<Boost> *rval);
CvStatus Boost_Load(const String& filepath, const String& nodeName, Ptr<Boost> *rval);

CvStatus ANN_MLP_GetTrainMethod(const ANN_MLP *cs, int *rval);
CvStatus ANN_MLP_SetTrainMethod(ANN_MLP *cs, int method, double param1, double param2);
CvStatus ANN_MLP_SetActivationFunction(ANN_MLP *cs, int type, double param1, double param2);
CvStatus ANN_MLP_SetLayerSizes(ANN_MLP *cs, InputArray _layer_sizes);
CvStatus ANN_MLP_GetLayerSizes(const ANN_MLP *cs, cv::Mat *rval);
CvStatus ANN_MLP_GetTermCriteria(const ANN_MLP *cs, TermCriteria *rval);
CvStatus ANN_MLP_SetTermCriteria(ANN_MLP *cs, TermCriteria val);
CvStatus ANN_MLP_GetBackpropWeightScale(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetBackpropWeightScale(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetBackpropMomentumScale(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetBackpropMomentumScale(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDW0(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDW0(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWPlus(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWPlus(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMinus(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMinus(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMin(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMin(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetRpropDWMax(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetRpropDWMax(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealInitialT(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealInitialT(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealFinalT(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealFinalT(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealCoolingRatio(const ANN_MLP *cs, double *rval);
CvStatus ANN_MLP_SetAnnealCoolingRatio(ANN_MLP *cs, double val);
CvStatus ANN_MLP_GetAnnealItePerStep(const ANN_MLP *cs, int *rval);
CvStatus ANN_MLP_SetAnnealItePerStep(ANN_MLP *cs, int val);
CvStatus ANN_MLP_GetWeights(const ANN_MLP *cs, int layerIdx, Mat *rval);
CvStatus ANN_MLP_Create(Ptr<ANN_MLP> *rval);
CvStatus ANN_MLP_Load(const String& filepath, Ptr<ANN_MLP> *rval);

CvStatus LogisticRegression_GetLearningRate(const LogisticRegression *cs, double *rval);
CvStatus LogisticRegression_SetLearningRate(LogisticRegression *cs, double val);
CvStatus LogisticRegression_GetIterations(const LogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetIterations(LogisticRegression *cs, int val);
CvStatus LogisticRegression_GetRegularization(const LogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetRegularization(LogisticRegression *cs, int val);
CvStatus LogisticRegression_GetTrainMethod(const LogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetTrainMethod(LogisticRegression *cs, int val);
CvStatus LogisticRegression_GetMiniBatchSize(const LogisticRegression *cs, int *rval);
CvStatus LogisticRegression_SetMiniBatchSize(LogisticRegression *cs, int val);
CvStatus LogisticRegression_GetTermCriteria(const LogisticRegression *cs, TermCriteria *rval);
CvStatus LogisticRegression_SetTermCriteria(LogisticRegression *cs, TermCriteria val);
CvStatus LogisticRegression_Predict(const LogisticRegression *cs, InputArray samples, OutputArray results, int flags, float *rval);
CvStatus LogisticRegression_GetLearntThetas(const LogisticRegression *cs, Mat *rval);
CvStatus LogisticRegression_Create(Ptr<LogisticRegression> *rval);
CvStatus LogisticRegression_Load(const String& filepath, const String& nodeName, Ptr<LogisticRegression> *rval);

CvStatus SVMSGD_GetWeights(SVMSGD *cs, Mat *rval);
CvStatus SVMSGD_GetShift(SVMSGD *cs, float *rval);
CvStatus SVMSGD_Create(Ptr<SVMSGD> *rval);
CvStatus SVMSGD_Load(const String& filepath, const String& nodeName, Ptr<SVMSGD> *rval);
CvStatus SVMSGD_SetOptimalParameters(SVMSGD *cs, int svmsgdType, int marginType);
CvStatus SVMSGD_GetSvmsgdType(const SVMSGD *cs, int *rval);
CvStatus SVMSGD_SetSvmsgdType(SVMSGD *cs, int svmsgdType);
CvStatus SVMSGD_GetMarginType(const SVMSGD *cs, int *rval);
CvStatus SVMSGD_SetMarginType(SVMSGD *cs, int marginType);
CvStatus SVMSGD_GetMarginRegularization(const SVMSGD *cs, float *rval);
CvStatus SVMSGD_SetMarginRegularization(SVMSGD *cs, float marginRegularization);
CvStatus SVMSGD_GetInitialStepSize(const SVMSGD *cs, float *rval);
CvStatus SVMSGD_SetInitialStepSize(SVMSGD *cs, float InitialStepSize);
CvStatus SVMSGD_GetStepDecreasingPower(const SVMSGD *cs, float *rval);
CvStatus SVMSGD_SetStepDecreasingPower(SVMSGD *cs, float stepDecreasingPower);
CvStatus SVMSGD_GetTermCriteria(const SVMSGD *cs, TermCriteria *rval);
CvStatus SVMSGD_SetTermCriteria(SVMSGD *cs, const cv::TermCriteria &val);

CvStatus RandMVNormal(InputArray mean, InputArray cov, int nsamples, OutputArray samples);
CvStatus CreateConcentricSpheresTestSet(int nsamples, int nfeatures, int nclasses, OutputArray samples, OutputArray responses);

template<class SimulatedAnnealingSolverSystem>
CvStatus SimulatedAnnealingSolver(SimulatedAnnealingSolverSystem& solverSystem, double initialTemperature, double finalTemperature, double coolingRatio, size_t iterationsPerStep, double* lastTemperature, cv::RNG& rngEnergy, int *rval);

//! @} ml

}
}
#ifdef __cplusplus

}

#endif

#endif //_OPENCV3_ML_H_

