#include "ml.h"
#include <memory>
#include <vector>

CvStatus ParamGrid_New(ParamGrid *rval) {
    BEGIN_WRAP
    *rval = ParamGrid();
    END_WRAP
}

CvStatus ParamGrid_NewWithParams(double _minVal, double _maxVal, double _logStep, ParamGrid *rval) {
    BEGIN_WRAP
    *rval = ParamGrid(_minVal, _maxVal, _logStep);
    END_WRAP
}

CvStatus ParamGrid_Create(double minVal, double maxVal, double logstep, PtrParamGrid *rval) {
    BEGIN_WRAP
    *rval = cv::ml::ParamGrid::create(minVal, maxVal, logstep);
    END_WRAP
}

CvStatus TrainData_MissingValue(float *rval) {
    BEGIN_WRAP
    *rval = cv::ml::TrainData::missingValue();
    END_WRAP
}

CvStatus TrainData_Close(PtrTrainData *cs) {
    delete cs;
    return CvStatus::OK;
}

CvStatus TrainData_GetLayout(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getLayout();
    END_WRAP
}

CvStatus TrainData_GetNTrainSamples(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNTrainSamples();
    END_WRAP
}

CvStatus TrainData_GetNTestSamples(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNTestSamples();
    END_WRAP
}

CvStatus TrainData_GetNSamples(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNSamples();
    END_WRAP
}

CvStatus TrainData_GetNVars(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNVars();
    END_WRAP
}

CvStatus TrainData_GetNAllVars(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNAllVars();
    END_WRAP
}

CvStatus TrainData_GetSample(const PtrTrainData *cs, CVInputArray varIdx, int sidx, float* buf) {
    BEGIN_WRAP
    cs->get()->getSample(varIdx, sidx, buf);
    END_WRAP
}

CvStatus TrainData_GetSamples(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getSamples();
    END_WRAP
}

CvStatus TrainData_GetMissing(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMissing();
    END_WRAP
}

CvStatus TrainData_GetTrainSamples(const PtrTrainData *cs, int layout, bool compressSamples, bool compressVars, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainSamples(layout, compressSamples, compressVars);
    END_WRAP
}

CvStatus TrainData_GetTrainResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainResponses();
    END_WRAP
}

CvStatus TrainData_GetTrainNormCatResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetTestResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTestResponses();
    END_WRAP
}

CvStatus TrainData_GetTestNormCatResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTestNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getResponses();
    END_WRAP
}

CvStatus TrainData_GetNormCatResponses(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetSampleWeights(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetTrainSampleWeights(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetTestSampleWeights(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTestSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetVarIdx(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getVarIdx();
    END_WRAP
}

CvStatus TrainData_GetVarType(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getVarType();
    END_WRAP
}

CvStatus TrainData_GetVarSymbolFlags(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getVarSymbolFlags();
    END_WRAP
}

CvStatus TrainData_GetResponseType(const PtrTrainData *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getResponseType();
    END_WRAP
}

CvStatus TrainData_GetTrainSampleIdx(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainSampleIdx();
    END_WRAP
}

CvStatus TrainData_GetTestSampleIdx(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTestSampleIdx();
    END_WRAP
}

CvStatus TrainData_GetValues(const PtrTrainData *cs, int vi, CVInputArray sidx, float* values) {
    BEGIN_WRAP
    cs->get()->getValues(vi, sidx, values);
    END_WRAP
}

CvStatus TrainData_GetNormCatValues(const PtrTrainData *cs, int vi, CVInputArray sidx, int* values) {
    BEGIN_WRAP
    cs->get()->getNormCatValues(vi, sidx, values);
    END_WRAP
}

CvStatus TrainData_GetDefaultSubstValues(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getDefaultSubstValues();
    END_WRAP
}

CvStatus TrainData_GetCatCount(const PtrTrainData *cs, int vi, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCatCount(vi);
    END_WRAP
}

CvStatus TrainData_GetClassLabels(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getClassLabels();
    END_WRAP
}

CvStatus TrainData_GetCatOfs(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCatOfs();
    END_WRAP
}

CvStatus TrainData_GetCatMap(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCatMap();
    END_WRAP
}

CvStatus TrainData_SetTrainTestSplit(PtrTrainData *cs, int count, bool shuffle) {
    BEGIN_WRAP
    cs->get()->setTrainTestSplit(count, shuffle);
    END_WRAP
}

CvStatus TrainData_SetTrainTestSplitRatio(PtrTrainData *cs, double ratio, bool shuffle) {
    BEGIN_WRAP
    cs->get()->setTrainTestSplitRatio(ratio, shuffle);
    END_WRAP
}

CvStatus TrainData_ShuffleTrainTest(PtrTrainData *cs) {
    BEGIN_WRAP
    cs->get()->shuffleTrainTest();
    END_WRAP
}

CvStatus TrainData_GetTestSamples(const PtrTrainData *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTestSamples();
    END_WRAP
}

CvStatus TrainData_GetNames(const PtrTrainData *cs, std::vector<CVString>& names) {
    BEGIN_WRAP
    cs->get()->getNames(names);
    END_WRAP
}

CvStatus TrainData_GetSubVector(const cv::Mat& vec, const cv::Mat& idx, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cv::ml::TrainData::getSubVector(vec, idx);
    END_WRAP
}

CvStatus TrainData_GetSubMatrix(const cv::Mat& matrix, const cv::Mat& idx, int layout, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cv::ml::TrainData::getSubMatrix(matrix, idx, layout);
    END_WRAP
}

CvStatus TrainData_LoadFromCSV(const CVString& filename, int headerLineCount, int responseStartIdx, int responseEndIdx, const CVString& varTypeSpec, char delimiter, char missch, PtrTrainData *rval) {
    BEGIN_WRAP
    *rval = cv::ml::TrainData::loadFromCSV(filename, headerLineCount, responseStartIdx, responseEndIdx, varTypeSpec, delimiter, missch);
    END_WRAP
}

CvStatus TrainData_Create(CVInputArray samples, int layout, CVInputArray responses, CVInputArray varIdx, CVInputArray sampleIdx, CVInputArray sampleWeights, CVInputArray varType, PtrTrainData *rval) {
    BEGIN_WRAP
    *rval = cv::ml::TrainData::create(samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType);
    END_WRAP
}

CvStatus StatModel_GetVarCount(const PtrStatModel *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getVarCount();
    END_WRAP
}

CvStatus StatModel_Empty(const PtrStatModel *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->empty();
    END_WRAP
}

CvStatus StatModel_IsTrained(const PtrStatModel *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->isTrained();
    END_WRAP
}

CvStatus StatModel_IsClassifier(const PtrStatModel *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->isClassifier();
    END_WRAP
}

CvStatus StatModel_Train(PtrStatModel *cs, const PtrTrainData& trainData, int flags, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->train(trainData, flags);
    END_WRAP
}

CvStatus StatModel_TrainWithData(PtrStatModel *cs, CVInputArray samples, int layout, CVInputArray responses, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->train(samples, layout, responses);
    END_WRAP
}

CvStatus StatModel_CalcError(const PtrStatModel *cs, const PtrTrainData& data, bool test, CVOutputArray resp, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->calcError(data, test, resp);
    END_WRAP
}

CvStatus StatModel_Predict(const PtrStatModel *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->predict(samples, results, flags);
    END_WRAP
}

CvStatus NormalBayesClassifier_PredictProb(const PtrNormalBayesClassifier *cs, CVInputArray inputs, CVOutputArray outputs, CVOutputArray outputProbs, int flags, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->predictProb(inputs, outputs, outputProbs, flags);
    END_WRAP
}

CvStatus NormalBayesClassifier_Create(PtrNormalBayesClassifier *rval) {
    BEGIN_WRAP
    *rval = cv::ml::NormalBayesClassifier::create();
    END_WRAP
}

CvStatus NormalBayesClassifier_Load(const CVString& filepath, const CVString& nodeName, PtrNormalBayesClassifier *rval) {
    BEGIN_WRAP
    *rval = cv::ml::NormalBayesClassifier::load(filepath, nodeName);
    END_WRAP
}

CvStatus KNearest_GetDefaultK(const PtrKNearest *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getDefaultK();
    END_WRAP
}

CvStatus KNearest_SetDefaultK(PtrKNearest *cs, int val) {
    BEGIN_WRAP
    cs->get()->setDefaultK(val);
    END_WRAP
}

CvStatus KNearest_GetIsClassifier(const PtrKNearest *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getIsClassifier();
    END_WRAP
}

CvStatus KNearest_SetIsClassifier(PtrKNearest *cs, bool val) {
    BEGIN_WRAP
    cs->get()->setIsClassifier(val);
    END_WRAP
}

CvStatus KNearest_GetEmax(const PtrKNearest *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getEmax();
    END_WRAP
}

CvStatus KNearest_SetEmax(PtrKNearest *cs, int val) {
    BEGIN_WRAP
    cs->get()->setEmax(val);
    END_WRAP
}

CvStatus KNearest_GetAlgorithmType(const PtrKNearest *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getAlgorithmType();
    END_WRAP
}

CvStatus KNearest_SetAlgorithmType(PtrKNearest *cs, int val) {
    BEGIN_WRAP
    cs->get()->setAlgorithmType(val);
    END_WRAP
}

CvStatus KNearest_FindNearest(const PtrKNearest *cs, CVInputArray samples, int k, CVOutputArray results, CVOutputArray neighborResponses, CVOutputArray dist, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->findNearest(samples, k, results, neighborResponses, dist);
    END_WRAP
}

CvStatus KNearest_Create(PtrKNearest *rval) {
    BEGIN_WRAP
    *rval = cv::ml::KNearest::create();
    END_WRAP
}

CvStatus KNearest_Load(const CVString& filepath, PtrKNearest *rval) {
    BEGIN_WRAP
    *rval = cv::ml::KNearest::load(filepath);
    END_WRAP
}

CvStatus SVM_GetType(const PtrSVM *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getType();
    END_WRAP
}

CvStatus SVM_SetType(PtrSVM *cs, int val) {
    BEGIN_WRAP
    cs->get()->setType(val);
    END_WRAP
}

CvStatus SVM_GetGamma(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getGamma();
    END_WRAP
}

CvStatus SVM_SetGamma(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setGamma(val);
    END_WRAP
}

CvStatus SVM_GetCoef0(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCoef0();
    END_WRAP
}

CvStatus SVM_SetCoef0(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setCoef0(val);
    END_WRAP
}

CvStatus SVM_GetDegree(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getDegree();
    END_WRAP
}

CvStatus SVM_SetDegree(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setDegree(val);
    END_WRAP
}

CvStatus SVM_GetC(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getC();
    END_WRAP
}

CvStatus SVM_SetC(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setC(val);
    END_WRAP
}

CvStatus SVM_GetNu(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getNu();
    END_WRAP
}

CvStatus SVM_SetNu(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setNu(val);
    END_WRAP
}

CvStatus SVM_GetP(const PtrSVM *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getP();
    END_WRAP
}

CvStatus SVM_SetP(PtrSVM *cs, double val) {
    BEGIN_WRAP
    cs->get()->setP(val);
    END_WRAP
}

CvStatus SVM_GetClassWeights(const PtrSVM *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getClassWeights();
    END_WRAP
}

CvStatus SVM_SetClassWeights(PtrSVM *cs, const cv::Mat &val) {
    BEGIN_WRAP
    cs->get()->setClassWeights(val);
    END_WRAP
}

CvStatus SVM_GetTermCriteria(const PtrSVM *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus SVM_SetTermCriteria(PtrSVM *cs, const cv::TermCriteria &val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus SVM_GetKernelType(const PtrSVM *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getKernelType();
    END_WRAP
}

CvStatus SVM_SetKernel(PtrSVM *cs, int kernelType) {
    BEGIN_WRAP
    cs->get()->setKernel(kernelType);
    END_WRAP
}

CvStatus SVM_SetCustomKernel(PtrSVM *cs, const cv::Ptr<cv::ml::SVM::Kernel> &_kernel) {
    BEGIN_WRAP
    cs->get()->setCustomKernel(_kernel);
    END_WRAP
}

CvStatus SVM_TrainAuto(PtrSVM *cs, const PtrTrainData& data, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, bool balanced, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->trainAuto(data, kFold, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);
    END_WRAP
}

CvStatus SVM_TrainAutoWithData(PtrSVM *cs, CVInputArray samples, int layout, CVInputArray responses, int kFold, PtrParamGrid Cgrid, PtrParamGrid gammaGrid, PtrParamGrid pGrid, PtrParamGrid nuGrid, PtrParamGrid coeffGrid, PtrParamGrid degreeGrid, bool balanced, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->trainAuto(samples, layout, responses, kFold, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);
    END_WRAP
}

CvStatus SVM_GetSupportVectors(const PtrSVM *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getSupportVectors();
    END_WRAP
}

CvStatus SVM_GetUncompressedSupportVectors(const PtrSVM *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getUncompressedSupportVectors();
    END_WRAP
}

CvStatus SVM_GetDecisionFunction(const PtrSVM *cs, int i, CVOutputArray alpha, CVOutputArray svidx, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getDecisionFunction(i, alpha, svidx);
    END_WRAP
}

CvStatus SVM_GetDefaultGrid(int param_id, ParamGrid *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVM::getDefaultGrid(param_id);
    END_WRAP
}

CvStatus SVM_GetDefaultGridPtr(int param_id, PtrParamGrid *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVM::getDefaultGridPtr(param_id);
    END_WRAP
}

CvStatus SVM_Create(PtrSVM *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVM::create();
    END_WRAP
}

CvStatus SVM_Load(const CVString& filepath, PtrSVM *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVM::load(filepath);
    END_WRAP
}

CvStatus EM_GetClustersNumber(const PtrEM *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getClustersNumber();
    END_WRAP
}

CvStatus EM_SetClustersNumber(PtrEM *cs, int val) {
    BEGIN_WRAP
    cs->get()->setClustersNumber(val);
    END_WRAP
}

CvStatus EM_GetCovarianceMatrixType(const PtrEM *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCovarianceMatrixType();
    END_WRAP
}

CvStatus EM_SetCovarianceMatrixType(PtrEM *cs, int val) {
    BEGIN_WRAP
    cs->get()->setCovarianceMatrixType(val);
    END_WRAP
}

CvStatus EM_GetTermCriteria(const PtrEM *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus EM_SetTermCriteria(PtrEM *cs, const cv::TermCriteria &val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus EM_GetWeights(const PtrEM *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getWeights();
    END_WRAP
}

CvStatus EM_GetMeans(const PtrEM *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMeans();
    END_WRAP
}

CvStatus EM_GetCovs(const PtrEM *cs, std::vector<cv::Mat>& covs) {
    BEGIN_WRAP
    cs->get()->getCovs(covs);
    END_WRAP
}

CvStatus EM_Predict(const PtrEM *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->predict(samples, results, flags);
    END_WRAP
}

CvStatus EM_Predict2(const PtrEM *cs, CVInputArray sample, CVOutputArray probs, cv::Vec2d *rval) {
    BEGIN_WRAP
    *rval = cs->get()->predict2(sample, probs);
    END_WRAP
}

CvStatus EM_TrainEM(PtrEM *cs, CVInputArray samples, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->trainEM(samples, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_TrainE(PtrEM *cs, CVInputArray samples, CVInputArray means0, CVInputArray covs0, CVInputArray weights0, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_TrainM(PtrEM *cs, CVInputArray samples, CVInputArray probs0, CVOutputArray logLikelihoods, CVOutputArray labels, CVOutputArray probs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->trainM(samples, probs0, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_Create(PtrEM *rval) {
    BEGIN_WRAP
    *rval = cv::ml::EM::create();
    END_WRAP
}

CvStatus EM_Load(const CVString& filepath, const CVString& nodeName, PtrEM *rval) {
    BEGIN_WRAP
    *rval = cv::ml::EM::load(filepath, nodeName);
    END_WRAP
}

CvStatus DTrees_GetMaxCategories(const PtrDTrees *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMaxCategories();
    END_WRAP
}

CvStatus DTrees_SetMaxCategories(PtrDTrees *cs, int val) {
    BEGIN_WRAP
    cs->get()->setMaxCategories(val);
    END_WRAP
}

CvStatus DTrees_GetMaxDepth(const PtrDTrees *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMaxDepth();
    END_WRAP
}

CvStatus DTrees_SetMaxDepth(PtrDTrees *cs, int val) {
    BEGIN_WRAP
    cs->get()->setMaxDepth(val);
    END_WRAP
}

CvStatus DTrees_GetMinSampleCount(const PtrDTrees *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMinSampleCount();
    END_WRAP
}

CvStatus DTrees_SetMinSampleCount(PtrDTrees *cs, int val) {
    BEGIN_WRAP
    cs->get()->setMinSampleCount(val);
    END_WRAP
}

CvStatus DTrees_GetCVFolds(const PtrDTrees *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCVFolds();
    END_WRAP
}

CvStatus DTrees_SetCVFolds(PtrDTrees *cs, int val) {
    BEGIN_WRAP
    cs->get()->setCVFolds(val);
    END_WRAP
}

CvStatus DTrees_GetUseSurrogates(const PtrDTrees *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getUseSurrogates();
    END_WRAP
}

CvStatus DTrees_SetUseSurrogates(PtrDTrees *cs, bool val) {
    BEGIN_WRAP
    cs->get()->setUseSurrogates(val);
    END_WRAP
}

CvStatus DTrees_GetUse1SERule(const PtrDTrees *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getUse1SERule();
    END_WRAP
}

CvStatus DTrees_SetUse1SERule(PtrDTrees *cs, bool val) {
    BEGIN_WRAP
    cs->get()->setUse1SERule(val);
    END_WRAP
}

CvStatus DTrees_GetTruncatePrunedTree(const PtrDTrees *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTruncatePrunedTree();
    END_WRAP
}

CvStatus DTrees_SetTruncatePrunedTree(PtrDTrees *cs, bool val) {
    BEGIN_WRAP
    cs->get()->setTruncatePrunedTree(val);
    END_WRAP
}

CvStatus DTrees_GetRegressionAccuracy(const PtrDTrees *cs, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRegressionAccuracy();
    END_WRAP
}

CvStatus DTrees_SetRegressionAccuracy(PtrDTrees *cs, float val) {
    BEGIN_WRAP
    cs->get()->setRegressionAccuracy(val);
    END_WRAP
}

CvStatus DTrees_GetPriors(const PtrDTrees *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getPriors();
    END_WRAP
}

CvStatus DTrees_SetPriors(PtrDTrees *cs, const cv::Mat &val) {
    BEGIN_WRAP
    cs->get()->setPriors(val);
    END_WRAP
}

CvStatus DTrees_GetRoots(const PtrDTrees *cs, const std::vector<int>** rval) {
    BEGIN_WRAP
    *rval = &cs->get()->getRoots();
    END_WRAP
}

CvStatus DTrees_GetNodes(const PtrDTrees *cs, const std::vector<cv::ml::DTrees::Node>** rval) {
    BEGIN_WRAP
    *rval = &cs->get()->getNodes();
    END_WRAP
}

CvStatus DTrees_GetSplits(const PtrDTrees *cs, const std::vector<cv::ml::DTrees::Split>** rval) {
    BEGIN_WRAP
    *rval = &cs->get()->getSplits();
    END_WRAP
}

CvStatus DTrees_GetSubsets(const PtrDTrees *cs, const std::vector<int>** rval) {
    BEGIN_WRAP
    *rval = &cs->get()->getSubsets();
    END_WRAP
}

CvStatus DTrees_Create(PtrDTrees *rval) {
    BEGIN_WRAP
    *rval = cv::ml::DTrees::create();
    END_WRAP
}

CvStatus DTrees_Load(const CVString& filepath, const CVString& nodeName, PtrDTrees *rval) {
    BEGIN_WRAP
    *rval = cv::ml::DTrees::load(filepath, nodeName);
    END_WRAP
}

CvStatus RTrees_GetCalculateVarImportance(const PtrRTrees *cs, bool *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getCalculateVarImportance();
    END_WRAP
}

CvStatus RTrees_SetCalculateVarImportance(PtrRTrees *cs, bool val) {
    BEGIN_WRAP
    cs->get()->setCalculateVarImportance(val);
    END_WRAP
}

CvStatus RTrees_GetActiveVarCount(const PtrRTrees *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getActiveVarCount();
    END_WRAP
}

CvStatus RTrees_SetActiveVarCount(PtrRTrees *cs, int val) {
    BEGIN_WRAP
    cs->get()->setActiveVarCount(val);
    END_WRAP
}

CvStatus RTrees_GetTermCriteria(const PtrRTrees *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus RTrees_SetTermCriteria(PtrRTrees *cs, const cv::TermCriteria &val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus RTrees_GetVarImportance(const PtrRTrees *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getVarImportance();
    END_WRAP
}

CvStatus RTrees_GetVotes(const PtrRTrees *cs, CVInputArray samples, CVOutputArray results, int flags) {
    BEGIN_WRAP
    cs->get()->getVotes(samples, results, flags);
    END_WRAP
}

CvStatus RTrees_GetOOBError(const PtrRTrees *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getOOBError();
    END_WRAP
}

CvStatus RTrees_Create(PtrRTrees *rval) {
    BEGIN_WRAP
    *rval = cv::ml::RTrees::create();
    END_WRAP
}

CvStatus RTrees_Load(const CVString& filepath, const CVString& nodeName, PtrRTrees *rval) {
    BEGIN_WRAP
    *rval = cv::ml::RTrees::load(filepath, nodeName);
    END_WRAP
}

CvStatus Boost_GetBoostType(const PtrBoost *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getBoostType();
    END_WRAP
}

CvStatus Boost_SetBoostType(PtrBoost *cs, int val) {
    BEGIN_WRAP
    cs->get()->setBoostType(val);
    END_WRAP
}

CvStatus Boost_GetWeakCount(const PtrBoost *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getWeakCount();
    END_WRAP
}

CvStatus Boost_SetWeakCount(PtrBoost *cs, int val) {
    BEGIN_WRAP
    cs->get()->setWeakCount(val);
    END_WRAP
}

CvStatus Boost_GetWeightTrimRate(const PtrBoost *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getWeightTrimRate();
    END_WRAP
}

CvStatus Boost_SetWeightTrimRate(PtrBoost *cs, double val) {
    BEGIN_WRAP
    cs->get()->setWeightTrimRate(val);
    END_WRAP
}

CvStatus Boost_Create(PtrBoost *rval) {
    BEGIN_WRAP
    *rval = cv::ml::Boost::create();
    END_WRAP
}

CvStatus Boost_Load(const CVString& filepath, const CVString& nodeName, PtrBoost *rval) {
    BEGIN_WRAP
    *rval = cv::ml::Boost::load(filepath, nodeName);
    END_WRAP
}

CvStatus ANN_MLP_GetTrainMethod(const PtrANN_MLP *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainMethod();
    END_WRAP
}

CvStatus ANN_MLP_SetTrainMethod(PtrANN_MLP *cs, int method, double param1, double param2) {
    BEGIN_WRAP
    cs->get()->setTrainMethod(method, param1, param2);
    END_WRAP
}

CvStatus ANN_MLP_SetActivationFunction(PtrANN_MLP *cs, int type, double param1, double param2) {
    BEGIN_WRAP
    cs->get()->setActivationFunction(type, param1, param2);
    END_WRAP
}

CvStatus ANN_MLP_SetLayerSizes(PtrANN_MLP *cs, CVInputArray _layer_sizes) {
    BEGIN_WRAP
    cs->get()->setLayerSizes(_layer_sizes);
    END_WRAP
}

CvStatus ANN_MLP_GetLayerSizes(const PtrANN_MLP *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getLayerSizes();
    END_WRAP
}

CvStatus ANN_MLP_GetTermCriteria(const PtrANN_MLP *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus ANN_MLP_SetTermCriteria(PtrANN_MLP *cs, cv::TermCriteria val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus ANN_MLP_GetBackpropWeightScale(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getBackpropWeightScale();
    END_WRAP
}

CvStatus ANN_MLP_SetBackpropWeightScale(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setBackpropWeightScale(val);
    END_WRAP
}

CvStatus ANN_MLP_GetBackpropMomentumScale(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getBackpropMomentumScale();
    END_WRAP
}

CvStatus ANN_MLP_SetBackpropMomentumScale(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setBackpropMomentumScale(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDW0(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRpropDW0();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDW0(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setRpropDW0(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWPlus(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRpropDWPlus();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWPlus(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setRpropDWPlus(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMinus(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRpropDWMinus();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMinus(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setRpropDWMinus(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMin(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRpropDWMin();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMin(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setRpropDWMin(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMax(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRpropDWMax();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMax(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setRpropDWMax(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealInitialT(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getAnnealInitialT();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealInitialT(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setAnnealInitialT(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealFinalT(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getAnnealFinalT();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealFinalT(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setAnnealFinalT(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealCoolingRatio(const PtrANN_MLP *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getAnnealCoolingRatio();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealCoolingRatio(PtrANN_MLP *cs, double val) {
    BEGIN_WRAP
    cs->get()->setAnnealCoolingRatio(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealItePerStep(const PtrANN_MLP *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getAnnealItePerStep();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealItePerStep(PtrANN_MLP *cs, int val) {
    BEGIN_WRAP
    cs->get()->setAnnealItePerStep(val);
    END_WRAP
}

CvStatus ANN_MLP_GetWeights(const PtrANN_MLP *cs, int layerIdx, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getWeights(layerIdx);
    END_WRAP
}

CvStatus ANN_MLP_Create(PtrANN_MLP *rval) {
    BEGIN_WRAP
    *rval = cv::ml::ANN_MLP::create();
    END_WRAP
}

CvStatus ANN_MLP_Load(const CVString& filepath, PtrANN_MLP *rval) {
    BEGIN_WRAP
    *rval = cv::ml::ANN_MLP::load(filepath);
    END_WRAP
}

CvStatus LogisticRegression_GetLearningRate(const PtrLogisticRegression *cs, double *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getLearningRate();
    END_WRAP
}

CvStatus LogisticRegression_SetLearningRate(PtrLogisticRegression *cs, double val) {
    BEGIN_WRAP
    cs->get()->setLearningRate(val);
    END_WRAP
}

CvStatus LogisticRegression_GetIterations(const PtrLogisticRegression *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getIterations();
    END_WRAP
}

CvStatus LogisticRegression_SetIterations(PtrLogisticRegression *cs, int val) {
    BEGIN_WRAP
    cs->get()->setIterations(val);
    END_WRAP
}

CvStatus LogisticRegression_GetRegularization(const PtrLogisticRegression *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getRegularization();
    END_WRAP
}

CvStatus LogisticRegression_SetRegularization(PtrLogisticRegression *cs, int val) {
    BEGIN_WRAP
    cs->get()->setRegularization(val);
    END_WRAP
}

CvStatus LogisticRegression_GetTrainMethod(const PtrLogisticRegression *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTrainMethod();
    END_WRAP
}

CvStatus LogisticRegression_SetTrainMethod(PtrLogisticRegression *cs, int val) {
    BEGIN_WRAP
    cs->get()->setTrainMethod(val);
    END_WRAP
}

CvStatus LogisticRegression_GetMiniBatchSize(const PtrLogisticRegression *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMiniBatchSize();
    END_WRAP
}

CvStatus LogisticRegression_SetMiniBatchSize(PtrLogisticRegression *cs, int val) {
    BEGIN_WRAP
    cs->get()->setMiniBatchSize(val);
    END_WRAP
}

CvStatus LogisticRegression_GetTermCriteria(const PtrLogisticRegression *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus LogisticRegression_SetTermCriteria(PtrLogisticRegression *cs, cv::TermCriteria val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus LogisticRegression_Predict(const PtrLogisticRegression *cs, CVInputArray samples, CVOutputArray results, int flags, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->predict(samples, results, flags);
    END_WRAP
}

CvStatus LogisticRegression_GetLearntThetas(const PtrLogisticRegression *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->get_learnt_thetas();
    END_WRAP
}

CvStatus LogisticRegression_Create(PtrLogisticRegression *rval) {
    BEGIN_WRAP
    *rval = cv::ml::LogisticRegression::create();
    END_WRAP
}

CvStatus LogisticRegression_Load(const CVString& filepath, const CVString& nodeName, PtrLogisticRegression *rval) {
    BEGIN_WRAP
    *rval = cv::ml::LogisticRegression::load(filepath, nodeName);
    END_WRAP
}

CvStatus SVMSGD_GetWeights(PtrSVMSGD *cs, cv::Mat *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getWeights();
    END_WRAP
}

CvStatus SVMSGD_GetShift(PtrSVMSGD *cs, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getShift();
    END_WRAP
}

CvStatus SVMSGD_Create(PtrSVMSGD *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVMSGD::create();
    END_WRAP
}

CvStatus SVMSGD_Load(const CVString& filepath, const CVString& nodeName, PtrSVMSGD *rval) {
    BEGIN_WRAP
    *rval = cv::ml::SVMSGD::load(filepath, nodeName);
    END_WRAP
}

CvStatus SVMSGD_SetOptimalParameters(PtrSVMSGD *cs, int svmsgdType, int marginType) {
    BEGIN_WRAP
    cs->get()->setOptimalParameters(svmsgdType, marginType);
    END_WRAP
}

CvStatus SVMSGD_GetSvmsgdType(const PtrSVMSGD *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getSvmsgdType();
    END_WRAP
}

CvStatus SVMSGD_SetSvmsgdType(PtrSVMSGD *cs, int svmsgdType) {
    BEGIN_WRAP
    cs->get()->setSvmsgdType(svmsgdType);
    END_WRAP
}

CvStatus SVMSGD_GetMarginType(const PtrSVMSGD *cs, int *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMarginType();
    END_WRAP
}

CvStatus SVMSGD_SetMarginType(PtrSVMSGD *cs, int marginType) {
    BEGIN_WRAP
    cs->get()->setMarginType(marginType);
    END_WRAP
}

CvStatus SVMSGD_GetMarginRegularization(const PtrSVMSGD *cs, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getMarginRegularization();
    END_WRAP
}

CvStatus SVMSGD_SetMarginRegularization(PtrSVMSGD *cs, float marginRegularization) {
    BEGIN_WRAP
    cs->get()->setMarginRegularization(marginRegularization);
    END_WRAP
}

CvStatus SVMSGD_GetInitialStepSize(const PtrSVMSGD *cs, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getInitialStepSize();
    END_WRAP
}

CvStatus SVMSGD_SetInitialStepSize(PtrSVMSGD *cs, float InitialStepSize) {
    BEGIN_WRAP
    cs->get()->setInitialStepSize(InitialStepSize);
    END_WRAP
}

CvStatus SVMSGD_GetStepDecreasingPower(const PtrSVMSGD *cs, float *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getStepDecreasingPower();
    END_WRAP
}

CvStatus SVMSGD_SetStepDecreasingPower(PtrSVMSGD *cs, float stepDecreasingPower) {
    BEGIN_WRAP
    cs->get()->setStepDecreasingPower(stepDecreasingPower);
    END_WRAP
}

CvStatus SVMSGD_GetTermCriteria(const PtrSVMSGD *cs, cv::TermCriteria *rval) {
    BEGIN_WRAP
    *rval = cs->get()->getTermCriteria();
    END_WRAP
}

CvStatus SVMSGD_SetTermCriteria(PtrSVMSGD *cs, const cv::TermCriteria &val) {
    BEGIN_WRAP
    cs->get()->setTermCriteria(val);
    END_WRAP
}

CvStatus RandMVNormal(CVInputArray mean, CVInputArray cov, int nsamples, CVOutputArray samples) {
    BEGIN_WRAP
    cv::randMVNormal(mean, cov, nsamples, samples);
    END_WRAP
}

CvStatus CreateConcentricSpheresTestSet(int nsamples, int nfeatures, int nclasses, CVOutputArray samples, CVOutputArray responses) {
    BEGIN_WRAP
    cv::createConcentricSpheresTestSet(nsamples, nfeatures, nclasses, samples, responses);
    END_WRAP
}

template<class SimulatedAnnealingSolverSystem>
CvStatus SimulatedAnnealingSolver(SimulatedAnnealingSolverSystem& solverSystem, double initialTemperature, double finalTemperature, double coolingRatio, size_t iterationsPerStep, double* lastTemperature, cv::RNG& rngEnergy, int *rval) {
    BEGIN_WRAP
    *rval = cv::simulatedAnnealingSolver(solverSystem, initialTemperature, finalTemperature, coolingRatio, iterationsPerStep, lastTemperature, rngEnergy);
    END_WRAP
}
