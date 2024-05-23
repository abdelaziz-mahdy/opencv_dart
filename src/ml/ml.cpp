
#include "ml.h"
#include <memory>
#include <vector>



CvStatus ParamGrid_New(ParamGrid *rval)
{
    BEGIN_WRAP
    *rval = ParamGrid();
    END_WRAP
}

CvStatus ParamGrid_NewWithParams(double _minVal, double _maxVal, double _logStep, ParamGrid *rval)
{
    BEGIN_WRAP
    *rval = ParamGrid(_minVal, _maxVal, _logStep);
    END_WRAP
}

CvStatus ParamGrid_Create(double minVal, double maxVal, double logstep, Ptr<ParamGrid> *rval)
{
    BEGIN_WRAP
    *rval = ParamGrid::create(minVal, maxVal, logstep);
    END_WRAP
}

CvStatus TrainData_MissingValue(float *rval)
{
    BEGIN_WRAP
    *rval = TrainData::missingValue();
    END_WRAP
}

CvStatus TrainData_Close(TrainData *cs)
{
    delete cs;
    return CvStatus::OK;
}

CvStatus TrainData_GetLayout(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getLayout();
    END_WRAP
}

CvStatus TrainData_GetNTrainSamples(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getNTrainSamples();
    END_WRAP
}

CvStatus TrainData_GetNTestSamples(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getNTestSamples();
    END_WRAP
}

CvStatus TrainData_GetNSamples(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getNSamples();
    END_WRAP
}

CvStatus TrainData_GetNVars(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getNVars();
    END_WRAP
}

CvStatus TrainData_GetNAllVars(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getNAllVars();
    END_WRAP
}

CvStatus TrainData_GetSample(const TrainData *cs, InputArray varIdx, int sidx, float* buf)
{
    BEGIN_WRAP
    cs->getSample(varIdx, sidx, buf);
    END_WRAP
}

CvStatus TrainData_GetSamples(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getSamples();
    END_WRAP
}

CvStatus TrainData_GetMissing(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getMissing();
    END_WRAP
}

CvStatus TrainData_GetTrainSamples(const TrainData *cs, int layout, bool compressSamples, bool compressVars, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainSamples(layout, compressSamples, compressVars);
    END_WRAP
}

CvStatus TrainData_GetTrainResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainResponses();
    END_WRAP
}

CvStatus TrainData_GetTrainNormCatResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetTestResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTestResponses();
    END_WRAP
}

CvStatus TrainData_GetTestNormCatResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTestNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getResponses();
    END_WRAP
}

CvStatus TrainData_GetNormCatResponses(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getNormCatResponses();
    END_WRAP
}

CvStatus TrainData_GetSampleWeights(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetTrainSampleWeights(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetTestSampleWeights(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTestSampleWeights();
    END_WRAP
}

CvStatus TrainData_GetVarIdx(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getVarIdx();
    END_WRAP
}

CvStatus TrainData_GetVarType(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getVarType();
    END_WRAP
}

CvStatus TrainData_GetVarSymbolFlags(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getVarSymbolFlags();
    END_WRAP
}

CvStatus TrainData_GetResponseType(const TrainData *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getResponseType();
    END_WRAP
}

CvStatus TrainData_GetTrainSampleIdx(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainSampleIdx();
    END_WRAP
}

CvStatus TrainData_GetTestSampleIdx(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTestSampleIdx();
    END_WRAP
}

CvStatus TrainData_GetValues(const TrainData *cs, int vi, InputArray sidx, float* values)
{
    BEGIN_WRAP
    cs->getValues(vi, sidx, values);
    END_WRAP
}

CvStatus TrainData_GetNormCatValues(const TrainData *cs, int vi, InputArray sidx, int* values)
{
    BEGIN_WRAP
    cs->getNormCatValues(vi, sidx, values);
    END_WRAP
}

CvStatus TrainData_GetDefaultSubstValues(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getDefaultSubstValues();
    END_WRAP
}

CvStatus TrainData_GetCatCount(const TrainData *cs, int vi, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getCatCount(vi);
    END_WRAP
}

CvStatus TrainData_GetClassLabels(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getClassLabels();
    END_WRAP
}

CvStatus TrainData_GetCatOfs(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getCatOfs();
    END_WRAP
}

CvStatus TrainData_GetCatMap(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getCatMap();
    END_WRAP
}

CvStatus TrainData_SetTrainTestSplit(TrainData *cs, int count, bool shuffle)
{
    BEGIN_WRAP
    cs->setTrainTestSplit(count, shuffle);
    END_WRAP
}

CvStatus TrainData_SetTrainTestSplitRatio(TrainData *cs, double ratio, bool shuffle)
{
    BEGIN_WRAP
    cs->setTrainTestSplitRatio(ratio, shuffle);
    END_WRAP
}

CvStatus TrainData_ShuffleTrainTest(TrainData *cs)
{
    BEGIN_WRAP
    cs->shuffleTrainTest();
    END_WRAP
}

CvStatus TrainData_GetTestSamples(const TrainData *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getTestSamples();
    END_WRAP
}

CvStatus TrainData_GetNames(const TrainData *cs, std::vector<String>& names)
{
    BEGIN_WRAP
    cs->getNames(names);
    END_WRAP
}

CvStatus TrainData_GetSubVector(const Mat& vec, const Mat& idx, Mat *rval)
{
    BEGIN_WRAP
    *rval = TrainData::getSubVector(vec, idx);
    END_WRAP
}

CvStatus TrainData_GetSubMatrix(const Mat& matrix, const Mat& idx, int layout, Mat *rval)
{
    BEGIN_WRAP
    *rval = TrainData::getSubMatrix(matrix, idx, layout);
    END_WRAP
}

CvStatus TrainData_LoadFromCSV(const String& filename, int headerLineCount, int responseStartIdx, int responseEndIdx, const String& varTypeSpec, char delimiter, char missch, Ptr<TrainData> *rval)
{
    BEGIN_WRAP
    *rval = TrainData::loadFromCSV(filename, headerLineCount, responseStartIdx, responseEndIdx, varTypeSpec, delimiter, missch);
    END_WRAP
}

CvStatus TrainData_Create(InputArray samples, int layout, InputArray responses,

 InputArray varIdx, InputArray sampleIdx, InputArray sampleWeights, InputArray varType, Ptr<TrainData> *rval)
{
    BEGIN_WRAP
    *rval = TrainData::create(samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType);
    END_WRAP
}

CvStatus StatModel_GetVarCount(const StatModel *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getVarCount();
    END_WRAP
}

CvStatus StatModel_Empty(const StatModel *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->empty();
    END_WRAP
}

CvStatus StatModel_IsTrained(const StatModel *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->isTrained();
    END_WRAP
}

CvStatus StatModel_IsClassifier(const StatModel *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->isClassifier();
    END_WRAP
}

CvStatus StatModel_Train(StatModel *cs, const Ptr<TrainData>& trainData, int flags, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->train(trainData, flags);
    END_WRAP
}

CvStatus StatModel_TrainWithData(StatModel *cs, InputArray samples, int layout, InputArray responses, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->train(samples, layout, responses);
    END_WRAP
}

CvStatus StatModel_CalcError(const StatModel *cs, const Ptr<TrainData>& data, bool test, OutputArray resp, float *rval)
{
    BEGIN_WRAP
    *rval = cs->calcError(data, test, resp);
    END_WRAP
}

CvStatus StatModel_Predict(const StatModel *cs, InputArray samples, OutputArray results, int flags, float *rval)
{
    BEGIN_WRAP
    *rval = cs->predict(samples, results, flags);
    END_WRAP
}

CvStatus NormalBayesClassifier_PredictProb(const NormalBayesClassifier *cs, InputArray inputs, OutputArray outputs, OutputArray outputProbs, int flags, float *rval)
{
    BEGIN_WRAP
    *rval = cs->predictProb(inputs, outputs, outputProbs, flags);
    END_WRAP
}

CvStatus NormalBayesClassifier_Create(Ptr<NormalBayesClassifier> *rval)
{
    BEGIN_WRAP
    *rval = NormalBayesClassifier::create();
    END_WRAP
}

CvStatus NormalBayesClassifier_Load(const String& filepath, const String& nodeName, Ptr<NormalBayesClassifier> *rval)
{
    BEGIN_WRAP
    *rval = NormalBayesClassifier::load(filepath, nodeName);
    END_WRAP
}

CvStatus KNearest_GetDefaultK(const KNearest *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getDefaultK();
    END_WRAP
}

CvStatus KNearest_SetDefaultK(KNearest *cs, int val)
{
    BEGIN_WRAP
    cs->setDefaultK(val);
    END_WRAP
}

CvStatus KNearest_GetIsClassifier(const KNearest *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->getIsClassifier();
    END_WRAP
}

CvStatus KNearest_SetIsClassifier(KNearest *cs, bool val)
{
    BEGIN_WRAP
    cs->setIsClassifier(val);
    END_WRAP
}

CvStatus KNearest_GetEmax(const KNearest *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getEmax();
    END_WRAP
}

CvStatus KNearest_SetEmax(KNearest *cs, int val)
{
    BEGIN_WRAP
    cs->setEmax(val);
    END_WRAP
}

CvStatus KNearest_GetAlgorithmType(const KNearest *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getAlgorithmType();
    END_WRAP
}

CvStatus KNearest_SetAlgorithmType(KNearest *cs, int val)
{
    BEGIN_WRAP
    cs->setAlgorithmType(val);
    END_WRAP
}

CvStatus KNearest_FindNearest(const KNearest *cs, InputArray samples, int k, OutputArray results, OutputArray neighborResponses, OutputArray dist, float *rval)
{
    BEGIN_WRAP
    *rval = cs->findNearest(samples, k, results, neighborResponses, dist);
    END_WRAP
}

CvStatus KNearest_Create(Ptr<KNearest> *rval)
{
    BEGIN_WRAP
    *rval = KNearest::create();
    END_WRAP
}

CvStatus KNearest_Load(const String& filepath, Ptr<KNearest> *rval)
{
    BEGIN_WRAP
    *rval = KNearest::load(filepath);
    END_WRAP
}

CvStatus SVM_GetType(const SVM *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getType();
    END_WRAP
}

CvStatus SVM_SetType(SVM *cs, int val)
{
    BEGIN_WRAP
    cs->setType(val);
    END_WRAP
}

CvStatus SVM_GetGamma(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getGamma();
    END_WRAP
}

CvStatus SVM_SetGamma(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setGamma(val);
    END_WRAP
}

CvStatus SVM_GetCoef0(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getCoef0();
    END_WRAP
}

CvStatus SVM_SetCoef0(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setCoef0(val);
    END_WRAP
}

CvStatus SVM_GetDegree(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getDegree();
    END_WRAP
}

CvStatus SVM_SetDegree(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setDegree(val);
    END_WRAP
}

CvStatus SVM_GetC(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getC();
    END_WRAP
}

CvStatus SVM_SetC(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setC(val);
    END_WRAP
}

CvStatus SVM_GetNu(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getNu();
    END_WRAP
}

CvStatus SVM_SetNu(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setNu(val);
    END_WRAP
}

CvStatus SVM_GetP(const SVM *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getP();
    END_WRAP
}

CvStatus SVM_SetP(SVM *cs, double val)
{
    BEGIN_WRAP
    cs->setP(val);
    END_WRAP
}

CvStatus SVM_GetClassWeights(const SVM *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getClassWeights();
    END_WRAP
}

CvStatus SVM_SetClassWeights(SVM *cs, const Mat &val)
{
    BEGIN_WRAP
    cs->setClassWeights(val);
    END_WRAP
}

CvStatus SVM_GetTermCriteria(const SVM *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus SVM_SetTermCriteria(SVM *cs, const TermCriteria &val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus SVM_GetKernelType(const SVM *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getKernelType();
    END_WRAP
}

CvStatus SVM_SetKernel(SVM *cs, int kernelType)
{
    BEGIN_WRAP
    cs->setKernel(kernelType);
    END_WRAP
}

CvStatus SVM_SetCustomKernel(SVM *cs, const Ptr<SVM::Kernel> &_kernel)
{
    BEGIN_WRAP
    cs->setCustomKernel(_kernel);
    END_WRAP
}

CvStatus SVM_TrainAuto(SVM *cs, const Ptr<TrainData>& data, int kFold, ParamGrid Cgrid, ParamGrid gammaGrid, ParamGrid pGrid, ParamGrid nuGrid, ParamGrid coeffGrid, ParamGrid degreeGrid, bool balanced, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->trainAuto(data, kFold, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);
    END_WRAP
}

CvStatus SVM_TrainAutoWithData(SVM *cs, InputArray samples, int layout, InputArray responses, int kFold, Ptr<ParamGrid> Cgrid, Ptr<ParamGrid> gammaGrid, Ptr<ParamGrid> pGrid, Ptr<ParamGrid> nuGrid, Ptr<ParamGrid> coeffGrid, Ptr<ParamGrid> degreeGrid, bool balanced, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->trainAuto(samples, layout, responses, kFold, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);
    END_WRAP
}

CvStatus SVM_GetSupportVectors(const SVM *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getSupportVectors();
    END_WRAP
}

CvStatus

 SVM_GetUncompressedSupportVectors(const SVM *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getUncompressedSupportVectors();
    END_WRAP
}

CvStatus SVM_GetDecisionFunction(const SVM *cs, int i, OutputArray alpha, OutputArray svidx, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getDecisionFunction(i, alpha, svidx);
    END_WRAP
}

CvStatus SVM_GetDefaultGrid(int param_id, ParamGrid *rval)
{
    BEGIN_WRAP
    *rval = SVM::getDefaultGrid(param_id);
    END_WRAP
}

CvStatus SVM_GetDefaultGridPtr(int param_id, Ptr<ParamGrid> *rval)
{
    BEGIN_WRAP
    *rval = SVM::getDefaultGridPtr(param_id);
    END_WRAP
}

CvStatus SVM_Create(Ptr<SVM> *rval)
{
    BEGIN_WRAP
    *rval = SVM::create();
    END_WRAP
}

CvStatus SVM_Load(const String& filepath, Ptr<SVM> *rval)
{
    BEGIN_WRAP
    *rval = SVM::load(filepath);
    END_WRAP
}

CvStatus EM_GetClustersNumber(const EM *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getClustersNumber();
    END_WRAP
}

CvStatus EM_SetClustersNumber(EM *cs, int val)
{
    BEGIN_WRAP
    cs->setClustersNumber(val);
    END_WRAP
}

CvStatus EM_GetCovarianceMatrixType(const EM *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getCovarianceMatrixType();
    END_WRAP
}

CvStatus EM_SetCovarianceMatrixType(EM *cs, int val)
{
    BEGIN_WRAP
    cs->setCovarianceMatrixType(val);
    END_WRAP
}

CvStatus EM_GetTermCriteria(const EM *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus EM_SetTermCriteria(EM *cs, const TermCriteria &val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus EM_GetWeights(const EM *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getWeights();
    END_WRAP
}

CvStatus EM_GetMeans(const EM *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getMeans();
    END_WRAP
}

CvStatus EM_GetCovs(const EM *cs, std::vector<Mat>& covs)
{
    BEGIN_WRAP
    cs->getCovs(covs);
    END_WRAP
}

CvStatus EM_Predict(const EM *cs, InputArray samples, OutputArray results, int flags, float *rval)
{
    BEGIN_WRAP
    *rval = cs->predict(samples, results, flags);
    END_WRAP
}

CvStatus EM_Predict2(const EM *cs, InputArray sample, OutputArray probs, Vec2d *rval)
{
    BEGIN_WRAP
    *rval = cs->predict2(sample, probs);
    END_WRAP
}

CvStatus EM_TrainEM(EM *cs, InputArray samples, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->trainEM(samples, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_TrainE(EM *cs, InputArray samples, InputArray means0, InputArray covs0, InputArray weights0, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_TrainM(EM *cs, InputArray samples, InputArray probs0, OutputArray logLikelihoods, OutputArray labels, OutputArray probs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->trainM(samples, probs0, logLikelihoods, labels, probs);
    END_WRAP
}

CvStatus EM_Create(Ptr<EM> *rval)
{
    BEGIN_WRAP
    *rval = EM::create();
    END_WRAP
}

CvStatus EM_Load(const String& filepath, const String& nodeName, Ptr<EM> *rval)
{
    BEGIN_WRAP
    *rval = EM::load(filepath, nodeName);
    END_WRAP
}

CvStatus DTrees_GetMaxCategories(const DTrees *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getMaxCategories();
    END_WRAP
}

CvStatus DTrees_SetMaxCategories(DTrees *cs, int val)
{
    BEGIN_WRAP
    cs->setMaxCategories(val);
    END_WRAP
}

CvStatus DTrees_GetMaxDepth(const DTrees *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getMaxDepth();
    END_WRAP
}

CvStatus DTrees_SetMaxDepth(DTrees *cs, int val)
{
    BEGIN_WRAP
    cs->setMaxDepth(val);
    END_WRAP
}

CvStatus DTrees_GetMinSampleCount(const DTrees *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getMinSampleCount();
    END_WRAP
}

CvStatus DTrees_SetMinSampleCount(DTrees *cs, int val)
{
    BEGIN_WRAP
    cs->setMinSampleCount(val);
    END_WRAP
}

CvStatus DTrees_GetCVFolds(const DTrees *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getCVFolds();
    END_WRAP
}

CvStatus DTrees_SetCVFolds(DTrees *cs, int val)
{
    BEGIN_WRAP
    cs->setCVFolds(val);
    END_WRAP
}

CvStatus DTrees_GetUseSurrogates(const DTrees *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->getUseSurrogates();
    END_WRAP
}

CvStatus DTrees_SetUseSurrogates(DTrees *cs, bool val)
{
    BEGIN_WRAP
    cs->setUseSurrogates(val);
    END_WRAP
}

CvStatus DTrees_GetUse1SERule(const DTrees *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->getUse1SERule();
    END_WRAP
}

CvStatus DTrees_SetUse1SERule(DTrees *cs, bool val)
{
    BEGIN_WRAP
    cs->setUse1SERule(val);
    END_WRAP
}

CvStatus DTrees_GetTruncatePrunedTree(const DTrees *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->getTruncatePrunedTree();
    END_WRAP
}

CvStatus DTrees_SetTruncatePrunedTree(DTrees *cs, bool val)
{
    BEGIN_WRAP
    cs->setTruncatePrunedTree(val);
    END_WRAP
}

CvStatus DTrees_GetRegressionAccuracy(const DTrees *cs, float *rval)
{
    BEGIN_WRAP
    *rval = cs->getRegressionAccuracy();
    END_WRAP
}

CvStatus DTrees_SetRegressionAccuracy(DTrees *cs, float val)
{
    BEGIN_WRAP
    cs->setRegressionAccuracy(val);
    END_WRAP
}

CvStatus DTrees_GetPriors(const DTrees *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getPriors();
    END_WRAP
}

CvStatus DTrees_SetPriors(DTrees *cs, const Mat &val)
{
    BEGIN_WRAP
    cs->setPriors(val);
    END_WRAP
}

CvStatus DTrees_GetRoots(const DTrees *cs, const std::vector<int>** rval)
{
    BEGIN_WRAP
    *rval = &cs->getRoots();
    END_WRAP
}

CvStatus DTrees_GetNodes(const DTrees *cs, const std::vector<DTrees::Node>** rval)
{
    BEGIN_WRAP
    *rval = &cs->getNodes();
    END_WRAP
}

CvStatus DTrees_GetSplits(const DTrees *cs, const std::vector<DTrees::Split>** rval)
{
    BEGIN_WRAP
    *rval = &cs->getSplits();
    END_WRAP
}

CvStatus DTrees_GetSubsets(const DTrees *cs, const std::vector<int>** rval)
{
    BEGIN_WRAP
    *rval = &cs->getSubsets();
    END_WRAP
}

CvStatus DTrees_Create(Ptr<DTrees> *rval)
{
    BEGIN_WRAP
    *rval = DTrees::create();
    END_WRAP
}

CvStatus DTrees_Load(const String& filepath, const String& nodeName, Ptr<DTrees> *rval)
{
    BEGIN_WRAP
    *rval = DTrees::load(filepath, nodeName);
    END_WRAP
}

CvStatus RTrees_GetCalculateVarImportance(const RTrees *cs, bool *rval)
{
    BEGIN_WRAP
    *rval = cs->getCalculateVarImportance();
    END_WRAP
}

CvStatus RTrees_SetCalculateVarImportance(RTrees *cs, bool val)
{
    BEGIN_WRAP
    cs->setCalculateVarImportance(val);
    END_WRAP
}

CvStatus RTrees_GetActiveVarCount(const RTrees *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getActiveVar

Count();
    END_WRAP
}

CvStatus RTrees_SetActiveVarCount(RTrees *cs, int val)
{
    BEGIN_WRAP
    cs->setActiveVarCount(val);
    END_WRAP
}

CvStatus RTrees_GetTermCriteria(const RTrees *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus RTrees_SetTermCriteria(RTrees *cs, const TermCriteria &val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus RTrees_GetVarImportance(const RTrees *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getVarImportance();
    END_WRAP
}

CvStatus RTrees_GetVotes(const RTrees *cs, InputArray samples, OutputArray results, int flags)
{
    BEGIN_WRAP
    cs->getVotes(samples, results, flags);
    END_WRAP
}

CvStatus RTrees_GetOOBError(const RTrees *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getOOBError();
    END_WRAP
}

CvStatus RTrees_Create(Ptr<RTrees> *rval)
{
    BEGIN_WRAP
    *rval = RTrees::create();
    END_WRAP
}

CvStatus RTrees_Load(const String& filepath, const String& nodeName, Ptr<RTrees> *rval)
{
    BEGIN_WRAP
    *rval = RTrees::load(filepath, nodeName);
    END_WRAP
}

CvStatus Boost_GetBoostType(const Boost *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getBoostType();
    END_WRAP
}

CvStatus Boost_SetBoostType(Boost *cs, int val)
{
    BEGIN_WRAP
    cs->setBoostType(val);
    END_WRAP
}

CvStatus Boost_GetWeakCount(const Boost *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getWeakCount();
    END_WRAP
}

CvStatus Boost_SetWeakCount(Boost *cs, int val)
{
    BEGIN_WRAP
    cs->setWeakCount(val);
    END_WRAP
}

CvStatus Boost_GetWeightTrimRate(const Boost *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getWeightTrimRate();
    END_WRAP
}

CvStatus Boost_SetWeightTrimRate(Boost *cs, double val)
{
    BEGIN_WRAP
    cs->setWeightTrimRate(val);
    END_WRAP
}

CvStatus Boost_Create(Ptr<Boost> *rval)
{
    BEGIN_WRAP
    *rval = Boost::create();
    END_WRAP
}

CvStatus Boost_Load(const String& filepath, const String& nodeName, Ptr<Boost> *rval)
{
    BEGIN_WRAP
    *rval = Boost::load(filepath, nodeName);
    END_WRAP
}

CvStatus ANN_MLP_GetTrainMethod(const ANN_MLP *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainMethod();
    END_WRAP
}

CvStatus ANN_MLP_SetTrainMethod(ANN_MLP *cs, int method, double param1, double param2)
{
    BEGIN_WRAP
    cs->setTrainMethod(method, param1, param2);
    END_WRAP
}

CvStatus ANN_MLP_SetActivationFunction(ANN_MLP *cs, int type, double param1, double param2)
{
    BEGIN_WRAP
    cs->setActivationFunction(type, param1, param2);
    END_WRAP
}

CvStatus ANN_MLP_SetLayerSizes(ANN_MLP *cs, InputArray _layer_sizes)
{
    BEGIN_WRAP
    cs->setLayerSizes(_layer_sizes);
    END_WRAP
}

CvStatus ANN_MLP_GetLayerSizes(const ANN_MLP *cs, cv::Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getLayerSizes();
    END_WRAP
}

CvStatus ANN_MLP_GetTermCriteria(const ANN_MLP *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus ANN_MLP_SetTermCriteria(ANN_MLP *cs, TermCriteria val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus ANN_MLP_GetBackpropWeightScale(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getBackpropWeightScale();
    END_WRAP
}

CvStatus ANN_MLP_SetBackpropWeightScale(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setBackpropWeightScale(val);
    END_WRAP
}

CvStatus ANN_MLP_GetBackpropMomentumScale(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getBackpropMomentumScale();
    END_WRAP
}

CvStatus ANN_MLP_SetBackpropMomentumScale(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setBackpropMomentumScale(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDW0(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getRpropDW0();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDW0(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setRpropDW0(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWPlus(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getRpropDWPlus();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWPlus(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setRpropDWPlus(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMinus(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getRpropDWMinus();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMinus(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setRpropDWMinus(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMin(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getRpropDWMin();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMin(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setRpropDWMin(val);
    END_WRAP
}

CvStatus ANN_MLP_GetRpropDWMax(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getRpropDWMax();
    END_WRAP
}

CvStatus ANN_MLP_SetRpropDWMax(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setRpropDWMax(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealInitialT(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getAnnealInitialT();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealInitialT(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setAnnealInitialT(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealFinalT(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getAnnealFinalT();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealFinalT(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setAnnealFinalT(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealCoolingRatio(const ANN_MLP *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getAnnealCoolingRatio();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealCoolingRatio(ANN_MLP *cs, double val)
{
    BEGIN_WRAP
    cs->setAnnealCoolingRatio(val);
    END_WRAP
}

CvStatus ANN_MLP_GetAnnealItePerStep(const ANN_MLP *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getAnnealItePerStep();
    END_WRAP
}

CvStatus ANN_MLP_SetAnnealItePerStep(ANN_MLP *cs, int val)
{
    BEGIN_WRAP
    cs->setAnnealItePerStep(val);
    END_WRAP
}

CvStatus ANN_MLP_GetWeights(const ANN_MLP *cs, int layerIdx, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getWeights(layerIdx);
    END_WRAP
}

CvStatus ANN_MLP_Create(Ptr<ANN_MLP> *rval)
{
    BEGIN_WRAP
    *rval = ANN_MLP::create();
    END_WRAP
}

CvStatus ANN_MLP_Load(const String& filepath, Ptr<ANN_MLP> *rval)
{
    BEGIN_WRAP
    *rval = ANN_MLP::load(filepath);
    END_WRAP
}

CvStatus LogisticRegression_GetLearningRate(const LogisticRegression *cs, double *rval)
{
    BEGIN_WRAP
    *rval = cs->getLearningRate();
    END_WRAP
}

CvStatus LogisticRegression_SetLearningRate(LogisticRegression *cs, double val)
{
    BEGIN_WRAP


    cs->setLearningRate(val);
    END_WRAP
}

CvStatus LogisticRegression_GetIterations(const LogisticRegression *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getIterations();
    END_WRAP
}

CvStatus LogisticRegression_SetIterations(LogisticRegression *cs, int val)
{
    BEGIN_WRAP
    cs->setIterations(val);
    END_WRAP
}

CvStatus LogisticRegression_GetRegularization(const LogisticRegression *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getRegularization();
    END_WRAP
}

CvStatus LogisticRegression_SetRegularization(LogisticRegression *cs, int val)
{
    BEGIN_WRAP
    cs->setRegularization(val);
    END_WRAP
}

CvStatus LogisticRegression_GetTrainMethod(const LogisticRegression *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getTrainMethod();
    END_WRAP
}

CvStatus LogisticRegression_SetTrainMethod(LogisticRegression *cs, int val)
{
    BEGIN_WRAP
    cs->setTrainMethod(val);
    END_WRAP
}

CvStatus LogisticRegression_GetMiniBatchSize(const LogisticRegression *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getMiniBatchSize();
    END_WRAP
}

CvStatus LogisticRegression_SetMiniBatchSize(LogisticRegression *cs, int val)
{
    BEGIN_WRAP
    cs->setMiniBatchSize(val);
    END_WRAP
}

CvStatus LogisticRegression_GetTermCriteria(const LogisticRegression *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus LogisticRegression_SetTermCriteria(LogisticRegression *cs, TermCriteria val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus LogisticRegression_Predict(const LogisticRegression *cs, InputArray samples, OutputArray results, int flags, float *rval)
{
    BEGIN_WRAP
    *rval = cs->predict(samples, results, flags);
    END_WRAP
}

CvStatus LogisticRegression_GetLearntThetas(const LogisticRegression *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->get_learnt_thetas();
    END_WRAP
}

CvStatus LogisticRegression_Create(Ptr<LogisticRegression> *rval)
{
    BEGIN_WRAP
    *rval = LogisticRegression::create();
    END_WRAP
}

CvStatus LogisticRegression_Load(const String& filepath, const String& nodeName, Ptr<LogisticRegression> *rval)
{
    BEGIN_WRAP
    *rval = LogisticRegression::load(filepath, nodeName);
    END_WRAP
}

CvStatus SVMSGD_GetWeights(SVMSGD *cs, Mat *rval)
{
    BEGIN_WRAP
    *rval = cs->getWeights();
    END_WRAP
}

CvStatus SVMSGD_GetShift(SVMSGD *cs, float *rval)
{
    BEGIN_WRAP
    *rval = cs->getShift();
    END_WRAP
}

CvStatus SVMSGD_Create(Ptr<SVMSGD> *rval)
{
    BEGIN_WRAP
    *rval = SVMSGD::create();
    END_WRAP
}

CvStatus SVMSGD_Load(const String& filepath, const String& nodeName, Ptr<SVMSGD> *rval)
{
    BEGIN_WRAP
    *rval = SVMSGD::load(filepath, nodeName);
    END_WRAP
}

CvStatus SVMSGD_SetOptimalParameters(SVMSGD *cs, int svmsgdType, int marginType)
{
    BEGIN_WRAP
    cs->setOptimalParameters(svmsgdType, marginType);
    END_WRAP
}

CvStatus SVMSGD_GetSvmsgdType(const SVMSGD *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getSvmsgdType();
    END_WRAP
}

CvStatus SVMSGD_SetSvmsgdType(SVMSGD *cs, int svmsgdType)
{
    BEGIN_WRAP
    cs->setSvmsgdType(svmsgdType);
    END_WRAP
}

CvStatus SVMSGD_GetMarginType(const SVMSGD *cs, int *rval)
{
    BEGIN_WRAP
    *rval = cs->getMarginType();
    END_WRAP
}

CvStatus SVMSGD_SetMarginType(SVMSGD *cs, int marginType)
{
    BEGIN_WRAP
    cs->setMarginType(marginType);
    END_WRAP
}

CvStatus SVMSGD_GetMarginRegularization(const SVMSGD *cs, float *rval)
{
    BEGIN_WRAP
    *rval = cs->getMarginRegularization();
    END_WRAP
}

CvStatus SVMSGD_SetMarginRegularization(SVMSGD *cs, float marginRegularization)
{
    BEGIN_WRAP
    cs->setMarginRegularization(marginRegularization);
    END_WRAP
}

CvStatus SVMSGD_GetInitialStepSize(const SVMSGD *cs, float *rval)
{
    BEGIN_WRAP
    *rval = cs->getInitialStepSize();
    END_WRAP
}

CvStatus SVMSGD_SetInitialStepSize(SVMSGD *cs, float InitialStepSize)
{
    BEGIN_WRAP
    cs->setInitialStepSize(InitialStepSize);
    END_WRAP
}

CvStatus SVMSGD_GetStepDecreasingPower(const SVMSGD *cs, float *rval)
{
    BEGIN_WRAP
    *rval = cs->getStepDecreasingPower();
    END_WRAP
}

CvStatus SVMSGD_SetStepDecreasingPower(SVMSGD *cs, float stepDecreasingPower)
{
    BEGIN_WRAP
    cs->setStepDecreasingPower(stepDecreasingPower);
    END_WRAP
}

CvStatus SVMSGD_GetTermCriteria(const SVMSGD *cs, TermCriteria *rval)
{
    BEGIN_WRAP
    *rval = cs->getTermCriteria();
    END_WRAP
}

CvStatus SVMSGD_SetTermCriteria(SVMSGD *cs, const cv::TermCriteria &val)
{
    BEGIN_WRAP
    cs->setTermCriteria(val);
    END_WRAP
}

CvStatus RandMVNormal(InputArray mean, InputArray cov, int nsamples, OutputArray samples)
{
    BEGIN_WRAP
    cv::randMVNormal(mean, cov, nsamples, samples);
    END_WRAP
}

CvStatus CreateConcentricSpheresTestSet(int nsamples, int nfeatures, int nclasses, OutputArray samples, OutputArray responses)
{
    BEGIN_WRAP
    cv::createConcentricSpheresTestSet(nsamples, nfeatures, nclasses, samples, responses);
    END_WRAP
}

template<class SimulatedAnnealingSolverSystem>
CvStatus SimulatedAnnealingSolver(SimulatedAnnealingSolverSystem& solverSystem, double initialTemperature, double finalTemperature, double coolingRatio, size_t iterationsPerStep, double* lastTemperature, cv::RNG& rngEnergy, int *rval)
{
    BEGIN_WRAP
    *rval = cv::simulatedAnnealingSolver(solverSystem, initialTemperature, finalTemperature, coolingRatio, iterationsPerStep, lastTemperature, rngEnergy);
    END_WRAP
}
