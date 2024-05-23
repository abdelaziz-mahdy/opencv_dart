// ignore_for_file: non_constant_identifier_names
library cv_ml;

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

import '../core/base.dart';
import '../core/mat.dart';
import '../opencv.g.dart' as cvg;

class KNearest extends CvStruct<cvg.KNearest> {
  KNearest._(cvg.KNearestPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory KNearest.fromNative(cvg.KNearestPtr ptr) => KNearest._(ptr);

  factory KNearest.empty() {
    final p = calloc<cvg.KNearest>();
    CFFI.KNearest_Create(p);
    return KNearest._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.KNearest_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat findNearest(Mat samples, int k,
      {Mat? results, Mat? neighborResponses, Mat? dists}) {
    results ??= Mat.empty();
    neighborResponses ??= Mat.empty();
    dists ??= Mat.empty();

    // Allocate pointers for the results, neighborResponses, dists, and rval
    final resultsPtr = results.ptr.cast<cvg.Mat>();
    final neighborResponsesPtr = neighborResponses.ptr.cast<cvg.Mat>();
    final distsPtr = dists.ptr.cast<cvg.Mat>();
    final rvalPtr = calloc<ffi.Float>();

    cvRun(() => CFFI.KNearest_FindNearest(ref, samples.ref, k, resultsPtr,
        neighborResponsesPtr, distsPtr, rvalPtr));

    final rval = rvalPtr.value;
    calloc.free(rvalPtr);

    return results;
  }

  static final finalizer =
      OcvFinalizer<cvg.KNearestPtr>(CFFI.addresses.KNearest_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.KNearest get ref => ptr.ref;
}

class SVM extends CvStruct<cvg.SVM> {
  SVM._(cvg.SVMPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory SVM.fromNative(cvg.SVMPtr ptr) => SVM._(ptr);

  factory SVM.empty() {
    final p = calloc<cvg.SVM>();
    CFFI.SVM_Create(p);
    return SVM._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.SVM_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.SVM_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer = OcvFinalizer<cvg.SVMPtr>(CFFI.addresses.SVM_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.SVM get ref => ptr.ref;
}

class DTrees extends CvStruct<cvg.DTrees> {
  DTrees._(cvg.DTreesPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory DTrees.fromNative(cvg.DTreesPtr ptr) => DTrees._(ptr);

  factory DTrees.empty() {
    final p = calloc<cvg.DTrees>();
    CFFI.DTrees_Create(p);
    return DTrees._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.DTrees_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.DTrees_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer =
      OcvFinalizer<cvg.DTreesPtr>(CFFI.addresses.DTrees_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.DTrees get ref => ptr.ref;
}

class RTrees extends CvStruct<cvg.RTrees> {
  RTrees._(cvg.RTreesPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory RTrees.fromNative(cvg.RTreesPtr ptr) => RTrees._(ptr);

  factory RTrees.empty() {
    final p = calloc<cvg.RTrees>();
    CFFI.RTrees_Create(p);
    return RTrees._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.RTrees_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.RTrees_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer =
      OcvFinalizer<cvg.RTreesPtr>(CFFI.addresses.RTrees_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.RTrees get ref => ptr.ref;
}

class Boost extends CvStruct<cvg.Boost> {
  Boost._(cvg.BoostPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory Boost.fromNative(cvg.BoostPtr ptr) => Boost._(ptr);

  factory Boost.empty() {
    final p = calloc<cvg.Boost>();
    CFFI.Boost_Create(p);
    return Boost._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.Boost_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.Boost_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer =
      OcvFinalizer<cvg.BoostPtr>(CFFI.addresses.Boost_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.Boost get ref => ptr.ref;
}

class ANN_MLP extends CvStruct<cvg.ANN_MLP> {
  ANN_MLP._(cvg.ANN_MLPPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory ANN_MLP.fromNative(cvg.ANN_MLPPtr ptr) => ANN_MLP._(ptr);

  factory ANN_MLP.empty() {
    final p = calloc<cvg.ANN_MLP>();
    CFFI.ANN_MLP_Create(p);
    return ANN_MLP._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.ANN_MLP_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.ANN_MLP_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer =
      OcvFinalizer<cvg.ANN_MLPPtr>(CFFI.addresses.ANN_MLP_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.ANN_MLP get ref => ptr.ref;
}

class LogisticRegression extends CvStruct<cvg.LogisticRegression> {
  LogisticRegression._(cvg.LogisticRegressionPtr ptr) : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory LogisticRegression.fromNative(cvg.LogisticRegressionPtr ptr) =>
      LogisticRegression._(ptr);

  factory LogisticRegression.empty() {
    final p = calloc<cvg.LogisticRegression>();
    CFFI.LogisticRegression_Create(p);
    return LogisticRegression._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() =>
        CFFI.LogisticRegression_Train(ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() =>
        CFFI.LogisticRegression_Predict(ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer = OcvFinalizer<cvg.LogisticRegressionPtr>(
      CFFI.addresses.LogisticRegression_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.LogisticRegression get ref => ptr.ref;
}

class NormalBayesClassifier extends CvStruct<cvg.NormalBayesClassifier> {
  NormalBayesClassifier._(cvg.NormalBayesClassifierPtr ptr)
      : super.fromPointer(ptr) {
    finalizer.attach(this, ptr.cast());
  }

  factory NormalBayesClassifier.fromNative(cvg.NormalBayesClassifierPtr ptr) =>
      NormalBayesClassifier._(ptr);

  factory NormalBayesClassifier.empty() {
    final p = calloc<cvg.NormalBayesClassifier>();
    CFFI.NormalBayesClassifier_Create(p);
    return NormalBayesClassifier._(p);
  }

  void train(Mat samples, int layout, Mat responses) {
    cvRun(() => CFFI.NormalBayesClassifier_Train(
        ref, samples.ref, layout, responses.ref));
  }

  Mat predict(Mat samples, {Mat? results, int flags = 0}) {
    results ??= Mat.empty();
    cvRun(() => CFFI.NormalBayesClassifier_Predict(
        ref, samples.ref, results!.ref, flags));
    return results;
  }

  static final finalizer = OcvFinalizer<cvg.NormalBayesClassifierPtr>(
      CFFI.addresses.NormalBayesClassifier_Close);

  @override
  List<int> get props => [ptr.address];

  @override
  cvg.NormalBayesClassifier get ref => ptr.ref;
}

void initializeML() {
  CFFI.initialize();
}
