// coverage:ignore-file
// opencv_dart - OpenCV bindings for Dart language
//    some c wrappers were from gocv: https://github.com/hybridgroup/gocv
//    License: Apache-2.0 https://github.com/hybridgroup/gocv/blob/release/LICENSE.txt
// Author: Rainyl
// License: Apache-2.0
// Date: 2024/01/28

// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.
// ignore_for_file: type=lint
import 'dart:ffi' as ffi;
import 'package:dartcv4/src/g/types.g.dart' as imp1;

/// Native bindings for OpenCV - Dnn
///
class CvNativeDnn {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  CvNativeDnn(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  CvNativeDnn.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  void cv_dnn_AsyncArray_close(
    AsyncArrayPtr a,
  ) {
    return _cv_dnn_AsyncArray_close(
      a,
    );
  }

  late final _cv_dnn_AsyncArray_closePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(AsyncArrayPtr)>>(
          'cv_dnn_AsyncArray_close');
  late final _cv_dnn_AsyncArray_close =
      _cv_dnn_AsyncArray_closePtr.asFunction<void Function(AsyncArrayPtr)>();

  ffi.Pointer<CvStatus> cv_dnn_AsyncArray_get(
    AsyncArray async_out,
    Mat out,
  ) {
    return _cv_dnn_AsyncArray_get(
      async_out,
      out,
    );
  }

  late final _cv_dnn_AsyncArray_getPtr = _lookup<
          ffi.NativeFunction<ffi.Pointer<CvStatus> Function(AsyncArray, Mat)>>(
      'cv_dnn_AsyncArray_get');
  late final _cv_dnn_AsyncArray_get = _cv_dnn_AsyncArray_getPtr
      .asFunction<ffi.Pointer<CvStatus> Function(AsyncArray, Mat)>();

  ffi.Pointer<CvStatus> cv_dnn_AsyncArray_new(
    ffi.Pointer<AsyncArray> rval,
  ) {
    return _cv_dnn_AsyncArray_new(
      rval,
    );
  }

  late final _cv_dnn_AsyncArray_newPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<AsyncArray>)>>('cv_dnn_AsyncArray_new');
  late final _cv_dnn_AsyncArray_new = _cv_dnn_AsyncArray_newPtr
      .asFunction<ffi.Pointer<CvStatus> Function(ffi.Pointer<AsyncArray>)>();

  void cv_dnn_Layer_close(
    LayerPtr layer,
  ) {
    return _cv_dnn_Layer_close(
      layer,
    );
  }

  late final _cv_dnn_Layer_closePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(LayerPtr)>>(
          'cv_dnn_Layer_close');
  late final _cv_dnn_Layer_close =
      _cv_dnn_Layer_closePtr.asFunction<void Function(LayerPtr)>();

  ffi.Pointer<CvStatus> cv_dnn_Layer_getName(
    Layer layer,
    ffi.Pointer<ffi.Pointer<ffi.Char>> rval,
  ) {
    return _cv_dnn_Layer_getName(
      layer,
      rval,
    );
  }

  late final _cv_dnn_Layer_getNamePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Layer,
              ffi.Pointer<ffi.Pointer<ffi.Char>>)>>('cv_dnn_Layer_getName');
  late final _cv_dnn_Layer_getName = _cv_dnn_Layer_getNamePtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Layer, ffi.Pointer<ffi.Pointer<ffi.Char>>)>();

  ffi.Pointer<CvStatus> cv_dnn_Layer_getType(
    Layer layer,
    ffi.Pointer<ffi.Pointer<ffi.Char>> rval,
  ) {
    return _cv_dnn_Layer_getType(
      layer,
      rval,
    );
  }

  late final _cv_dnn_Layer_getTypePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Layer,
              ffi.Pointer<ffi.Pointer<ffi.Char>>)>>('cv_dnn_Layer_getType');
  late final _cv_dnn_Layer_getType = _cv_dnn_Layer_getTypePtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Layer, ffi.Pointer<ffi.Pointer<ffi.Char>>)>();

  ffi.Pointer<CvStatus> cv_dnn_Layer_inputNameToIndex(
    Layer layer,
    ffi.Pointer<ffi.Char> name,
    ffi.Pointer<ffi.Int> rval,
  ) {
    return _cv_dnn_Layer_inputNameToIndex(
      layer,
      name,
      rval,
    );
  }

  late final _cv_dnn_Layer_inputNameToIndexPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Layer, ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Int>)>>('cv_dnn_Layer_inputNameToIndex');
  late final _cv_dnn_Layer_inputNameToIndex =
      _cv_dnn_Layer_inputNameToIndexPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Layer, ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Int>)>();

  ffi.Pointer<CvStatus> cv_dnn_Layer_outputNameToIndex(
    Layer layer,
    ffi.Pointer<ffi.Char> name,
    ffi.Pointer<ffi.Int> rval,
  ) {
    return _cv_dnn_Layer_outputNameToIndex(
      layer,
      name,
      rval,
    );
  }

  late final _cv_dnn_Layer_outputNameToIndexPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Layer, ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Int>)>>('cv_dnn_Layer_outputNameToIndex');
  late final _cv_dnn_Layer_outputNameToIndex =
      _cv_dnn_Layer_outputNameToIndexPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Layer, ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Int>)>();

  ffi.Pointer<CvStatus> cv_dnn_NMSBoxes(
    VecRect bboxes,
    VecF32 scores,
    double score_threshold,
    double nms_threshold,
    ffi.Pointer<VecI32> indices,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_NMSBoxes(
      bboxes,
      scores,
      score_threshold,
      nms_threshold,
      indices,
      callback,
    );
  }

  late final _cv_dnn_NMSBoxesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecRect, VecF32, ffi.Float, ffi.Float,
              ffi.Pointer<VecI32>, imp1.CvCallback_0)>>('cv_dnn_NMSBoxes');
  late final _cv_dnn_NMSBoxes = _cv_dnn_NMSBoxesPtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecRect, VecF32, double, double,
          ffi.Pointer<VecI32>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_NMSBoxes_1(
    VecRect bboxes,
    VecF32 scores,
    double score_threshold,
    double nms_threshold,
    ffi.Pointer<VecI32> indices,
    double eta,
    int top_k,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_NMSBoxes_1(
      bboxes,
      scores,
      score_threshold,
      nms_threshold,
      indices,
      eta,
      top_k,
      callback,
    );
  }

  late final _cv_dnn_NMSBoxes_1Ptr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecRect,
              VecF32,
              ffi.Float,
              ffi.Float,
              ffi.Pointer<VecI32>,
              ffi.Float,
              ffi.Int,
              imp1.CvCallback_0)>>('cv_dnn_NMSBoxes_1');
  late final _cv_dnn_NMSBoxes_1 = _cv_dnn_NMSBoxes_1Ptr.asFunction<
      ffi.Pointer<CvStatus> Function(VecRect, VecF32, double, double,
          ffi.Pointer<VecI32>, double, int, imp1.CvCallback_0)>();

  void cv_dnn_Net_close(
    NetPtr net,
  ) {
    return _cv_dnn_Net_close(
      net,
    );
  }

  late final _cv_dnn_Net_closePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(NetPtr)>>(
          'cv_dnn_Net_close');
  late final _cv_dnn_Net_close =
      _cv_dnn_Net_closePtr.asFunction<void Function(NetPtr)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_create(
    ffi.Pointer<Net> rval,
  ) {
    return _cv_dnn_Net_create(
      rval,
    );
  }

  late final _cv_dnn_Net_createPtr = _lookup<
          ffi.NativeFunction<ffi.Pointer<CvStatus> Function(ffi.Pointer<Net>)>>(
      'cv_dnn_Net_create');
  late final _cv_dnn_Net_create = _cv_dnn_Net_createPtr
      .asFunction<ffi.Pointer<CvStatus> Function(ffi.Pointer<Net>)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_dump(
    Net net,
    ffi.Pointer<ffi.Pointer<ffi.Char>> rval,
  ) {
    return _cv_dnn_Net_dump(
      net,
      rval,
    );
  }

  late final _cv_dnn_Net_dumpPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<ffi.Pointer<ffi.Char>>)>>('cv_dnn_Net_dump');
  late final _cv_dnn_Net_dump = _cv_dnn_Net_dumpPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Net, ffi.Pointer<ffi.Pointer<ffi.Char>>)>();

  bool cv_dnn_Net_empty(
    Net net,
  ) {
    return _cv_dnn_Net_empty(
      net,
    );
  }

  late final _cv_dnn_Net_emptyPtr =
      _lookup<ffi.NativeFunction<ffi.Bool Function(Net)>>('cv_dnn_Net_empty');
  late final _cv_dnn_Net_empty =
      _cv_dnn_Net_emptyPtr.asFunction<bool Function(Net)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_forward(
    Net net,
    ffi.Pointer<ffi.Char> outputName,
    ffi.Pointer<Mat> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_forward(
      net,
      outputName,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_forwardPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<ffi.Char>,
              ffi.Pointer<Mat>, imp1.CvCallback_0)>>('cv_dnn_Net_forward');
  late final _cv_dnn_Net_forward = _cv_dnn_Net_forwardPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Net, ffi.Pointer<ffi.Char>, ffi.Pointer<Mat>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_forwardAsync(
    Net net,
    ffi.Pointer<ffi.Char> outputName,
    ffi.Pointer<AsyncArray> rval,
  ) {
    return _cv_dnn_Net_forwardAsync(
      net,
      outputName,
      rval,
    );
  }

  late final _cv_dnn_Net_forwardAsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<ffi.Char>,
              ffi.Pointer<AsyncArray>)>>('cv_dnn_Net_forwardAsync');
  late final _cv_dnn_Net_forwardAsync = _cv_dnn_Net_forwardAsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Net, ffi.Pointer<ffi.Char>, ffi.Pointer<AsyncArray>)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_forwardLayers(
    Net net,
    ffi.Pointer<VecMat> outputBlobs,
    VecVecChar outBlobNames,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_forwardLayers(
      net,
      outputBlobs,
      outBlobNames,
      callback,
    );
  }

  late final _cv_dnn_Net_forwardLayersPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<VecMat>, VecVecChar,
              imp1.CvCallback_0)>>('cv_dnn_Net_forwardLayers');
  late final _cv_dnn_Net_forwardLayers =
      _cv_dnn_Net_forwardLayersPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<VecMat>, VecVecChar, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_fromNet(
    Net net,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_fromNet(
      net,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_fromNetPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<Net>, imp1.CvCallback_0)>>('cv_dnn_Net_fromNet');
  late final _cv_dnn_Net_fromNet = _cv_dnn_Net_fromNetPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Net, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getInputDetails(
    Net net,
    ffi.Pointer<VecF32> scales,
    ffi.Pointer<VecI32> zeropoints,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_getInputDetails(
      net,
      scales,
      zeropoints,
      callback,
    );
  }

  late final _cv_dnn_Net_getInputDetailsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Net,
              ffi.Pointer<VecF32>,
              ffi.Pointer<VecI32>,
              imp1.CvCallback_0)>>('cv_dnn_Net_getInputDetails');
  late final _cv_dnn_Net_getInputDetails =
      _cv_dnn_Net_getInputDetailsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<VecF32>,
              ffi.Pointer<VecI32>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getLayer(
    Net net,
    int layerid,
    ffi.Pointer<Layer> rval,
  ) {
    return _cv_dnn_Net_getLayer(
      net,
      layerid,
      rval,
    );
  }

  late final _cv_dnn_Net_getLayerPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Int, ffi.Pointer<Layer>)>>('cv_dnn_Net_getLayer');
  late final _cv_dnn_Net_getLayer = _cv_dnn_Net_getLayerPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Net, int, ffi.Pointer<Layer>)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getLayerNames(
    Net net,
    ffi.Pointer<VecVecChar> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_getLayerNames(
      net,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_getLayerNamesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<VecVecChar>,
              imp1.CvCallback_0)>>('cv_dnn_Net_getLayerNames');
  late final _cv_dnn_Net_getLayerNames =
      _cv_dnn_Net_getLayerNamesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<VecVecChar>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getPerfProfile(
    Net net,
    ffi.Pointer<ffi.Int64> rval,
    ffi.Pointer<VecF64> layersTimes,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_getPerfProfile(
      net,
      rval,
      layersTimes,
      callback,
    );
  }

  late final _cv_dnn_Net_getPerfProfilePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Net,
              ffi.Pointer<ffi.Int64>,
              ffi.Pointer<VecF64>,
              imp1.CvCallback_0)>>('cv_dnn_Net_getPerfProfile');
  late final _cv_dnn_Net_getPerfProfile =
      _cv_dnn_Net_getPerfProfilePtr.asFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<ffi.Int64>,
              ffi.Pointer<VecF64>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getUnconnectedOutLayers(
    Net net,
    ffi.Pointer<VecI32> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_getUnconnectedOutLayers(
      net,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_getUnconnectedOutLayersPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<VecI32>,
              imp1.CvCallback_0)>>('cv_dnn_Net_getUnconnectedOutLayers');
  late final _cv_dnn_Net_getUnconnectedOutLayers =
      _cv_dnn_Net_getUnconnectedOutLayersPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<VecI32>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_getUnconnectedOutLayersNames(
    Net net,
    ffi.Pointer<VecVecChar> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_getUnconnectedOutLayersNames(
      net,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_getUnconnectedOutLayersNamesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, ffi.Pointer<VecVecChar>,
              imp1.CvCallback_0)>>('cv_dnn_Net_getUnconnectedOutLayersNames');
  late final _cv_dnn_Net_getUnconnectedOutLayersNames =
      _cv_dnn_Net_getUnconnectedOutLayersNamesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Net, ffi.Pointer<VecVecChar>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNet(
    ffi.Pointer<ffi.Char> model,
    ffi.Pointer<ffi.Char> config,
    ffi.Pointer<ffi.Char> framework,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNet(
      model,
      config,
      framework,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNet');
  late final _cv_dnn_Net_readNet = _cv_dnn_Net_readNetPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>,
          ffi.Pointer<ffi.Char>,
          ffi.Pointer<ffi.Char>,
          ffi.Pointer<Net>,
          imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetBytes(
    ffi.Pointer<ffi.Char> framework,
    VecUChar model,
    VecUChar config,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetBytes(
      framework,
      model,
      config,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetBytesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              VecUChar,
              VecUChar,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetBytes');
  late final _cv_dnn_Net_readNetBytes = _cv_dnn_Net_readNetBytesPtr.asFunction<
      ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, VecUChar, VecUChar,
          ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromCaffe(
    ffi.Pointer<ffi.Char> prototxt,
    ffi.Pointer<ffi.Char> caffeModel,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromCaffe(
      prototxt,
      caffeModel,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromCaffePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromCaffe');
  late final _cv_dnn_Net_readNetFromCaffe =
      _cv_dnn_Net_readNetFromCaffePtr.asFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromCaffeBytes(
    VecUChar prototxt,
    VecUChar caffeModel,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromCaffeBytes(
      prototxt,
      caffeModel,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromCaffeBytesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecUChar, VecUChar, ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromCaffeBytes');
  late final _cv_dnn_Net_readNetFromCaffeBytes =
      _cv_dnn_Net_readNetFromCaffeBytesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, VecUChar, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromONNX(
    ffi.Pointer<ffi.Char> model,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromONNX(
      model,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromONNXPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromONNX');
  late final _cv_dnn_Net_readNetFromONNX =
      _cv_dnn_Net_readNetFromONNXPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromONNXBytes(
    VecUChar model,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromONNXBytes(
      model,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromONNXBytesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecUChar, ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromONNXBytes');
  late final _cv_dnn_Net_readNetFromONNXBytes =
      _cv_dnn_Net_readNetFromONNXBytesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromTFLite(
    ffi.Pointer<ffi.Char> model,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromTFLite(
      model,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromTFLitePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromTFLite');
  late final _cv_dnn_Net_readNetFromTFLite =
      _cv_dnn_Net_readNetFromTFLitePtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromTFLiteBytes(
    VecUChar bufferModel,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromTFLiteBytes(
      bufferModel,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromTFLiteBytesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecUChar, ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromTFLiteBytes');
  late final _cv_dnn_Net_readNetFromTFLiteBytes =
      _cv_dnn_Net_readNetFromTFLiteBytesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromTensorflow(
    ffi.Pointer<ffi.Char> model,
    ffi.Pointer<ffi.Char> config,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromTensorflow(
      model,
      config,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromTensorflowPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromTensorflow');
  late final _cv_dnn_Net_readNetFromTensorflow =
      _cv_dnn_Net_readNetFromTensorflowPtr.asFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>,
              ffi.Pointer<ffi.Char>, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromTensorflowBytes(
    VecUChar model,
    VecUChar config,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromTensorflowBytes(
      model,
      config,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromTensorflowBytesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecUChar, VecUChar, ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromTensorflowBytes');
  late final _cv_dnn_Net_readNetFromTensorflowBytes =
      _cv_dnn_Net_readNetFromTensorflowBytesPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, VecUChar, ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_readNetFromTorch(
    ffi.Pointer<ffi.Char> model,
    bool isBinary,
    bool evaluate,
    ffi.Pointer<Net> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_readNetFromTorch(
      model,
      isBinary,
      evaluate,
      rval,
      callback,
    );
  }

  late final _cv_dnn_Net_readNetFromTorchPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              ffi.Bool,
              ffi.Bool,
              ffi.Pointer<Net>,
              imp1.CvCallback_0)>>('cv_dnn_Net_readNetFromTorch');
  late final _cv_dnn_Net_readNetFromTorch =
      _cv_dnn_Net_readNetFromTorchPtr.asFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, bool, bool,
              ffi.Pointer<Net>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_setInput(
    Net net,
    Mat blob,
    ffi.Pointer<ffi.Char> name,
    double scalefactor,
    Scalar mean,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_Net_setInput(
      net,
      blob,
      name,
      scalefactor,
      mean,
      callback,
    );
  }

  late final _cv_dnn_Net_setInputPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Net, Mat, ffi.Pointer<ffi.Char>,
              ffi.Double, Scalar, imp1.CvCallback_0)>>('cv_dnn_Net_setInput');
  late final _cv_dnn_Net_setInput = _cv_dnn_Net_setInputPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Net, Mat, ffi.Pointer<ffi.Char>, double,
          Scalar, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_setPreferableBackend(
    Net net,
    int backend,
  ) {
    return _cv_dnn_Net_setPreferableBackend(
      net,
      backend,
    );
  }

  late final _cv_dnn_Net_setPreferableBackendPtr =
      _lookup<ffi.NativeFunction<ffi.Pointer<CvStatus> Function(Net, ffi.Int)>>(
          'cv_dnn_Net_setPreferableBackend');
  late final _cv_dnn_Net_setPreferableBackend =
      _cv_dnn_Net_setPreferableBackendPtr
          .asFunction<ffi.Pointer<CvStatus> Function(Net, int)>();

  ffi.Pointer<CvStatus> cv_dnn_Net_setPreferableTarget(
    Net net,
    int target,
  ) {
    return _cv_dnn_Net_setPreferableTarget(
      net,
      target,
    );
  }

  late final _cv_dnn_Net_setPreferableTargetPtr =
      _lookup<ffi.NativeFunction<ffi.Pointer<CvStatus> Function(Net, ffi.Int)>>(
          'cv_dnn_Net_setPreferableTarget');
  late final _cv_dnn_Net_setPreferableTarget =
      _cv_dnn_Net_setPreferableTargetPtr
          .asFunction<ffi.Pointer<CvStatus> Function(Net, int)>();

  ffi.Pointer<CvStatus> cv_dnn_blobFromImage(
    Mat image,
    Mat blob,
    double scalefactor,
    CvSize size,
    Scalar mean,
    bool swapRB,
    bool crop,
    int ddepth,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_blobFromImage(
      image,
      blob,
      scalefactor,
      size,
      mean,
      swapRB,
      crop,
      ddepth,
      callback,
    );
  }

  late final _cv_dnn_blobFromImagePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat,
              Mat,
              ffi.Double,
              CvSize,
              Scalar,
              ffi.Bool,
              ffi.Bool,
              ffi.Int,
              imp1.CvCallback_0)>>('cv_dnn_blobFromImage');
  late final _cv_dnn_blobFromImage = _cv_dnn_blobFromImagePtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Mat, double, CvSize, Scalar, bool,
          bool, int, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_blobFromImages(
    VecMat images,
    Mat blob,
    double scalefactor,
    CvSize size,
    Scalar mean,
    bool swapRB,
    bool crop,
    int ddepth,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_blobFromImages(
      images,
      blob,
      scalefactor,
      size,
      mean,
      swapRB,
      crop,
      ddepth,
      callback,
    );
  }

  late final _cv_dnn_blobFromImagesPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecMat,
              Mat,
              ffi.Double,
              CvSize,
              Scalar,
              ffi.Bool,
              ffi.Bool,
              ffi.Int,
              imp1.CvCallback_0)>>('cv_dnn_blobFromImages');
  late final _cv_dnn_blobFromImages = _cv_dnn_blobFromImagesPtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecMat, Mat, double, CvSize, Scalar, bool,
          bool, int, imp1.CvCallback_0)>();

  void cv_dnn_enableModelDiagnostics(
    bool isDiagnosticsMode,
  ) {
    return _cv_dnn_enableModelDiagnostics(
      isDiagnosticsMode,
    );
  }

  late final _cv_dnn_enableModelDiagnosticsPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Bool)>>(
          'cv_dnn_enableModelDiagnostics');
  late final _cv_dnn_enableModelDiagnostics =
      _cv_dnn_enableModelDiagnosticsPtr.asFunction<void Function(bool)>();

  void cv_dnn_getAvailableBackends(
    ffi.Pointer<VecPoint> rval,
  ) {
    return _cv_dnn_getAvailableBackends(
      rval,
    );
  }

  late final _cv_dnn_getAvailableBackendsPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<VecPoint>)>>(
          'cv_dnn_getAvailableBackends');
  late final _cv_dnn_getAvailableBackends = _cv_dnn_getAvailableBackendsPtr
      .asFunction<void Function(ffi.Pointer<VecPoint>)>();

  ffi.Pointer<CvStatus> cv_dnn_getAvailableTargets(
    int be,
    ffi.Pointer<VecI32> rval,
  ) {
    return _cv_dnn_getAvailableTargets(
      be,
      rval,
    );
  }

  late final _cv_dnn_getAvailableTargetsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Int, ffi.Pointer<VecI32>)>>('cv_dnn_getAvailableTargets');
  late final _cv_dnn_getAvailableTargets = _cv_dnn_getAvailableTargetsPtr
      .asFunction<ffi.Pointer<CvStatus> Function(int, ffi.Pointer<VecI32>)>();

  ffi.Pointer<CvStatus> cv_dnn_getBlobChannel(
    Mat blob,
    int imgidx,
    int chnidx,
    ffi.Pointer<Mat> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_getBlobChannel(
      blob,
      imgidx,
      chnidx,
      rval,
      callback,
    );
  }

  late final _cv_dnn_getBlobChannelPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, ffi.Int, ffi.Int,
              ffi.Pointer<Mat>, imp1.CvCallback_0)>>('cv_dnn_getBlobChannel');
  late final _cv_dnn_getBlobChannel = _cv_dnn_getBlobChannelPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, int, int, ffi.Pointer<Mat>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_dnn_getBlobSize(
    Mat blob,
    ffi.Pointer<VecI32> rval,
  ) {
    return _cv_dnn_getBlobSize(
      blob,
      rval,
    );
  }

  late final _cv_dnn_getBlobSizePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, ffi.Pointer<VecI32>)>>('cv_dnn_getBlobSize');
  late final _cv_dnn_getBlobSize = _cv_dnn_getBlobSizePtr
      .asFunction<ffi.Pointer<CvStatus> Function(Mat, ffi.Pointer<VecI32>)>();

  ffi.Pointer<CvStatus> cv_dnn_imagesFromBlob(
    Mat blob,
    ffi.Pointer<VecMat> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_dnn_imagesFromBlob(
      blob,
      rval,
      callback,
    );
  }

  late final _cv_dnn_imagesFromBlobPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, ffi.Pointer<VecMat>,
              imp1.CvCallback_0)>>('cv_dnn_imagesFromBlob');
  late final _cv_dnn_imagesFromBlob = _cv_dnn_imagesFromBlobPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, ffi.Pointer<VecMat>, imp1.CvCallback_0)>();

  late final addresses = _SymbolAddresses(this);
}

class _SymbolAddresses {
  final CvNativeDnn _library;
  _SymbolAddresses(this._library);
  ffi.Pointer<ffi.NativeFunction<ffi.Void Function(AsyncArrayPtr)>>
      get cv_dnn_AsyncArray_close => _library._cv_dnn_AsyncArray_closePtr;
  ffi.Pointer<ffi.NativeFunction<ffi.Void Function(LayerPtr)>>
      get cv_dnn_Layer_close => _library._cv_dnn_Layer_closePtr;
  ffi.Pointer<ffi.NativeFunction<ffi.Void Function(NetPtr)>>
      get cv_dnn_Net_close => _library._cv_dnn_Net_closePtr;
}

final class AsyncArray extends ffi.Struct {
  external ffi.Pointer<ffi.Void> ptr;
}

typedef AsyncArrayPtr = ffi.Pointer<AsyncArray>;
typedef CvSize = imp1.CvSize;
typedef CvStatus = imp1.CvStatus;

final class Layer extends ffi.Struct {
  external ffi.Pointer<ffi.Void> ptr;
}

typedef LayerPtr = ffi.Pointer<Layer>;
typedef Mat = imp1.Mat;

final class Net extends ffi.Struct {
  external ffi.Pointer<ffi.Void> ptr;
}

typedef NetPtr = ffi.Pointer<Net>;
typedef Scalar = imp1.Scalar;
typedef VecF32 = imp1.VecF32;
typedef VecF64 = imp1.VecF64;
typedef VecI32 = imp1.VecI32;
typedef VecMat = imp1.VecMat;
typedef VecPoint = imp1.VecPoint;
typedef VecRect = imp1.VecRect;
typedef VecUChar = imp1.VecUChar;
typedef VecVecChar = imp1.VecVecChar;