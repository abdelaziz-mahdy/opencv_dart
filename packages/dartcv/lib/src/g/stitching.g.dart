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

/// Native bindings for OpenCV - Stitching
///
class CvNativeStitching {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  CvNativeStitching(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  CvNativeStitching.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  void cv_Stitcher_close(
    StitcherPtr stitcher,
  ) {
    return _cv_Stitcher_close(
      stitcher,
    );
  }

  late final _cv_Stitcher_closePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(StitcherPtr)>>(
          'cv_Stitcher_close');
  late final _cv_Stitcher_close =
      _cv_Stitcher_closePtr.asFunction<void Function(StitcherPtr)>();

  ffi.Pointer<CvStatus> cv_Stitcher_component(
    Stitcher self,
    ffi.Pointer<VecI32> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_component(
      self,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_componentPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Stitcher, ffi.Pointer<VecI32>,
              imp1.CvCallback_0)>>('cv_Stitcher_component');
  late final _cv_Stitcher_component = _cv_Stitcher_componentPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Stitcher, ffi.Pointer<VecI32>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_Stitcher_composePanorama(
    Stitcher self,
    Mat rpano,
    ffi.Pointer<ffi.Int> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_composePanorama(
      self,
      rpano,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_composePanoramaPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Stitcher, Mat, ffi.Pointer<ffi.Int>,
              imp1.CvCallback_0)>>('cv_Stitcher_composePanorama');
  late final _cv_Stitcher_composePanorama =
      _cv_Stitcher_composePanoramaPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Stitcher, Mat, ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_Stitcher_composePanorama_1(
    Stitcher self,
    VecMat mats,
    Mat rpano,
    ffi.Pointer<ffi.Int> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_composePanorama_1(
      self,
      mats,
      rpano,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_composePanorama_1Ptr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Stitcher,
              VecMat,
              Mat,
              ffi.Pointer<ffi.Int>,
              imp1.CvCallback_0)>>('cv_Stitcher_composePanorama_1');
  late final _cv_Stitcher_composePanorama_1 =
      _cv_Stitcher_composePanorama_1Ptr.asFunction<
          ffi.Pointer<CvStatus> Function(Stitcher, VecMat, Mat,
              ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_Stitcher_create(
    int mode,
    ffi.Pointer<Stitcher> rval,
  ) {
    return _cv_Stitcher_create(
      mode,
      rval,
    );
  }

  late final _cv_Stitcher_createPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Int, ffi.Pointer<Stitcher>)>>('cv_Stitcher_create');
  late final _cv_Stitcher_create = _cv_Stitcher_createPtr
      .asFunction<ffi.Pointer<CvStatus> Function(int, ffi.Pointer<Stitcher>)>();

  ffi.Pointer<CvStatus> cv_Stitcher_estimateTransform(
    Stitcher self,
    VecMat mats,
    VecMat masks,
    ffi.Pointer<ffi.Int> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_estimateTransform(
      self,
      mats,
      masks,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_estimateTransformPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Stitcher,
              VecMat,
              VecMat,
              ffi.Pointer<ffi.Int>,
              imp1.CvCallback_0)>>('cv_Stitcher_estimateTransform');
  late final _cv_Stitcher_estimateTransform =
      _cv_Stitcher_estimateTransformPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Stitcher, VecMat, VecMat,
              ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>();

  double cv_Stitcher_get_compositingResol(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_compositingResol(
      self,
    );
  }

  late final _cv_Stitcher_get_compositingResolPtr =
      _lookup<ffi.NativeFunction<ffi.Double Function(Stitcher)>>(
          'cv_Stitcher_get_compositingResol');
  late final _cv_Stitcher_get_compositingResol =
      _cv_Stitcher_get_compositingResolPtr
          .asFunction<double Function(Stitcher)>();

  int cv_Stitcher_get_interpolationFlags(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_interpolationFlags(
      self,
    );
  }

  late final _cv_Stitcher_get_interpolationFlagsPtr =
      _lookup<ffi.NativeFunction<ffi.Int Function(Stitcher)>>(
          'cv_Stitcher_get_interpolationFlags');
  late final _cv_Stitcher_get_interpolationFlags =
      _cv_Stitcher_get_interpolationFlagsPtr
          .asFunction<int Function(Stitcher)>();

  double cv_Stitcher_get_panoConfidenceThresh(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_panoConfidenceThresh(
      self,
    );
  }

  late final _cv_Stitcher_get_panoConfidenceThreshPtr =
      _lookup<ffi.NativeFunction<ffi.Double Function(Stitcher)>>(
          'cv_Stitcher_get_panoConfidenceThresh');
  late final _cv_Stitcher_get_panoConfidenceThresh =
      _cv_Stitcher_get_panoConfidenceThreshPtr
          .asFunction<double Function(Stitcher)>();

  double cv_Stitcher_get_registrationResol(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_registrationResol(
      self,
    );
  }

  late final _cv_Stitcher_get_registrationResolPtr =
      _lookup<ffi.NativeFunction<ffi.Double Function(Stitcher)>>(
          'cv_Stitcher_get_registrationResol');
  late final _cv_Stitcher_get_registrationResol =
      _cv_Stitcher_get_registrationResolPtr
          .asFunction<double Function(Stitcher)>();

  double cv_Stitcher_get_seamEstimationResol(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_seamEstimationResol(
      self,
    );
  }

  late final _cv_Stitcher_get_seamEstimationResolPtr =
      _lookup<ffi.NativeFunction<ffi.Double Function(Stitcher)>>(
          'cv_Stitcher_get_seamEstimationResol');
  late final _cv_Stitcher_get_seamEstimationResol =
      _cv_Stitcher_get_seamEstimationResolPtr
          .asFunction<double Function(Stitcher)>();

  int cv_Stitcher_get_waveCorrectKind(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_waveCorrectKind(
      self,
    );
  }

  late final _cv_Stitcher_get_waveCorrectKindPtr =
      _lookup<ffi.NativeFunction<ffi.Int Function(Stitcher)>>(
          'cv_Stitcher_get_waveCorrectKind');
  late final _cv_Stitcher_get_waveCorrectKind =
      _cv_Stitcher_get_waveCorrectKindPtr.asFunction<int Function(Stitcher)>();

  bool cv_Stitcher_get_waveCorrection(
    Stitcher self,
  ) {
    return _cv_Stitcher_get_waveCorrection(
      self,
    );
  }

  late final _cv_Stitcher_get_waveCorrectionPtr =
      _lookup<ffi.NativeFunction<ffi.Bool Function(Stitcher)>>(
          'cv_Stitcher_get_waveCorrection');
  late final _cv_Stitcher_get_waveCorrection =
      _cv_Stitcher_get_waveCorrectionPtr.asFunction<bool Function(Stitcher)>();

  void cv_Stitcher_set_compositingResol(
    Stitcher self,
    double val,
  ) {
    return _cv_Stitcher_set_compositingResol(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_compositingResolPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Double)>>(
          'cv_Stitcher_set_compositingResol');
  late final _cv_Stitcher_set_compositingResol =
      _cv_Stitcher_set_compositingResolPtr
          .asFunction<void Function(Stitcher, double)>();

  void cv_Stitcher_set_interpolationFlags(
    Stitcher self,
    int val,
  ) {
    return _cv_Stitcher_set_interpolationFlags(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_interpolationFlagsPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Int)>>(
          'cv_Stitcher_set_interpolationFlags');
  late final _cv_Stitcher_set_interpolationFlags =
      _cv_Stitcher_set_interpolationFlagsPtr
          .asFunction<void Function(Stitcher, int)>();

  void cv_Stitcher_set_panoConfidenceThresh(
    Stitcher self,
    double val,
  ) {
    return _cv_Stitcher_set_panoConfidenceThresh(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_panoConfidenceThreshPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Double)>>(
          'cv_Stitcher_set_panoConfidenceThresh');
  late final _cv_Stitcher_set_panoConfidenceThresh =
      _cv_Stitcher_set_panoConfidenceThreshPtr
          .asFunction<void Function(Stitcher, double)>();

  void cv_Stitcher_set_registrationResol(
    Stitcher self,
    double val,
  ) {
    return _cv_Stitcher_set_registrationResol(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_registrationResolPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Double)>>(
          'cv_Stitcher_set_registrationResol');
  late final _cv_Stitcher_set_registrationResol =
      _cv_Stitcher_set_registrationResolPtr
          .asFunction<void Function(Stitcher, double)>();

  void cv_Stitcher_set_seamEstimationResol(
    Stitcher self,
    double val,
  ) {
    return _cv_Stitcher_set_seamEstimationResol(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_seamEstimationResolPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Double)>>(
          'cv_Stitcher_set_seamEstimationResol');
  late final _cv_Stitcher_set_seamEstimationResol =
      _cv_Stitcher_set_seamEstimationResolPtr
          .asFunction<void Function(Stitcher, double)>();

  void cv_Stitcher_set_waveCorrectKind(
    Stitcher self,
    int val,
  ) {
    return _cv_Stitcher_set_waveCorrectKind(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_waveCorrectKindPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Int)>>(
          'cv_Stitcher_set_waveCorrectKind');
  late final _cv_Stitcher_set_waveCorrectKind =
      _cv_Stitcher_set_waveCorrectKindPtr
          .asFunction<void Function(Stitcher, int)>();

  void cv_Stitcher_set_waveCorrection(
    Stitcher self,
    bool val,
  ) {
    return _cv_Stitcher_set_waveCorrection(
      self,
      val,
    );
  }

  late final _cv_Stitcher_set_waveCorrectionPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(Stitcher, ffi.Bool)>>(
          'cv_Stitcher_set_waveCorrection');
  late final _cv_Stitcher_set_waveCorrection =
      _cv_Stitcher_set_waveCorrectionPtr
          .asFunction<void Function(Stitcher, bool)>();

  ffi.Pointer<CvStatus> cv_Stitcher_stitch(
    Stitcher self,
    VecMat mats,
    Mat rpano,
    ffi.Pointer<ffi.Int> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_stitch(
      self,
      mats,
      rpano,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_stitchPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Stitcher, VecMat, Mat,
              ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>>('cv_Stitcher_stitch');
  late final _cv_Stitcher_stitch = _cv_Stitcher_stitchPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Stitcher, VecMat, Mat, ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> cv_Stitcher_stitch_1(
    Stitcher self,
    VecMat mats,
    VecMat masks,
    Mat rpano,
    ffi.Pointer<ffi.Int> rval,
    imp1.CvCallback_0 callback,
  ) {
    return _cv_Stitcher_stitch_1(
      self,
      mats,
      masks,
      rpano,
      rval,
      callback,
    );
  }

  late final _cv_Stitcher_stitch_1Ptr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Stitcher,
              VecMat,
              VecMat,
              Mat,
              ffi.Pointer<ffi.Int>,
              imp1.CvCallback_0)>>('cv_Stitcher_stitch_1');
  late final _cv_Stitcher_stitch_1 = _cv_Stitcher_stitch_1Ptr.asFunction<
      ffi.Pointer<CvStatus> Function(Stitcher, VecMat, VecMat, Mat,
          ffi.Pointer<ffi.Int>, imp1.CvCallback_0)>();

  late final addresses = _SymbolAddresses(this);
}

class _SymbolAddresses {
  final CvNativeStitching _library;
  _SymbolAddresses(this._library);
  ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StitcherPtr)>>
      get cv_Stitcher_close => _library._cv_Stitcher_closePtr;
}

typedef CvStatus = imp1.CvStatus;
typedef Mat = imp1.Mat;

const int STITCHING_ERR_CAMERA_PARAMS_ADJUST_FAIL = 3;

const int STITCHING_ERR_HOMOGRAPHY_EST_FAIL = 2;

const int STITCHING_ERR_NEED_MORE_IMGS = 1;

const int STITCHING_OK = 0;

const int STITCHING_PANORAMA = 0;

const int STITCHING_SCANS = 1;

final class Stitcher extends ffi.Struct {
  external ffi.Pointer<ffi.Pointer<ffi.Void>> ptr;
}

typedef StitcherPtr = ffi.Pointer<Stitcher>;
typedef VecI32 = imp1.VecI32;
typedef VecMat = imp1.VecMat;