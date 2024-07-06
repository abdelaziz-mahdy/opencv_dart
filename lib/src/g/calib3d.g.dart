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
import 'package:opencv_dart/src/g/types.g.dart' as imp1;

/// Native bindings for OpenCV - Calib3d
///
class CvNativeCalib3d {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  CvNativeCalib3d(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  CvNativeCalib3d.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  ffi.Pointer<CvStatus> CalibrateCamera(
    VecVecPoint3f objectPoints,
    VecVecPoint2f imagePoints,
    Size imageSize,
    Mat cameraMatrix,
    Mat distCoeffs,
    Mat rvecs,
    Mat tvecs,
    int flag,
    TermCriteria criteria,
    ffi.Pointer<ffi.Double> rval,
  ) {
    return _CalibrateCamera(
      objectPoints,
      imagePoints,
      imageSize,
      cameraMatrix,
      distCoeffs,
      rvecs,
      tvecs,
      flag,
      criteria,
      rval,
    );
  }

  late final _CalibrateCameraPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecVecPoint3f,
              VecVecPoint2f,
              Size,
              Mat,
              Mat,
              Mat,
              Mat,
              ffi.Int,
              TermCriteria,
              ffi.Pointer<ffi.Double>)>>('CalibrateCamera');
  late final _CalibrateCamera = _CalibrateCameraPtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecVecPoint3f, VecVecPoint2f, Size, Mat,
          Mat, Mat, Mat, int, TermCriteria, ffi.Pointer<ffi.Double>)>();

  ffi.Pointer<CvStatus> DrawChessboardCorners(
    Mat image,
    Size patternSize,
    Mat corners,
    bool patternWasFound,
  ) {
    return _DrawChessboardCorners(
      image,
      patternSize,
      corners,
      patternWasFound,
    );
  }

  late final _DrawChessboardCornersPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Size, Mat, ffi.Bool)>>('DrawChessboardCorners');
  late final _DrawChessboardCorners = _DrawChessboardCornersPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Size, Mat, bool)>();

  ffi.Pointer<CvStatus> EstimateAffine2D(
    VecPoint2f from,
    VecPoint2f to,
    ffi.Pointer<Mat> rval,
  ) {
    return _EstimateAffine2D(
      from,
      to,
      rval,
    );
  }

  late final _EstimateAffine2DPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f, VecPoint2f, ffi.Pointer<Mat>)>>('EstimateAffine2D');
  late final _EstimateAffine2D = _EstimateAffine2DPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          VecPoint2f, VecPoint2f, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> EstimateAffine2DWithParams(
    VecPoint2f from,
    VecPoint2f to,
    Mat inliers,
    int method,
    double ransacReprojThreshold,
    int maxIters,
    double confidence,
    int refineIters,
    ffi.Pointer<Mat> rval,
  ) {
    return _EstimateAffine2DWithParams(
      from,
      to,
      inliers,
      method,
      ransacReprojThreshold,
      maxIters,
      confidence,
      refineIters,
      rval,
    );
  }

  late final _EstimateAffine2DWithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f,
              VecPoint2f,
              Mat,
              ffi.Int,
              ffi.Double,
              ffi.Size,
              ffi.Double,
              ffi.Size,
              ffi.Pointer<Mat>)>>('EstimateAffine2DWithParams');
  late final _EstimateAffine2DWithParams =
      _EstimateAffine2DWithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f, Mat, int,
              double, int, double, int, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> EstimateAffinePartial2D(
    VecPoint2f from,
    VecPoint2f to,
    ffi.Pointer<Mat> rval,
  ) {
    return _EstimateAffinePartial2D(
      from,
      to,
      rval,
    );
  }

  late final _EstimateAffinePartial2DPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f,
              ffi.Pointer<Mat>)>>('EstimateAffinePartial2D');
  late final _EstimateAffinePartial2D = _EstimateAffinePartial2DPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          VecPoint2f, VecPoint2f, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> EstimateAffinePartial2DWithParams(
    VecPoint2f from,
    VecPoint2f to,
    Mat inliers,
    int method,
    double ransacReprojThreshold,
    int maxIters,
    double confidence,
    int refineIters,
    ffi.Pointer<Mat> rval,
  ) {
    return _EstimateAffinePartial2DWithParams(
      from,
      to,
      inliers,
      method,
      ransacReprojThreshold,
      maxIters,
      confidence,
      refineIters,
      rval,
    );
  }

  late final _EstimateAffinePartial2DWithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f,
              VecPoint2f,
              Mat,
              ffi.Int,
              ffi.Double,
              ffi.Size,
              ffi.Double,
              ffi.Size,
              ffi.Pointer<Mat>)>>('EstimateAffinePartial2DWithParams');
  late final _EstimateAffinePartial2DWithParams =
      _EstimateAffinePartial2DWithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f, Mat, int,
              double, int, double, int, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> FindChessboardCorners(
    Mat image,
    Size patternSize,
    Mat corners,
    int flags,
    ffi.Pointer<ffi.Bool> rval,
  ) {
    return _FindChessboardCorners(
      image,
      patternSize,
      corners,
      flags,
      rval,
    );
  }

  late final _FindChessboardCornersPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, Mat, ffi.Int,
              ffi.Pointer<ffi.Bool>)>>('FindChessboardCorners');
  late final _FindChessboardCorners = _FindChessboardCornersPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, Size, Mat, int, ffi.Pointer<ffi.Bool>)>();

  ffi.Pointer<CvStatus> FindChessboardCornersSB(
    Mat image,
    Size patternSize,
    Mat corners,
    int flags,
    ffi.Pointer<ffi.Bool> rval,
  ) {
    return _FindChessboardCornersSB(
      image,
      patternSize,
      corners,
      flags,
      rval,
    );
  }

  late final _FindChessboardCornersSBPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, Mat, ffi.Int,
              ffi.Pointer<ffi.Bool>)>>('FindChessboardCornersSB');
  late final _FindChessboardCornersSB = _FindChessboardCornersSBPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, Size, Mat, int, ffi.Pointer<ffi.Bool>)>();

  ffi.Pointer<CvStatus> FindChessboardCornersSBWithMeta(
    Mat image,
    Size patternSize,
    Mat corners,
    int flags,
    Mat meta,
    ffi.Pointer<ffi.Bool> rval,
  ) {
    return _FindChessboardCornersSBWithMeta(
      image,
      patternSize,
      corners,
      flags,
      meta,
      rval,
    );
  }

  late final _FindChessboardCornersSBWithMetaPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, Mat, ffi.Int, Mat,
              ffi.Pointer<ffi.Bool>)>>('FindChessboardCornersSBWithMeta');
  late final _FindChessboardCornersSBWithMeta =
      _FindChessboardCornersSBWithMetaPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Size, Mat, int, Mat, ffi.Pointer<ffi.Bool>)>();

  ffi.Pointer<CvStatus> Fisheye_EstimateNewCameraMatrixForUndistortRectify(
    Mat k,
    Mat d,
    Size imgSize,
    Mat r,
    Mat p,
    double balance,
    Size newSize,
    double fovScale,
  ) {
    return _Fisheye_EstimateNewCameraMatrixForUndistortRectify(
      k,
      d,
      imgSize,
      r,
      p,
      balance,
      newSize,
      fovScale,
    );
  }

  late final _Fisheye_EstimateNewCameraMatrixForUndistortRectifyPtr = _lookup<
          ffi.NativeFunction<
              ffi.Pointer<CvStatus> Function(
                  Mat, Mat, Size, Mat, Mat, ffi.Double, Size, ffi.Double)>>(
      'Fisheye_EstimateNewCameraMatrixForUndistortRectify');
  late final _Fisheye_EstimateNewCameraMatrixForUndistortRectify =
      _Fisheye_EstimateNewCameraMatrixForUndistortRectifyPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Size, Mat, Mat, double, Size, double)>();

  ffi.Pointer<CvStatus> Fisheye_UndistortImage(
    Mat distorted,
    Mat undistorted,
    Mat k,
    Mat d,
  ) {
    return _Fisheye_UndistortImage(
      distorted,
      undistorted,
      k,
      d,
    );
  }

  late final _Fisheye_UndistortImagePtr = _lookup<
          ffi
          .NativeFunction<ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat)>>(
      'Fisheye_UndistortImage');
  late final _Fisheye_UndistortImage = _Fisheye_UndistortImagePtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat)>();

  ffi.Pointer<CvStatus> Fisheye_UndistortImageWithParams(
    Mat distorted,
    Mat undistorted,
    Mat k,
    Mat d,
    Mat knew,
    Size size,
  ) {
    return _Fisheye_UndistortImageWithParams(
      distorted,
      undistorted,
      k,
      d,
      knew,
      size,
    );
  }

  late final _Fisheye_UndistortImageWithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat,
              Size)>>('Fisheye_UndistortImageWithParams');
  late final _Fisheye_UndistortImageWithParams =
      _Fisheye_UndistortImageWithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat, Size)>();

  ffi.Pointer<CvStatus> Fisheye_UndistortPoints(
    Mat distorted,
    Mat undistorted,
    Mat k,
    Mat d,
    Mat R,
    Mat P,
  ) {
    return _Fisheye_UndistortPoints(
      distorted,
      undistorted,
      k,
      d,
      R,
      P,
    );
  }

  late final _Fisheye_UndistortPointsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Mat, Mat)>>('Fisheye_UndistortPoints');
  late final _Fisheye_UndistortPoints = _Fisheye_UndistortPointsPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat, Mat)>();

  ffi.Pointer<CvStatus> GetOptimalNewCameraMatrixWithParams(
    Mat cameraMatrix,
    Mat distCoeffs,
    Size size,
    double alpha,
    Size newImgSize,
    ffi.Pointer<Rect> validPixROI,
    bool centerPrincipalPoint,
    ffi.Pointer<Mat> rval,
  ) {
    return _GetOptimalNewCameraMatrixWithParams(
      cameraMatrix,
      distCoeffs,
      size,
      alpha,
      newImgSize,
      validPixROI,
      centerPrincipalPoint,
      rval,
    );
  }

  late final _GetOptimalNewCameraMatrixWithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat,
              Mat,
              Size,
              ffi.Double,
              Size,
              ffi.Pointer<Rect>,
              ffi.Bool,
              ffi.Pointer<Mat>)>>('GetOptimalNewCameraMatrixWithParams');
  late final _GetOptimalNewCameraMatrixWithParams =
      _GetOptimalNewCameraMatrixWithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Size, double, Size,
              ffi.Pointer<Rect>, bool, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> InitUndistortRectifyMap(
    Mat cameraMatrix,
    Mat distCoeffs,
    Mat r,
    Mat newCameraMatrix,
    Size size,
    int m1type,
    Mat map1,
    Mat map2,
  ) {
    return _InitUndistortRectifyMap(
      cameraMatrix,
      distCoeffs,
      r,
      newCameraMatrix,
      size,
      m1type,
      map1,
      map2,
    );
  }

  late final _InitUndistortRectifyMapPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Size, ffi.Int, Mat,
              Mat)>>('InitUndistortRectifyMap');
  late final _InitUndistortRectifyMap = _InitUndistortRectifyMapPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, Mat, Mat, Mat, Size, int, Mat, Mat)>();

  ffi.Pointer<CvStatus> Undistort(
    Mat src,
    Mat dst,
    Mat cameraMatrix,
    Mat distCoeffs,
    Mat newCameraMatrix,
  ) {
    return _Undistort(
      src,
      dst,
      cameraMatrix,
      distCoeffs,
      newCameraMatrix,
    );
  }

  late final _UndistortPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Mat)>>('Undistort');
  late final _Undistort = _UndistortPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat)>();

  ffi.Pointer<CvStatus> UndistortPoints(
    Mat distorted,
    Mat undistorted,
    Mat k,
    Mat d,
    Mat r,
    Mat p,
    TermCriteria criteria,
  ) {
    return _UndistortPoints(
      distorted,
      undistorted,
      k,
      d,
      r,
      p,
      criteria,
    );
  }

  late final _UndistortPointsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Mat, Mat, TermCriteria)>>('UndistortPoints');
  late final _UndistortPoints = _UndistortPointsPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, Mat, Mat, Mat, Mat, Mat, TermCriteria)>();

  ffi.Pointer<CvStatus> calibrateCamera_Async(
    VecVecPoint3f objectPoints,
    VecVecPoint2f imagePoints,
    Size imageSize,
    Mat cameraMatrix,
    Mat distCoeffs,
    int flag,
    TermCriteria criteria,
    imp1.CvCallback_5 callback,
  ) {
    return _calibrateCamera_Async(
      objectPoints,
      imagePoints,
      imageSize,
      cameraMatrix,
      distCoeffs,
      flag,
      criteria,
      callback,
    );
  }

  late final _calibrateCamera_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecVecPoint3f,
              VecVecPoint2f,
              Size,
              Mat,
              Mat,
              ffi.Int,
              TermCriteria,
              imp1.CvCallback_5)>>('calibrateCamera_Async');
  late final _calibrateCamera_Async = _calibrateCamera_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecVecPoint3f, VecVecPoint2f, Size, Mat,
          Mat, int, TermCriteria, imp1.CvCallback_5)>();

  ffi.Pointer<CvStatus> drawChessboardCorners_Async(
    Mat image,
    Size patternSize,
    bool patternWasFound,
    imp1.CvCallback_0 callback,
  ) {
    return _drawChessboardCorners_Async(
      image,
      patternSize,
      patternWasFound,
      callback,
    );
  }

  late final _drawChessboardCorners_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, ffi.Bool,
              imp1.CvCallback_0)>>('drawChessboardCorners_Async');
  late final _drawChessboardCorners_Async =
      _drawChessboardCorners_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, bool, imp1.CvCallback_0)>();

  ffi.Pointer<CvStatus> estimateAffine2DWithParams_Async(
    VecPoint2f from,
    VecPoint2f to,
    int method,
    double ransacReprojThreshold,
    int maxIters,
    double confidence,
    int refineIters,
    imp1.CvCallback_2 callback,
  ) {
    return _estimateAffine2DWithParams_Async(
      from,
      to,
      method,
      ransacReprojThreshold,
      maxIters,
      confidence,
      refineIters,
      callback,
    );
  }

  late final _estimateAffine2DWithParams_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f,
              VecPoint2f,
              ffi.Int,
              ffi.Double,
              ffi.Size,
              ffi.Double,
              ffi.Size,
              imp1.CvCallback_2)>>('estimateAffine2DWithParams_Async');
  late final _estimateAffine2DWithParams_Async =
      _estimateAffine2DWithParams_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f, int, double,
              int, double, int, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> estimateAffine2D_Async(
    VecPoint2f from,
    VecPoint2f to,
    imp1.CvCallback_1 callback,
  ) {
    return _estimateAffine2D_Async(
      from,
      to,
      callback,
    );
  }

  late final _estimateAffine2D_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f,
              imp1.CvCallback_1)>>('estimateAffine2D_Async');
  late final _estimateAffine2D_Async = _estimateAffine2D_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          VecPoint2f, VecPoint2f, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> estimateAffinePartial2DWithParams_Async(
    VecPoint2f from,
    VecPoint2f to,
    int method,
    double ransacReprojThreshold,
    int maxIters,
    double confidence,
    int refineIters,
    imp1.CvCallback_2 callback,
  ) {
    return _estimateAffinePartial2DWithParams_Async(
      from,
      to,
      method,
      ransacReprojThreshold,
      maxIters,
      confidence,
      refineIters,
      callback,
    );
  }

  late final _estimateAffinePartial2DWithParams_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f,
              VecPoint2f,
              ffi.Int,
              ffi.Double,
              ffi.Size,
              ffi.Double,
              ffi.Size,
              imp1.CvCallback_2)>>('estimateAffinePartial2DWithParams_Async');
  late final _estimateAffinePartial2DWithParams_Async =
      _estimateAffinePartial2DWithParams_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f, int, double,
              int, double, int, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> estimateAffinePartial2D_Async(
    VecPoint2f from,
    VecPoint2f to,
    imp1.CvCallback_1 callback,
  ) {
    return _estimateAffinePartial2D_Async(
      from,
      to,
      callback,
    );
  }

  late final _estimateAffinePartial2D_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(VecPoint2f, VecPoint2f,
              imp1.CvCallback_1)>>('estimateAffinePartial2D_Async');
  late final _estimateAffinePartial2D_Async =
      _estimateAffinePartial2D_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              VecPoint2f, VecPoint2f, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> findChessboardCornersSBWithMeta_Async(
    Mat image,
    Size patternSize,
    int flags,
    imp1.CvCallback_3 callback,
  ) {
    return _findChessboardCornersSBWithMeta_Async(
      image,
      patternSize,
      flags,
      callback,
    );
  }

  late final _findChessboardCornersSBWithMeta_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, ffi.Int,
              imp1.CvCallback_3)>>('findChessboardCornersSBWithMeta_Async');
  late final _findChessboardCornersSBWithMeta_Async =
      _findChessboardCornersSBWithMeta_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, int, imp1.CvCallback_3)>();

  ffi.Pointer<CvStatus> findChessboardCornersSB_Async(
    Mat image,
    Size patternSize,
    int flags,
    imp1.CvCallback_2 callback,
  ) {
    return _findChessboardCornersSB_Async(
      image,
      patternSize,
      flags,
      callback,
    );
  }

  late final _findChessboardCornersSB_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, ffi.Int,
              imp1.CvCallback_2)>>('findChessboardCornersSB_Async');
  late final _findChessboardCornersSB_Async =
      _findChessboardCornersSB_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, int, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> findChessboardCorners_Async(
    Mat image,
    Size patternSize,
    int flags,
    imp1.CvCallback_2 callback,
  ) {
    return _findChessboardCorners_Async(
      image,
      patternSize,
      flags,
      callback,
    );
  }

  late final _findChessboardCorners_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, ffi.Int,
              imp1.CvCallback_2)>>('findChessboardCorners_Async');
  late final _findChessboardCorners_Async =
      _findChessboardCorners_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Size, int, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus>
      fisheye_estimateNewCameraMatrixForUndistortRectify_Async(
    Mat k,
    Mat d,
    Size imgSize,
    Mat r,
    double balance,
    Size newSize,
    double fovScale,
    imp1.CvCallback_1 p,
  ) {
    return _fisheye_estimateNewCameraMatrixForUndistortRectify_Async(
      k,
      d,
      imgSize,
      r,
      balance,
      newSize,
      fovScale,
      p,
    );
  }

  late final _fisheye_estimateNewCameraMatrixForUndistortRectify_AsyncPtr =
      _lookup<
              ffi.NativeFunction<
                  ffi.Pointer<CvStatus> Function(Mat, Mat, Size, Mat,
                      ffi.Double, Size, ffi.Double, imp1.CvCallback_1)>>(
          'fisheye_estimateNewCameraMatrixForUndistortRectify_Async');
  late final _fisheye_estimateNewCameraMatrixForUndistortRectify_Async =
      _fisheye_estimateNewCameraMatrixForUndistortRectify_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Size, Mat, double, Size, double, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> fisheye_undistortImageWithParams_Async(
    Mat distorted,
    Mat k,
    Mat d,
    Mat knew,
    Size size,
    imp1.CvCallback_1 callback,
  ) {
    return _fisheye_undistortImageWithParams_Async(
      distorted,
      k,
      d,
      knew,
      size,
      callback,
    );
  }

  late final _fisheye_undistortImageWithParams_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Size,
              imp1.CvCallback_1)>>('fisheye_undistortImageWithParams_Async');
  late final _fisheye_undistortImageWithParams_Async =
      _fisheye_undistortImageWithParams_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Size, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> fisheye_undistortImage_Async(
    Mat distorted,
    Mat k,
    Mat d,
    imp1.CvCallback_1 callback,
  ) {
    return _fisheye_undistortImage_Async(
      distorted,
      k,
      d,
      callback,
    );
  }

  late final _fisheye_undistortImage_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat,
              imp1.CvCallback_1)>>('fisheye_undistortImage_Async');
  late final _fisheye_undistortImage_Async =
      _fisheye_undistortImage_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> fisheye_undistortPoints_Async(
    Mat distorted,
    Mat k,
    Mat d,
    Mat R,
    Mat P,
    imp1.CvCallback_1 callback,
  ) {
    return _fisheye_undistortPoints_Async(
      distorted,
      k,
      d,
      R,
      P,
      callback,
    );
  }

  late final _fisheye_undistortPoints_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat,
              imp1.CvCallback_1)>>('fisheye_undistortPoints_Async');
  late final _fisheye_undistortPoints_Async =
      _fisheye_undistortPoints_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Mat, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> getOptimalNewCameraMatrix_Async(
    Mat cameraMatrix,
    Mat distCoeffs,
    Size size,
    double alpha,
    Size newImgSize,
    bool centerPrincipalPoint,
    imp1.CvCallback_2 callback,
  ) {
    return _getOptimalNewCameraMatrix_Async(
      cameraMatrix,
      distCoeffs,
      size,
      alpha,
      newImgSize,
      centerPrincipalPoint,
      callback,
    );
  }

  late final _getOptimalNewCameraMatrix_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Size, ffi.Double, Size,
              ffi.Bool, imp1.CvCallback_2)>>('getOptimalNewCameraMatrix_Async');
  late final _getOptimalNewCameraMatrix_Async =
      _getOptimalNewCameraMatrix_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Size, double, Size, bool, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> initUndistortRectifyMap_Async(
    Mat cameraMatrix,
    Mat distCoeffs,
    Mat r,
    Mat newCameraMatrix,
    Size size,
    int m1type,
    imp1.CvCallback_2 callback,
  ) {
    return _initUndistortRectifyMap_Async(
      cameraMatrix,
      distCoeffs,
      r,
      newCameraMatrix,
      size,
      m1type,
      callback,
    );
  }

  late final _initUndistortRectifyMap_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Size, ffi.Int,
              imp1.CvCallback_2)>>('initUndistortRectifyMap_Async');
  late final _initUndistortRectifyMap_Async =
      _initUndistortRectifyMap_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, Size, int, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> undistortPoints_Async(
    Mat distorted,
    Mat k,
    Mat d,
    Mat r,
    Mat p,
    TermCriteria criteria,
    imp1.CvCallback_1 callback,
  ) {
    return _undistortPoints_Async(
      distorted,
      k,
      d,
      r,
      p,
      criteria,
      callback,
    );
  }

  late final _undistortPoints_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, Mat, TermCriteria,
              imp1.CvCallback_1)>>('undistortPoints_Async');
  late final _undistortPoints_Async = _undistortPoints_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          Mat, Mat, Mat, Mat, Mat, TermCriteria, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> undistort_Async(
    Mat src,
    Mat cameraMatrix,
    Mat distCoeffs,
    Mat newCameraMatrix,
    imp1.CvCallback_1 callback,
  ) {
    return _undistort_Async(
      src,
      cameraMatrix,
      distCoeffs,
      newCameraMatrix,
      callback,
    );
  }

  late final _undistort_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              Mat, Mat, Mat, Mat, imp1.CvCallback_1)>>('undistort_Async');
  late final _undistort_Async = _undistort_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(Mat, Mat, Mat, Mat, imp1.CvCallback_1)>();
}

typedef CvStatus = imp1.CvStatus;
typedef Mat = imp1.Mat;
typedef Rect = imp1.Rect;
typedef Size = imp1.Size;
typedef TermCriteria = imp1.TermCriteria;
typedef VecPoint2f = imp1.VecPoint2f;
typedef VecVecPoint2f = imp1.VecVecPoint2f;
typedef VecVecPoint3f = imp1.VecVecPoint3f;
