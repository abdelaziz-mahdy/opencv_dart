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

/// Native bindings for OpenCV - Imgcodecs
///
class CvNativeImgcodecs {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  CvNativeImgcodecs(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  CvNativeImgcodecs.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  ffi.Pointer<CvStatus> Image_IMDecode(
    VecUChar buf,
    int flags,
    ffi.Pointer<Mat> rval,
  ) {
    return _Image_IMDecode(
      buf,
      flags,
      rval,
    );
  }

  late final _Image_IMDecodePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, ffi.Int, ffi.Pointer<Mat>)>>('Image_IMDecode');
  late final _Image_IMDecode = _Image_IMDecodePtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecUChar, int, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> Image_IMDecode_Async(
    VecUChar buf,
    int flags,
    imp1.CvCallback_1 callback,
  ) {
    return _Image_IMDecode_Async(
      buf,
      flags,
      callback,
    );
  }

  late final _Image_IMDecode_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              VecUChar, ffi.Int, imp1.CvCallback_1)>>('Image_IMDecode_Async');
  late final _Image_IMDecode_Async = _Image_IMDecode_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(VecUChar, int, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> Image_IMEncode(
    ffi.Pointer<ffi.Char> fileExt,
    Mat img,
    ffi.Pointer<ffi.Bool> success,
    ffi.Pointer<VecUChar> rval,
  ) {
    return _Image_IMEncode(
      fileExt,
      img,
      success,
      rval,
    );
  }

  late final _Image_IMEncodePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat,
              ffi.Pointer<ffi.Bool>, ffi.Pointer<VecUChar>)>>('Image_IMEncode');
  late final _Image_IMEncode = _Image_IMEncodePtr.asFunction<
      ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat,
          ffi.Pointer<ffi.Bool>, ffi.Pointer<VecUChar>)>();

  ffi.Pointer<CvStatus> Image_IMEncode_Async(
    ffi.Pointer<ffi.Char> fileExt,
    Mat img,
    imp1.CvCallback_2 callback,
  ) {
    return _Image_IMEncode_Async(
      fileExt,
      img,
      callback,
    );
  }

  late final _Image_IMEncode_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat,
              imp1.CvCallback_2)>>('Image_IMEncode_Async');
  late final _Image_IMEncode_Async = _Image_IMEncode_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>, Mat, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> Image_IMEncode_WithParams(
    ffi.Pointer<ffi.Char> fileExt,
    Mat img,
    VecI32 params,
    ffi.Pointer<ffi.Bool> success,
    ffi.Pointer<VecUChar> rval,
  ) {
    return _Image_IMEncode_WithParams(
      fileExt,
      img,
      params,
      success,
      rval,
    );
  }

  late final _Image_IMEncode_WithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>,
              Mat,
              VecI32,
              ffi.Pointer<ffi.Bool>,
              ffi.Pointer<VecUChar>)>>('Image_IMEncode_WithParams');
  late final _Image_IMEncode_WithParams =
      _Image_IMEncode_WithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat, VecI32,
              ffi.Pointer<ffi.Bool>, ffi.Pointer<VecUChar>)>();

  ffi.Pointer<CvStatus> Image_IMEncode_WithParams_Async(
    ffi.Pointer<ffi.Char> fileExt,
    Mat img,
    VecI32 params,
    imp1.CvCallback_2 callback,
  ) {
    return _Image_IMEncode_WithParams_Async(
      fileExt,
      img,
      params,
      callback,
    );
  }

  late final _Image_IMEncode_WithParams_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat, VecI32,
              imp1.CvCallback_2)>>('Image_IMEncode_WithParams_Async');
  late final _Image_IMEncode_WithParams_Async =
      _Image_IMEncode_WithParams_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>, Mat, VecI32, imp1.CvCallback_2)>();

  ffi.Pointer<CvStatus> Image_IMRead(
    ffi.Pointer<ffi.Char> filename,
    int flags,
    ffi.Pointer<Mat> rval,
  ) {
    return _Image_IMRead(
      filename,
      flags,
      rval,
    );
  }

  late final _Image_IMReadPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, ffi.Int,
              ffi.Pointer<Mat>)>>('Image_IMRead');
  late final _Image_IMRead = _Image_IMReadPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>, int, ffi.Pointer<Mat>)>();

  ffi.Pointer<CvStatus> Image_IMRead_Async(
    ffi.Pointer<ffi.Char> filename,
    int flags,
    imp1.CvCallback_1 callback,
  ) {
    return _Image_IMRead_Async(
      filename,
      flags,
      callback,
    );
  }

  late final _Image_IMRead_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, ffi.Int,
              imp1.CvCallback_1)>>('Image_IMRead_Async');
  late final _Image_IMRead_Async = _Image_IMRead_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>, int, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> Image_IMWrite(
    ffi.Pointer<ffi.Char> filename,
    Mat img,
    ffi.Pointer<ffi.Bool> rval,
  ) {
    return _Image_IMWrite(
      filename,
      img,
      rval,
    );
  }

  late final _Image_IMWritePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat,
              ffi.Pointer<ffi.Bool>)>>('Image_IMWrite');
  late final _Image_IMWrite = _Image_IMWritePtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>, Mat, ffi.Pointer<ffi.Bool>)>();

  ffi.Pointer<CvStatus> Image_IMWrite_Async(
    ffi.Pointer<ffi.Char> filename,
    Mat img,
    imp1.CvCallback_1 callback,
  ) {
    return _Image_IMWrite_Async(
      filename,
      img,
      callback,
    );
  }

  late final _Image_IMWrite_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat,
              imp1.CvCallback_1)>>('Image_IMWrite_Async');
  late final _Image_IMWrite_Async = _Image_IMWrite_AsyncPtr.asFunction<
      ffi.Pointer<CvStatus> Function(
          ffi.Pointer<ffi.Char>, Mat, imp1.CvCallback_1)>();

  ffi.Pointer<CvStatus> Image_IMWrite_WithParams(
    ffi.Pointer<ffi.Char> filename,
    Mat img,
    VecI32 params,
    ffi.Pointer<ffi.Bool> rval,
  ) {
    return _Image_IMWrite_WithParams(
      filename,
      img,
      params,
      rval,
    );
  }

  late final _Image_IMWrite_WithParamsPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat, VecI32,
              ffi.Pointer<ffi.Bool>)>>('Image_IMWrite_WithParams');
  late final _Image_IMWrite_WithParams =
      _Image_IMWrite_WithParamsPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>, Mat, VecI32, ffi.Pointer<ffi.Bool>)>();

  ffi.Pointer<CvStatus> Image_IMWrite_WithParams_Async(
    ffi.Pointer<ffi.Char> filename,
    Mat img,
    VecI32 params,
    imp1.CvCallback_1 callback,
  ) {
    return _Image_IMWrite_WithParams_Async(
      filename,
      img,
      params,
      callback,
    );
  }

  late final _Image_IMWrite_WithParams_AsyncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<CvStatus> Function(ffi.Pointer<ffi.Char>, Mat, VecI32,
              imp1.CvCallback_1)>>('Image_IMWrite_WithParams_Async');
  late final _Image_IMWrite_WithParams_Async =
      _Image_IMWrite_WithParams_AsyncPtr.asFunction<
          ffi.Pointer<CvStatus> Function(
              ffi.Pointer<ffi.Char>, Mat, VecI32, imp1.CvCallback_1)>();
}

typedef CvStatus = imp1.CvStatus;
typedef Mat = imp1.Mat;
typedef VecI32 = imp1.VecI32;
typedef VecUChar = imp1.VecUChar;
