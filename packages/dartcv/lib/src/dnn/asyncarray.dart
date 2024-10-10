// Copyright (c) 2024, rainyl and all contributors. All rights reserved.
// Use of this source code is governed by a Apache-2.0 license
// that can be found in the LICENSE file.

library cv.dnn;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../core/base.dart';
import '../core/mat.dart';
import '../g/dnn.g.dart' as cvg;
import '../native_lib.dart' show cdnn;

class AsyncArray extends CvStruct<cvg.AsyncArray> {
  AsyncArray._(cvg.AsyncArrayPtr ptr, [bool attach = true]) : super.fromPointer(ptr) {
    if (attach) {
      finalizer.attach(this, ptr.cast(), detach: this);
    }
  }

  factory AsyncArray.fromPointer(cvg.AsyncArrayPtr ptr, [bool attach = true]) => AsyncArray._(ptr, attach);

  factory AsyncArray.empty() {
    final p = calloc<cvg.AsyncArray>();
    cvRun(() => cdnn.AsyncArray_New(p));
    final arr = AsyncArray._(p);
    return arr;
  }

  static final finalizer = OcvFinalizer<cvg.AsyncArrayPtr>(cdnn.addresses.AsyncArray_Close);

  void dispose() {
    finalizer.detach(this);
    cdnn.AsyncArray_Close(ptr);
  }

  Mat get() {
    final dst = Mat.empty();
    cvRun(() => cdnn.AsyncArray_Get(ref, dst.ref));
    return dst;
  }

  @override
  cvg.AsyncArray get ref => ptr.ref;
}
