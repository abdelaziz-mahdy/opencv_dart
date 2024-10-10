// Copyright (c) 2024, rainyl and all contributors. All rights reserved.
// Use of this source code is governed by a Apache-2.0 license
// that can be found in the LICENSE file.

library cv.imgproc.clahe;

import '../core/base.dart';
import '../core/mat.dart';
import '../core/size.dart';
import '../g/imgproc.g.dart' as cvg;
import '../native_lib.dart' show cimgproc;
import 'clahe.dart';

extension CLAHEAsync on CLAHE {
  static Future<CLAHE> createAsync([
    double clipLimit = 40,
    (int width, int height) tileGridSize = (8, 8),
  ]) async =>
      cvRunAsync(
        (callback) => cimgproc.CLAHE_CreateWithParams_Async(clipLimit, tileGridSize.toSize().ref, callback),
        (c, p) => c.complete(CLAHE.fromPointer(p.cast<cvg.CLAHE>())),
      );

  Future<Mat> applyAsync(Mat src) async =>
      cvRunAsync((callback) => cimgproc.CLAHE_Apply_Async(ref, src.ref, callback), matCompleter);
}
