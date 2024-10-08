// Copyright (c) 2024, rainyl and all contributors. All rights reserved.
// Use of this source code is governed by a Apache-2.0 license
// that can be found in the LICENSE file.

library cv.contrib;

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

import '../core/base.dart';
import '../core/mat.dart';
import '../core/vec.dart';
import '../g/contrib.g.dart' as cvg;
import '../native_lib.dart' show ccontrib;
import 'wechat_qrcode.dart';

extension WeChatQRCodeAsync on WeChatQRCode {
  static Future<WeChatQRCode> emptyAsync() async => cvRunAsync<WeChatQRCode>(
        ccontrib.WeChatQRCode_New_Async,
        (c, p) => c.complete(WeChatQRCode.fromPointer(p.cast<cvg.WeChatQRCode>())),
      );

  static Future<WeChatQRCode> createAsync([
    String detectorPrototxtPath = "",
    String detectorCaffeModelPath = "",
    String superResolutionPrototxtPath = "",
    String superResolutionCaffeModelPath = "",
  ]) async {
    final arena = Arena();
    final dp = detectorPrototxtPath.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final dm = detectorCaffeModelPath.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final srp = superResolutionPrototxtPath.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final srm = superResolutionCaffeModelPath.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final rval = cvRunAsync<WeChatQRCode>(
      (callback) => ccontrib.WeChatQRCode_NewWithParams_Async(dp, dm, srp, srm, callback),
      (c, p) => c.complete(WeChatQRCode.fromPointer(p.cast<cvg.WeChatQRCode>())),
    );
    arena.releaseAll();
    return rval;
  }

  Future<(List<String>, VecMat)> detectAndDecodeAsync(InputArray img) async =>
      cvRunAsync2<(List<String>, VecMat)>(
          (callback) => ccontrib.WeChatQRCode_DetectAndDecode_Async(ptr, img.ref, callback), (c, p, p2) {
        final vec = VecVecChar.fromPointer(p.cast<cvg.VecVecChar>());
        final points = VecMat.fromPointer(p2.cast<cvg.VecMat>());
        final rval = vec.asStringList();
        vec.dispose();
        return c.complete((rval, points));
      });
}
