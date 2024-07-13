import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../g/types.g.dart' as cvg;
import 'base.dart';

abstract class Vec<N extends ffi.Struct, T> with IterableMixin<T> implements ffi.Finalizable {
  Vec.fromPointer(this.ptr);

  ffi.Pointer<N> ptr;
  N get ref;
  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  int get length;

  T operator [](int idx) => elementAt(idx);

  void dispose();
  Vec clone();
  ffi.Pointer<ffi.Void> asVoid();

  @override
  bool operator ==(Object other) {
    if (other is! Vec) return false;
    if (identical(this, other)) return true;
    if (length != other.length) return false;
    return hashCode == other.hashCode;
  }

  @override
  int get hashCode => ref.hashCode;
}

abstract class VecIterator<T> implements Iterator<T> {
  int currentIndex = -1;
  int get length;
  T operator [](int idx);

  @override
  T get current {
    if (currentIndex >= 0 && currentIndex < length) {
      return this[currentIndex];
    }
    throw IndexError.withLength(currentIndex, length);
  }

  @override
  bool moveNext() {
    if (currentIndex < length - 1) {
      currentIndex++;
      return true;
    }
    return false;
  }
}

class VecUChar extends Vec<cvg.VecUChar, int> {
  VecUChar.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecUChar([int length = 0, int value = 0]) => VecUChar.generate(length, (i) => value);
  factory VecUChar.fromList(List<int> pts) => VecUChar.generate(pts.length, (i) => pts[i]);

  factory VecUChar.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecUChar>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.UnsignedChar>(length);
    for (var i = 0; i < length; i++) {
      pp.ref.ptr[i] = generator(i);
    }
    return VecUChar.fromPointer(pp);
  }

  @override
  int get length => ref.length;

  /// Returns a view of native pointer
  Uint8List get data => ref.ptr.cast<ffi.Uint8>().asTypedList(length);

  /// ~~alias of data~~
  ///
  /// This method will return a full copy of [data]
  ///
  /// https://github.com/rainyl/opencv_dart/issues/85
  Uint8List toU8List() => Uint8List.fromList(data);

  @override
  VecUChar clone() => VecUChar.generate(length, (idx) => this[idx]);

  @override
  Iterator<int> get iterator => VecUCharIterator(ref);

  @override
  cvg.VecUChar get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecUCharIterator extends VecIterator<int> {
  VecUCharIterator(this.ref);
  cvg.VecUChar ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

typedef VecU8 = VecUChar;

class VecChar extends Vec<cvg.VecChar, int> {
  VecChar.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecChar([int length = 0, int value = 0]) => VecChar.generate(length, (i) => value);
  factory VecChar.fromList(List<int> pts) => VecChar.generate(pts.length, (i) => pts[i]);

  factory VecChar.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecChar>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Char>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecChar.fromPointer(pp);
  }

  @override
  int get length => ref.length;

  Uint8List get data => ref.ptr.cast<ffi.Uint8>().asTypedList(length);

  String asString() => utf8.decode(data);

  @override
  VecChar clone() => VecChar.generate(length, (idx) => this[idx]);

  @override
  Iterator<int> get iterator => VecCharIterator(ref);

  @override
  cvg.VecChar get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecCharIterator extends VecIterator<int> {
  VecCharIterator(this.ref);
  cvg.VecChar ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

class VecVecChar extends Vec<cvg.VecVecChar, VecChar> {
  VecVecChar.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecVecChar.fromList(List<List<int>> pts) =>
      VecVecChar.generate(pts.length, (i) => pts[i].i8, dispose: false);

  factory VecVecChar.generate(int length, VecChar Function(int i) generator, {bool dispose = true}) {
    final pp = calloc<cvg.VecVecChar>()..ref.length = length;
    pp.ref.ptr = calloc<cvg.VecChar>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v.ref;
      if (dispose) v.dispose();
    }
    return VecVecChar.fromPointer(pp);
  }

  List<String> asStringList() {
    return map(String.fromCharCodes).toList();
  }

  @override
  VecVecChar clone() => VecVecChar.generate(length, (idx) => this[idx], dispose: false);

  @override
  Iterator<VecChar> get iterator => VecVecCharIterator(ref);
  @override
  cvg.VecVecChar get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecVecCharIterator extends VecIterator<VecChar> {
  VecVecCharIterator(this.ref);
  cvg.VecVecChar ref;

  @override
  int get length => ref.length;

  /// return the reference
  @override
  VecChar operator [](int idx) => VecChar.fromPointer(ref.ptr + idx, false);
}

class VecI8 extends Vec<cvg.VecI8, int> {
  VecI8.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecI8([int length = 0, int value = 0]) => VecI8.generate(length, (i) => value);
  factory VecI8.fromList(List<int> pts) => VecI8.generate(pts.length, (i) => pts[i]);

  factory VecI8.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecI8>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Int8>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecI8.fromPointer(pp);
  }

  @override
  int get length => ref.length;

  Int8List get data => ref.ptr.cast<ffi.Int8>().asTypedList(length);

  String asString() => utf8.decode(data);

  @override
  VecI8 clone() => VecI8.generate(length, (idx) => this[idx]);

  @override
  Iterator<int> get iterator => VecI8Iterator(ref);

  @override
  cvg.VecI8 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecI8Iterator extends VecIterator<int> {
  VecI8Iterator(this.ref);
  cvg.VecI8 ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

class VecU16 extends Vec<cvg.VecU16, int> {
  VecU16.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecU16([int length = 0, int value = 0]) => VecU16.generate(length, (i) => value);
  factory VecU16.fromList(List<int> pts) => VecU16.generate(pts.length, (i) => pts[i]);

  factory VecU16.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecU16>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Uint16>(length);
    for (var i = 0; i < length; i++) {
      pp.ref.ptr[i] = generator(i);
    }
    return VecU16.fromPointer(pp);
  }

  @override
  VecU16 clone() => VecU16.generate(length, (idx) => this[idx]);

  @override
  int get length => ref.length;

  Uint16List get data => ref.ptr.cast<ffi.Uint16>().asTypedList(length);
  @override
  Iterator<int> get iterator => VecU16Iterator(ref);
  @override
  cvg.VecU16 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecU16Iterator extends VecIterator<int> {
  VecU16Iterator(this.ref);
  cvg.VecU16 ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

class VecI16 extends Vec<cvg.VecI16, int> {
  VecI16.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecI16([int length = 0, int value = 0]) => VecI16.generate(length, (i) => value);
  factory VecI16.fromList(List<int> pts) => VecI16.generate(pts.length, (i) => pts[i]);

  factory VecI16.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecI16>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Int16>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecI16.fromPointer(pp);
  }

  @override
  VecI16 clone() => VecI16.generate(length, (idx) => this[idx]);

  @override
  int get length => ref.length;

  Int16List get data => ref.ptr.cast<ffi.Int16>().asTypedList(length);
  @override
  Iterator<int> get iterator => VecI16Iterator(ref);
  @override
  cvg.VecI16 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecI16Iterator extends VecIterator<int> {
  VecI16Iterator(this.ref);
  cvg.VecI16 ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

class VecI32 extends Vec<cvg.VecI32, int> {
  VecI32.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecI32([int length = 0, int value = 0]) => VecI32.generate(length, (i) => value);
  factory VecI32.fromList(List<int> pts) => VecI32.generate(pts.length, (i) => pts[i]);

  factory VecI32.generate(int length, int Function(int i) generator) {
    final pp = calloc<cvg.VecI32>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Int32>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecI32.fromPointer(pp);
  }

  @override
  VecI32 clone() => VecI32.generate(length, (idx) => this[idx]);

  @override
  int get length => ref.length;

  Int32List get data => ref.ptr.cast<ffi.Int32>().asTypedList(length);

  @override
  Iterator<int> get iterator => VecI32Iterator(ref);

  @override
  cvg.VecI32 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecI32Iterator extends VecIterator<int> {
  VecI32Iterator(this.ref);
  cvg.VecI32 ref;

  @override
  int get length => ref.length;

  @override
  int operator [](int idx) => ref.ptr[idx];
}

class VecF32 extends Vec<cvg.VecF32, double> {
  VecF32.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecF32([int length = 0, double value = 0.0]) => VecF32.generate(length, (i) => value);
  factory VecF32.fromList(List<double> pts) => VecF32.generate(pts.length, (i) => pts[i]);

  factory VecF32.generate(int length, double Function(int i) generator) {
    final pp = calloc<cvg.VecF32>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Float>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecF32.fromPointer(pp);
  }

  @override
  VecF32 clone() => VecF32.generate(length, (idx) => this[idx]);

  @override
  int get length => ref.length;

  Float32List get data => ref.ptr.cast<ffi.Float>().asTypedList(length);
  @override
  Iterator<double> get iterator => VecF32Iterator(ref);
  @override
  cvg.VecF32 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecF32Iterator extends VecIterator<double> {
  VecF32Iterator(this.ref);
  cvg.VecF32 ref;

  @override
  int get length => ref.length;

  @override
  double operator [](int idx) => ref.ptr[idx];
}

class VecF64 extends Vec<cvg.VecF64, double> {
  VecF64.fromPointer(super.ptr, [bool attach = true]) : super.fromPointer() {
    if (attach) {
      Vec.finalizer.attach(this, ptr.cast<ffi.Void>(), detach: this);
      Vec.finalizer.attach(this, ptr.ref.ptr.cast<ffi.Void>(), detach: this);
    }
  }

  factory VecF64([int length = 0, double value = 0.0]) => VecF64.generate(length, (i) => value);
  factory VecF64.fromList(List<double> pts) => VecF64.generate(pts.length, (i) => pts[i]);

  factory VecF64.generate(int length, double Function(int i) generator) {
    final pp = calloc<cvg.VecF64>()..ref.length = length;
    pp.ref.ptr = calloc<ffi.Double>(length);
    for (var i = 0; i < length; i++) {
      final v = generator(i);
      pp.ref.ptr[i] = v;
    }
    return VecF64.fromPointer(pp);
  }

  @override
  VecF64 clone() => VecF64.generate(length, (idx) => this[idx]);

  @override
  int get length => ref.length;

  Float64List get data => ref.ptr.cast<ffi.Double>().asTypedList(length);

  @override
  Iterator<double> get iterator => VecF64Iterator(ref);

  @override
  cvg.VecF64 get ref => ptr.ref;

  @override
  void dispose() {
    Vec.finalizer.detach(this);
    calloc.free(ptr.ref.ptr);
    calloc.free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => ref.ptr.cast<ffi.Void>();
}

class VecF64Iterator extends VecIterator<double> {
  VecF64Iterator(this.ref);
  cvg.VecF64 ref;

  @override
  int get length => ref.length;

  @override
  double operator [](int idx) => ref.ptr[idx];
}

extension StringVecExtension on String {
  VecUChar get u8 {
    return cvRunArena<VecUChar>((arena) {
      final p = toNativeUtf8(allocator: arena);
      final pp = p.cast<ffi.UnsignedChar>();
      final v = VecUChar.fromList(List.generate(p.length, (idx) => pp[idx]));
      return v;
    });
  }

  VecChar get i8 {
    final p = toNativeUtf8();
    final v = VecChar.fromList(p.cast<ffi.Int8>().asTypedList(p.length));
    return v;
  }
}

extension ListUCharExtension on List<int> {
  VecUChar get u8 => VecUChar.fromList(this);
  VecChar get i8 => VecChar.fromList(this);
  VecU16 get u16 => VecU16.fromList(this);
  VecI16 get i16 => VecI16.fromList(this);
  VecI32 get i32 => VecI32.fromList(this);
}

extension ListListCharExtension on List<List<int>> {
  VecVecChar get i8 => VecVecChar.fromList(this);
}

extension ListFloatExtension on List<double> {
  VecF32 get f32 => VecF32.fromList(this);
  VecF64 get f64 => VecF64.fromList(this);
}

extension ListStringExtension on List<String> {
  VecVecChar get i8 => VecVecChar.fromList(map((e) => e.i8.toList()).toList());
}

// async completers
void vecI32Completer(Completer<VecI32> completer, VoidPtr p) =>
    completer.complete(VecI32.fromPointer(p.cast<cvg.VecI32>()));

void vecF32Completer(Completer<VecF32> completer, VoidPtr p) =>
    completer.complete(VecF32.fromPointer(p.cast<cvg.VecF32>()));

void vecF64Completer(Completer<VecF64> completer, VoidPtr p) =>
    completer.complete(VecF64.fromPointer(p.cast<cvg.VecF64>()));
