import 'package:dartcv4/dartcv.dart' as cv;
import 'package:test/test.dart';

import 'vec_matcher.dart';

void main() {
  test('VecI32', () {
    final points = List.generate(100, (index) => index);
    final vec = points.i32;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.dispose();

    final vec2 = cv.VecI32(100, 10);
    expect(vec2.toList(), List.generate(100, (index) => 10));
    vec2.add(2541);
    expect(vec2.length, 101);
    expect(vec2[vec2.length - 1], 2541);

    vec2.reserve(21);
    vec2.resize(21);
    expect(vec2.length, 21);
    expect(vec2.size(), 21);

    final vec3 = cv.VecI32.generate(vec2.length, (i) => vec2[i]);
    vec2.extend(vec3);
    expect(vec2.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);
  });

  test('VecI16', () {
    final points = List.generate(100, (index) => index);
    final vec = points.i16;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.dispose();

    final vec2 = cv.VecI16(100, 10);
    expect(vec2.toList(), List.generate(100, (index) => 10));
    vec2.add(2541);
    expect(vec2.length, 101);
    expect(vec2[vec2.length - 1], 2541);

    vec2.reserve(21);
    vec2.resize(21);
    expect(vec2.length, 21);
    expect(vec2.size(), 21);

    final vec3 = cv.VecI16.generate(vec2.length, (i) => vec2[i]);
    vec2.extend(vec3);
    expect(vec2.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);
  });

  test('VecU16', () {
    final points = List.generate(100, (index) => index);
    final vec = points.u16;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.dispose();

    final vec2 = cv.VecU16(100, 10);
    expect(vec2.toList(), List.generate(100, (index) => 10));
    vec2.add(2541);
    expect(vec2.length, 101);
    expect(vec2[vec2.length - 1], 2541);

    vec2.reserve(21);
    vec2.resize(21);
    expect(vec2.length, 21);
    expect(vec2.size(), 21);

    final vec3 = cv.VecU16.generate(vec2.length, (i) => vec2[i]);
    vec2.extend(vec3);
    expect(vec2.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);
  });

  test('VecUChar', () {
    final points = List.generate(1000, (index) => index % 256);
    final vec = points.vecUChar;
    final u8List = vec.toU8List();
    expect(u8List.indexed.map((e) => e.$2 == points[e.$1]).every((e) => e), true);

    final u8 = vec.data;
    expect(u8.indexed.map((e) => e.$2 == points[e.$1]).every((e) => e), true);

    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));
    vec.dispose();

    vec1.add(254);
    expect(vec1.length, points.length + 1);
    expect(vec1[vec1.length - 1], 254);

    final vec3 = cv.VecUChar.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, vec3.length * 2);

    vec3.reserve(21);
    vec3.resize(21);
    expect(vec3.length, 21);
    expect(vec3.size(), 21);
    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);
  });

  test('VecChar', () {
    final points = List.generate(100, (index) => index);
    final vec = points.vecChar;

    final data = vec.data;
    expect(data.indexed.map((e) => e.$2 == points[e.$1]).every((e) => e), true);

    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    const hellos = [
      72, 101, 108, 108, 111, // Hello
      228, 189, 160, 229, 165, 189, // 你好
      236, 149, 136, 235, 133, 149, 237, 149, 152, 236, 132, 184, 236, 154, 148, // 안녕하세요
      208, 159, 209, 128, 208, 184, 208, 178, 208, 181, 209, 130, // Привет
      227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, // こんにちは
    ];

    final vec2 = cv.VecChar.fromList(hellos);
    expect(vec2.asString(), "Hello你好안녕하세요Приветこんにちは");

    vec1.add(121);
    expect(vec1.length, points.length + 1);
    expect(vec1[vec1.length - 1], 121);

    final vec3 = cv.VecChar.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, vec3.length * 2);

    vec3.reserve(21);
    vec3.resize(21);
    expect(vec3.length, 21);
    expect(vec3.size(), 21);
    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecVecChar', () {
    final points = List.generate(100, (index) => List.generate(100, (index) => index));
    final vec = points.vecVecChar;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    final vec1 = vec.clone();
    vec.dispose();
    expect(vec1.size(), points.length);
    expect(vec1.first, points.first);
    expect(vec1.last, points.last);

    expect(vec1[0], points[0]);

    vec1.dispose();
  });

  test('VecF32', () {
    final points = List.generate(100, (index) => index.toDouble());
    final vec = points.f32;

    final vec_ = List.generate(100, (index) => index).f32;
    expect(vec_ == vec, false);

    final data = vec.data;
    expect(data.indexed.map((e) => e.$2 == points[e.$1]).every((e) => e), true);

    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(2541);
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], 2541);

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecF32.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecF64', () {
    final points = List.generate(100, (index) => index.toDouble());
    final vec = points.f64;

    final data = vec.data;
    expect(data.indexed.map((e) => e.$2 == points[e.$1]).every((e) => e), true);

    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(2541);
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], 2541);

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecF64.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecF16', () {
    final points = List.generate(100, (index) => index.toDouble());
    final vec = points.f16;

    final data = vec.dataFp16;
    expect(data.indexed.every((e) => e.$2 - points[e.$1] < 1e-8), true);

    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    vec[24] = 1;
    expect(vec[24], 1);

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(241);
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], 241);

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecF16.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecRect', () {
    final points = List.generate(100, (index) => cv.Rect(index, index, index + 10, index + 20));
    final vec = points.cvd;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);
    expect(vec.first.toString(), "Rect(0, 0, 10, 20)");

    // get the reference
    final rect = vec[1]; // cv.Rect(1, 1, 11, 21)
    expect(rect, cv.Rect(1, 1, 11, 21));
    // change the reference will affect the original value
    rect.x = 100;
    expect(rect, cv.Rect(100, 1, 11, 21));
    // change the value
    vec[1] = cv.Rect(100, 100, 11, 21);
    expect(vec[1], cv.Rect(100, 100, 11, 21));

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(cv.Rect(0, 0, 100, 100));
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], cv.Rect(0, 0, 100, 100));

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecRect.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecRect2f', () {
    final points = List.generate(
      100,
      (index) => cv.Rect2f(index.toDouble(), index.toDouble(), index + 10, index + 20),
    );
    final vec = points.asVec();
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);
    expect(vec.first.toString(), "Rect2f(0.000, 0.000, 10.000, 20.000)");

    // get the reference
    final rect = vec[1]; // cv.Rect2f(1, 1, 11, 21)
    expect(rect, cv.Rect2f(1, 1, 11, 21));
    // change the reference will affect the original value
    rect.x = 100;
    expect(rect, cv.Rect2f(100, 1, 11, 21));
    // change the value
    vec[1] = cv.Rect2f(100, 100, 11, 21);
    expect(vec[1], cv.Rect2f(100, 100, 11, 21));

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(cv.Rect2f(0, 0, 100, 100));
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], cv.Rect2f(0, 0, 100, 100));

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecRect2f.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('RotatedRect', () {
    final rect = cv.RotatedRect(cv.Point2f(1, 1), (10, 10), 60);
    expect(rect.points.length, greaterThan(0));
    expect(rect.boundingRect, cv.Rect(-6, -6, 15, 15));
    expect(rect.center, cv.Point2f(1, 1));
    expect(rect.size, cv.Size2f(10, 10));
    expect(rect.angle, 60);
    expect(rect.toString(), 'RotatedRect(Point2f(1.000, 1.000), Size2f(10.000, 10.000), 60.000)');

    final rect2 = cv.RotatedRect.fromNative(rect.ref);
    expect(rect2, rect);

    rect2.dispose();
  });
  test('VecDMatch', () {
    final points = List.generate(100, (index) => cv.DMatch(index, index, index, index.toDouble()));
    final vec = points.cvd;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);
    expect(vec.first.toString(), "DMatch(0, 0, 0, 0.000)");

    // get the reference
    final dm = vec[1]; // cv.Rect(1, 1, 11, 21)
    expect(dm, cv.DMatch(1, 1, 1, 1.0));
    // change the reference will affect the original value
    dm.queryIdx = 100;
    dm.trainIdx = 2;
    dm.imgIdx = 3;
    dm.distance = 241.0;
    expect(dm, cv.DMatch(100, 2, 3, 241.0));
    // change the value
    vec[1] = cv.DMatch(100, 100, 11, 21.0);
    expect(vec[1], cv.DMatch(100, 100, 11, 21.0));

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(cv.DMatch(0, 0, 100, 100));
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], cv.DMatch(0, 0, 100, 100));

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecDMatch.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });

  test('VecVecDMatch', () {
    final points = List.generate(
      10,
      (index) => List.generate(10, (index) => cv.DMatch(index, index, index, index.toDouble())),
    );
    final vec = points.cvd;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);

    final vec1 = vec.clone();
    expect(vec1.length, vec.length);
    expect(vec1.first, vecElementEquals(vec.first));
    expect(vec1.last, vecElementEquals(vec.last));

    final list = vec1.copyToList();
    expect(list[0][0], vec1[0][0]);

    vec1.dispose();
  });

  test('VecKeyPoint', () {
    final points = List.generate(
      100,
      (index) => cv.KeyPoint(
        index.toDouble(),
        index.toDouble(),
        index.toDouble(),
        index.toDouble(),
        index.toDouble(),
        index,
        index,
      ),
    );
    final vec = points.cvd;
    expect(vec.length, points.length);
    expect(vec.first, points.first);
    expect(vec.last, points.last);
    expect(vec.first.toString(), "KeyPoint(0.000, 0.000, 0.000, 0.000, 0.000, 0, 0)");

    // get the reference
    final kp = vec[1];
    expect(kp, cv.KeyPoint(1.000, 1.000, 1.000, 1.000, 1.000, 1, 1));
    // change the reference will affect the original value
    kp.x = 100.0;
    kp.y = 100.0;
    kp.size = 100.0;
    kp.angle = 100.0;
    kp.response = 100.0;
    kp.octave = 100;
    kp.classID = 100;
    expect(kp, cv.KeyPoint(100.0, 100.0, 100.0, 100.0, 100.0, 100, 100));
    // change the value
    vec[1] = cv.KeyPoint(5.000, 2.000, 5.000, 4.000, 1.000, 0, 0);
    expect(vec[1], cv.KeyPoint(5.000, 2.000, 5.000, 4.000, 1.000, 0, 0));

    final vec1 = vec.clone();
    expect(vec1, vecElementEquals(vec));

    vec1.add(cv.KeyPoint(5.000, 2.000, 5.000, 4.000, 1.000, 0, 0));
    expect(vec1.length, 101);
    expect(vec1[vec1.length - 1], cv.KeyPoint(5.000, 2.000, 5.000, 4.000, 1.000, 0, 0));

    vec1.reserve(21);
    vec1.resize(21);
    expect(vec1.length, 21);
    expect(vec1.size(), 21);

    final vec3 = cv.VecKeyPoint.generate(vec1.length, (i) => vec1[i]);
    vec1.extend(vec3);
    expect(vec1.length, 42);

    vec3.clear();
    vec3.shrinkToFit();
    expect(vec3.length, 0);

    vec1.dispose();
  });
}
