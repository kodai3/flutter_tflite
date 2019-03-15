import 'dart:async';
// import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class Tflite {
  static const MethodChannel _channel = const MethodChannel('tflite');

  static Future<String> loadModel({
    @required String model,
    @required String labels,
    int numThreads = 1,
  }) async {
    return await _channel.invokeMethod(
      'loadModel',
      {"model": model, "labels": labels, "numThreads": numThreads},
    );
  }

  static Future<List> detectObjectOnImage({
    @required String path,
    String model = "SSDMobileNet",
    double imageMean = 127.5,
    double imageStd = 127.5,
    double threshold = 0.1,
    int numResultsPerClass = 5,
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnImage',
      {
        "path": path,
        "model": model,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "threshold": threshold,
        "numResultsPerClass": numResultsPerClass,
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }
}
