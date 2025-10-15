import 'dart:developer' as developer;

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';

class AppLogger {
  const AppLogger._();

  static void log(String message, {String name = 'InpaintingApp'}) {
    if (kDebugMode) {
      debugPrint('[$name] $message');
    }
    developer.log(message, name: name);
    _safeCrashlytics(() {
      FirebaseCrashlytics.instance.log('[$name] $message');
    });
  }

  static void error(
    String message,
    Object error,
    StackTrace stackTrace, {
    String name = 'InpaintingApp',
  }) {
    if (kDebugMode) {
      debugPrint('[$name][error] $message: $error');
      debugPrint(stackTrace.toString());
    }
    developer.log(
      message,
      name: name,
      error: error,
      stackTrace: stackTrace,
    );
    _safeCrashlytics(() {
      FirebaseCrashlytics.instance.recordError(
        error,
        stackTrace,
        reason: message,
      );
    });
  }

  static void _safeCrashlytics(void Function() action) {
    if (Firebase.apps.isEmpty) {
      return;
    }
    try {
      action();
    } catch (_) {
      // Ignore Crashlytics failures in debug environments.
    }
  }
}
