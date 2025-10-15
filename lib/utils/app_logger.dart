import 'dart:developer' as developer;

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';

class AppLogger {
  const AppLogger._();

  static const String _defaultName = 'InpaintingApp';

  static FirebaseCrashlytics get _crashlytics {
    if (Firebase.apps.isEmpty) {
      throw StateError(
        'Firebase.initializeApp must be called before using Crashlytics.',
      );
    }
    return FirebaseCrashlytics.instance;
  }

  static void log(
    String message, {
    String name = _defaultName,
    bool sendToCrashlytics = true,
    Object? error,
    StackTrace? stackTrace,
  }) {
    final formattedMessage =
        error == null ? message : '$message | error: ${error.toString()}';
    if (kDebugMode) {
      debugPrint('[$name] $formattedMessage');
      if (stackTrace != null) {
        debugPrint(stackTrace.toString());
      }
    }
    developer.log(
      message,
      name: name,
      error: error,
      stackTrace: stackTrace,
    );
    if (sendToCrashlytics) {
      _crashlytics.log('[$name] $formattedMessage');
    }
  }

  static void error(
    String message,
    Object error,
    StackTrace stackTrace, {
    String name = _defaultName,
    bool fatal = false,
  }) {
    log(
      message,
      name: name,
      error: error,
      stackTrace: stackTrace,
    );
    recordError(
      error,
      stackTrace,
      reason: message,
      fatal: fatal,
    );
  }

  static void setKey(String key, Object? value) {
    final crashlytics = _crashlytics;
    if (value is num || value is bool || value is String) {
      crashlytics.setCustomKey(key, value);
    } else if (value == null) {
      crashlytics.setCustomKey(key, 'null');
    } else {
      crashlytics.setCustomKey(key, value.toString());
    }
  }

  static void recordFlutterError(FlutterErrorDetails details) {
    _crashlytics.recordFlutterError(details);
  }

  static void recordError(
    Object error,
    StackTrace stackTrace, {
    String? reason,
    bool fatal = false,
  }) {
    final crashlytics = _crashlytics;
    crashlytics.recordError(
      error,
      stackTrace,
      reason: reason,
      fatal: fatal,
    );
  }
}
