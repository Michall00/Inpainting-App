import 'dart:developer' as developer;

import 'package:device_info_plus/device_info_plus.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';

class AppLogger {
  const AppLogger._();

  static const String _defaultName = 'InpaintingApp';
  static String _deviceInfo = 'unknown_device';
  static String _osVersion = 'unknown_os';

  static String get deviceInfo => _deviceInfo;
  static String get osVersion => _osVersion;

  static FirebaseCrashlytics get _crashlytics {
    if (Firebase.apps.isEmpty) {
      throw StateError(
        'Firebase.initializeApp must be called before using Crashlytics.',
      );
    }
    return FirebaseCrashlytics.instance;
  }

  static Future<void> init() async {
    try {
      final deviceInfoPlugin = DeviceInfoPlugin();
      final deviceInfo = await deviceInfoPlugin.deviceInfo;

      if (deviceInfo is AndroidDeviceInfo) {
        _deviceInfo = '${deviceInfo.manufacturer} ${deviceInfo.model}';
        _osVersion =
            'Android ${deviceInfo.version.release} (SDK ${deviceInfo.version.sdkInt})';
      } else if (deviceInfo is IosDeviceInfo) {
        _deviceInfo = '${deviceInfo.name} ${deviceInfo.model}';
        _osVersion = 'iOS ${deviceInfo.systemVersion}';
      } else {
        debugPrint('AppLogger: unsupported platform for device info');
      }

      setKey('device_info', _deviceInfo);
      setKey('os_version', _osVersion);
    } catch (error, stackTrace) {
      debugPrint('AppLogger init failed: $error');
      developer.log(
        'AppLogger init failed',
        error: error,
        stackTrace: stackTrace,
        name: _defaultName,
      );
    }
  }

  static void log(
    String message, {
    String name = _defaultName,
    bool sendToCrashlytics = true,
    Object? error,
    StackTrace? stackTrace,
  }) {
    final deviceContext = 'device=$_deviceInfo | os=$_osVersion';
    final formattedMessage =
        error == null ? '$message | $deviceContext' : '$message | error: ${error.toString()} | $deviceContext';
    if (kDebugMode) {
      debugPrint('[$name] $formattedMessage');
      if (stackTrace != null) {
        debugPrint(stackTrace.toString());
      }
    }
    developer.log(
      formattedMessage,
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
      crashlytics.setCustomKey(key, value ?? 'null');
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
