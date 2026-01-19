import 'package:flutter/material.dart';
import 'ui/pages/inpainting_page.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF5A7A5A),
      brightness: Brightness.light,
    );

    return MaterialApp(
      title: 'Inpainting',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: const Color(0xFFF6F3EE),
        fontFamily: 'sans-serif',
        textTheme: const TextTheme(
          headlineSmall: TextStyle(
            fontFamily: 'serif',
            fontWeight: FontWeight.w600,
            letterSpacing: -0.2,
          ),
          titleLarge: TextStyle(
            fontFamily: 'serif',
            fontWeight: FontWeight.w600,
            letterSpacing: -0.2,
          ),
          bodyMedium: TextStyle(
            fontFamily: 'sans-serif',
            height: 1.4,
          ),
          labelSmall: TextStyle(
            fontFamily: 'sans-serif',
            letterSpacing: 0.2,
          ),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          scrolledUnderElevation: 0,
          centerTitle: false,
        ),
        cardTheme: CardThemeData(
          color: colorScheme.surface,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
        ),
        chipTheme: ChipThemeData(
          backgroundColor: colorScheme.surfaceVariant,
          labelStyle: TextStyle(color: colorScheme.onSurfaceVariant),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(999),
          ),
        ),
      ),
      home: const InpaintingPage(),
    );
  }
}
