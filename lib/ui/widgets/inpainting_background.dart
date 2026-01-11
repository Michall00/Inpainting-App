import 'package:flutter/material.dart';

class InpaintingBackground extends StatelessWidget {
  const InpaintingBackground({super.key});

  @override
  Widget build(BuildContext context) {
    return const DecoratedBox(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Color(0xFFF6F2EA),
            Color(0xFFE7EFEA),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
    );
  }
}
