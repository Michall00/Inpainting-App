import 'package:flutter/material.dart';

class InpaintingEmptyState extends StatelessWidget {
  const InpaintingEmptyState({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 84,
            height: 84,
            decoration: BoxDecoration(
              color: colorScheme.primary.withOpacity(0.12),
              borderRadius: BorderRadius.circular(26),
            ),
            child: Icon(
              Icons.photo_size_select_actual_outlined,
              size: 44,
              color: colorScheme.primary,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Start with a photo',
            style: textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Pick an image to begin masking and inpainting.',
            style: textTheme.bodyMedium?.copyWith(
              color: colorScheme.onSurface.withOpacity(0.65),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
