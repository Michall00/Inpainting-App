import 'package:flutter/material.dart';

class InpaintingHeader extends StatelessWidget {
  final String segmentationPrecisionLabel;
  final String inpaintingModelLabel;
  final String executionProviderLabel;
  final bool isSegmentationInProgress;

  const InpaintingHeader({
    super.key,
    required this.segmentationPrecisionLabel,
    required this.inpaintingModelLabel,
    required this.executionProviderLabel,
    required this.isSegmentationInProgress,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final colorScheme = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Inpainting Studio',
            style: textTheme.headlineSmall?.copyWith(
              color: colorScheme.onSurface,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Mask, refine, and repair in a single workspace.',
            style: textTheme.bodyMedium?.copyWith(
              color: colorScheme.onSurface.withOpacity(0.7),
            ),
          ),
          const SizedBox(height: 14),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _StatusPill(
                icon: Icons.tune,
                label: segmentationPrecisionLabel,
              ),
              _StatusPill(
                icon: Icons.auto_fix_high,
                label: inpaintingModelLabel,
              ),
              _StatusPill(
                icon: Icons.computer,
                label: executionProviderLabel,
              ),
            ],
          ),
          if (isSegmentationInProgress) ...[
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                minHeight: 6,
                color: colorScheme.primary,
                backgroundColor: colorScheme.surfaceVariant,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color? color;

  const _StatusPill({
    required this.icon,
    required this.label,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final effectiveColor = color ?? colorScheme.primary;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: effectiveColor.withOpacity(0.12),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: effectiveColor.withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: effectiveColor),
          const SizedBox(width: 6),
          Text(
            label,
            style: Theme.of(context).textTheme.labelMedium?.copyWith(
                  color: effectiveColor,
                  fontWeight: FontWeight.w600,
                ),
          ),
        ],
      ),
    );
  }
}
