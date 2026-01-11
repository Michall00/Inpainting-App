import 'package:flutter/material.dart';

class InpaintingActionBar extends StatelessWidget {
  final VoidCallback onPick;
  final VoidCallback onCamera;
  final VoidCallback onInpaint;
  final VoidCallback? onSave;
  final VoidCallback onSelectSegmentationModel;
  final VoidCallback onSelectExecutionProvider;
  final VoidCallback onSelectInpaintingModel;
  final bool showHintControls;
  final bool isPositiveSelected;
  final bool isNegativeSelected;
  final bool isSegmentationInProgress;
  final VoidCallback onSelectPositive;
  final VoidCallback onSelectNegative;
  final bool showClear;
  final VoidCallback onClear;

  const InpaintingActionBar({
    super.key,
    required this.onPick,
    required this.onCamera,
    required this.onInpaint,
    required this.onSave,
    required this.onSelectSegmentationModel,
    required this.onSelectExecutionProvider,
    required this.onSelectInpaintingModel,
    required this.showHintControls,
    required this.isPositiveSelected,
    required this.isNegativeSelected,
    required this.isSegmentationInProgress,
    required this.onSelectPositive,
    required this.onSelectNegative,
    required this.showClear,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return SafeArea(
      child: Container(
        margin: const EdgeInsets.fromLTRB(16, 4, 16, 12),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        decoration: BoxDecoration(
          color: colorScheme.surface.withOpacity(0.92),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: colorScheme.outline.withOpacity(0.1)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.08),
              blurRadius: 16,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _ActionButton(
                icon: Icons.photo_library,
                label: 'Gallery',
                tooltip: 'Pick from gallery',
                onPressed: onPick,
              ),
              _ActionButton(
                icon: Icons.camera_alt,
                label: 'Camera',
                tooltip: 'Take a photo',
                onPressed: onCamera,
              ),
              _ActionButton(
                icon: Icons.auto_fix_high,
                label: 'Inpaint',
                tooltip: 'Run inpainting',
                isPrimary: true,
                onPressed: onInpaint,
              ),
              _ActionButton(
                icon: Icons.save_alt,
                label: 'Save',
                tooltip: 'Save to gallery',
                onPressed: onSave,
              ),
              _ActionButton(
                icon: Icons.tune,
                label: 'SAM',
                tooltip: 'Choose MobileSAM model',
                onPressed: onSelectSegmentationModel,
              ),
              _ActionButton(
                icon: Icons.computer,
                label: 'Exec',
                tooltip: 'Execution environment',
                onPressed: onSelectExecutionProvider,
              ),
              _ActionButton(
                icon: Icons.swap_vert,
                label: 'MI-GAN',
                tooltip: 'Choose MI-GAN model',
                onPressed: onSelectInpaintingModel,
              ),
              if (showHintControls)
                _ActionButton(
                  icon: Icons.add_circle,
                  label: 'Add',
                  tooltip: 'Positive hint',
                  isActive: isPositiveSelected,
                  activeColor: Colors.green,
                  onPressed:
                      isSegmentationInProgress ? null : onSelectPositive,
                ),
              if (showHintControls)
                _ActionButton(
                  icon: Icons.remove_circle,
                  label: 'Erase',
                  tooltip: 'Negative hint',
                  isActive: isNegativeSelected,
                  activeColor: Colors.red,
                  onPressed:
                      isSegmentationInProgress ? null : onSelectNegative,
                ),
              if (showClear)
                _ActionButton(
                  icon: Icons.clear,
                  label: 'Clear',
                  tooltip: 'Clear drawing',
                  onPressed: onClear,
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String tooltip;
  final VoidCallback? onPressed;
  final bool isActive;
  final bool isPrimary;
  final Color? activeColor;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.tooltip,
    required this.onPressed,
    this.isActive = false,
    this.isPrimary = false,
    this.activeColor,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final isEnabled = onPressed != null;
    final baseColor =
        isPrimary ? colorScheme.primary : colorScheme.surfaceVariant;
    final activeBase = activeColor ?? colorScheme.primary;

    final background = isActive ? activeBase : baseColor;
    final foreground =
        isActive ? colorScheme.onPrimary : colorScheme.onSurfaceVariant;

    final effectiveBackground =
        isEnabled ? background : background.withOpacity(0.5);
    final effectiveForeground =
        isEnabled ? foreground : foreground.withOpacity(0.4);

    return Tooltip(
      message: tooltip,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Material(
              color: effectiveBackground,
              borderRadius: BorderRadius.circular(16),
              child: InkWell(
                onTap: onPressed,
                borderRadius: BorderRadius.circular(16),
                child: SizedBox(
                  width: 48,
                  height: 48,
                  child: Icon(icon, color: effectiveForeground),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Text(
              label,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: isEnabled
                        ? colorScheme.onSurface
                        : colorScheme.onSurface.withOpacity(0.4),
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
