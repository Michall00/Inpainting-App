enum ExecutionProvider {
  auto,
  cpu,
  coreml,
}

String executionProviderLabel(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider.auto:
      return 'Auto (CoreML > CPU)';
    case ExecutionProvider.cpu:
      return 'CPU only';
    case ExecutionProvider.coreml:
      return 'CoreML only';
  }
}

String executionProviderValue(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider.auto:
      return 'auto';
    case ExecutionProvider.cpu:
      return 'cpu';
    case ExecutionProvider.coreml:
      return 'coreml';
  }
}
