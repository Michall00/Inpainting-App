enum ExecutionProvider {
  auto,
  cpu,
}

String executionProviderLabel(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider.auto:
      return 'CoreML';
    case ExecutionProvider.cpu:
      return 'CPU only';
  }
}

String executionProviderValue(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider.auto:
      return 'coreml';
    case ExecutionProvider.cpu:
      return 'cpu';
  }
}
