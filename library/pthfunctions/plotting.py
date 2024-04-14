import matplotlib.pyplot as plt
import pandas as pd


def _convert_metric_name(metric_name: str) -> str:
  """
  Converts a metric name to a format with spaces and capitalized first letters.

  Args:
      metric_name: A string representing the metric name (e.g., "test_loss").

  Returns:
      A string representing the metric name in a better format (e.g., "Test Loss").
  """
  words = metric_name.split('_')
  words = [word.capitalize() for word in words]
  return ' '.join(words)


def plot_results_comparison(**kwargs: pd.DataFrame):
  """
  Plots a comparison of all models' results, including train/test loss and accuracy,
  given that they trained for the same number of epochs.

  Args:
      **kwargs: Named arguments where keys are model names and values are pandas DataFrames
                 containing training and testing data. Each DataFrame is expected to have
                 columns named "train_loss", "test_loss", "train_acc", and "test_acc".
  """
  plt.figure(figsize=(15, 10))
  metrics = ["train_loss", "test_loss", "train_acc", "test_acc"]

  for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)

    for model_name, df in kwargs.items():
      epochs = range(len(df))
      plt.plot(epochs, df[metric], label=model_name)
      
    plt.title(_convert_metric_name(metric))
    plt.xlabel("Epochs")
    plt.legend()