import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
import copy

def accL(target_values, predicted_values, max_training_value):
    """
    Computes Loss Accuracy (accL) as 1 minus the Mean Absolute Error (MAE)
    with min-max scaling applied to target values.
    """
    target_values = np.array(target_values)
    predicted_values = np.array(predicted_values)
    scaled_target = target_values / max_training_value
    scaled_predicted = predicted_values / max_training_value
    mean_abs_error = np.mean(np.abs(scaled_predicted - scaled_target))
    return 1 - mean_abs_error

def accT(target_values, predicted_values, threshold=50):
    """
    Computes Threshold Accuracy (accT) as 1 minus the average count of errors exceeding the threshold.
    """
    target_values = np.array(target_values)
    predicted_values = np.array(predicted_values)
    errors_exceeding_threshold = np.sum(np.heaviside(np.abs(predicted_values - target_values) - threshold, 0))
    return 1 - (errors_exceeding_threshold / len(target_values))

def accS(target_values, predicted_values, sensitivity_coefficient=0.1, threshold=25):
    """
    Computes Sensitivity Accuracy (accS) with a dynamic threshold based on target values.
    """
    target_values = np.array(target_values)
    predicted_values = np.array(predicted_values)
    dynamic_threshold = sensitivity_coefficient * target_values + threshold
    errors_exceeding_threshold = np.sum(np.heaviside(np.abs(predicted_values - target_values) - dynamic_threshold, 0))
    return 1 - (errors_exceeding_threshold / len(target_values))

def calculate_metrics(outputs, predictions, max_training_value):
    mse_value = mean_squared_error(outputs, predictions)
    accL_value = accL(outputs, predictions, max_training_value)
    accT_value = accT(outputs, predictions)
    accS_value = accS(outputs, predictions)
    return mse_value, accL_value, accT_value, accS_value

def early_stopping(model, epoch_loss, min_loss, epochs_no_improve, best_model_state=None):
    if best_model_state is None:  # Initialize best model state only once
        best_model_state = copy.deepcopy(model.state_dict())

    if epoch_loss < min_loss:
        min_loss = epoch_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == 10:
        return True, min_loss, best_model_state, epochs_no_improve

    return False, min_loss, best_model_state, epochs_no_improve

def scale_predictions_output(predictions, outputs, outputs_reader, max_training_value):
    predictions_scaled = [pred * max_training_value for pred in predictions]
    outputs_scaled = [out * max_training_value for out in outputs]
    outputs_reader_scaled = [out * max_training_value for out in outputs_reader]
    test_outputs = []
    test_outputs_reader = []
    predictions = []
    # Iterate over each output array in outputs_scaled
    for i, output in enumerate(outputs_scaled):
        sublist_test_output_reader = []
        sublist_test_output = []
        sublist_prediction = []
        # Iterate over each element in the current output array
        for j, elem in enumerate(output):
            if elem >= 0:
                sublist_test_output.append(elem)
                sublist_test_output_reader.append(outputs_reader_scaled[i][j])
                sublist_prediction.append(predictions_scaled[i][j])

                if(outputs_reader_scaled[i][j] < 0):
                    print("Output reader is negative: ", outputs_reader_scaled[i][j], i, j, elem)

        test_outputs_reader.append(sublist_test_output_reader)
        test_outputs.append(sublist_test_output)
        predictions.append(sublist_prediction)

    flat_test_outputs = [elem for sublist in test_outputs for elem in sublist]
    flat_test_outputs_reader = [elem for sublist in test_outputs_reader for elem in sublist]
    flat_predictions = [elem for sublist in predictions for elem in sublist]

    return flat_predictions, flat_test_outputs, flat_test_outputs_reader

def my_data_loader(dataset, config):
    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=False
    )
    return loader