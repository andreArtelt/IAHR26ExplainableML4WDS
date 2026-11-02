import numpy as np
import matplotlib.pyplot as plt

from generate_data import pressure_sensor_locations


def get_f_dist(desc, epsilon=1e-3):
    if desc == "l0":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf) < epsilon)
    elif desc == "l1":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf))
    elif desc == "l2":
        return lambda x_orig, x_cf: np.sum(np.square(x_orig - x_cf))
    else:
        raise ValueError(f"Unknown distance function '{desc}'")


def evaluate_fault_detection(faults_time_pred, faults_time_truth, test_times, tol=10, false_alarms_tol=2, use_intervals=True):
    test_minus_pred = list(set(test_times) - set(faults_time_pred))

    # Compute detection delay
    detection_delay = None
    for t in faults_time_pred:
        if t in faults_time_truth:
            detection_delay = list(faults_time_truth).index(t)  # Check first TP only!
            break

    # Compute TPs, FPs, etc. for every point in time when a fault is present
    if len(faults_time_pred) == 0:
        TP_ex = 0.
        FP_ex = 0.
    else:
        TP_ex = np.sum([t in faults_time_truth for t in faults_time_pred]) / len(faults_time_pred)
        FP_ex = np.sum([t not in faults_time_truth for t in faults_time_pred]) / len(faults_time_pred)
    FN_ex = np.sum([t in faults_time_truth for t in test_minus_pred]) / len(test_minus_pred)
    TN_ex = np.sum([t not in faults_time_truth for t in test_minus_pred]) / len(test_minus_pred)

    # Return results
    return {"detection_delay": detection_delay, "tp_ex": TP_ex, "fp_ex": FP_ex, "fn_ex": FN_ex, "tn_ex": TN_ex}


def plot_counterfactual(cf: np.ndarray, export_fig: str = None) -> None:
    sensor_names = [f"Node {i}" for i in pressure_sensor_locations]

    plt.figure()
    plt.bar(sensor_names, cf)
    plt.xlabel("Pressure sensors")
    plt.xticks(rotation=90)
    plt.ylabel("Normalized amount of change")

    if export_fig is None:
        plt.show()
    else:
        plt.savefig(export_fig, format="pdf", bbox_inches="tight")
