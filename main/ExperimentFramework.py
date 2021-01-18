from gpbasics import global_parameters as global_param

global_param.ensure_init()

import pandas as pd
import numpy as np
import gpmretrieval.AutomaticGpmRetrieval as agr
import gpbasics.Metrics.Metrics as met
import gpbasics.DataHandling.DataInput as di
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.MeanFunctionBasics.BaseMeanFunctions as bmf
import gpbasics.KernelBasics.BaseKernels as bk
import gpbasics.KernelBasics.Kernel as k
import tensorflow as tf
import plotly.graph_objects as go
from main.CovarianceFingerprint import get_covariance_fingerprint, get_matching_regions_from_fingerprint
import json

global_param.p_used_base_kernel = [bk.PeriodicKernel, bk.SquaredExponentialKernel, bk.LinearKernel]


def normalize(vector: np.ndarray):
    vector_min = np.min(vector)
    vector = vector - vector_min

    vector_max = np.max(vector)
    vector = np.reshape(np.array(vector / vector_max), [-1, 1])

    return vector, vector_min, vector_max


def denormalize(vector: np.ndarray, vector_min, vector_max):
    return (vector * vector_max) + vector_min


def matching_region_analysis(x_data, y_data, win_size, stride, decimal_year_min, decimal_year_max, name, path,
                             exemplary_interval, template_kernel: k.Kernel = None):
    x_data, min_x, max_x = normalize(x_data)

    y_data, _, _ = normalize(y_data)

    training_interval = np.array(pd.to_datetime(pd.Series(exemplary_interval))).astype("float")
    training_interval = (training_interval - min_x) / max_x

    training_indices = np.where(np.logical_and(x_data >= training_interval[0], x_data < training_interval[1]))[0]
    training_x = np.reshape(np.take(x_data, training_indices), [-1, 1])
    training_y = np.reshape(np.take(y_data, training_indices), [-1, 1])

    print("Training interval length: %i records" % len(training_x))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.reshape(training_x, [-1, ]),
        y=np.reshape(training_y, [-1, ]), mode='lines', name='data'))

    fig.show()

    if win_size is None:
        win_size = len(training_x)

    data_input = di.DataInput(training_x, training_y, training_x, training_y)
    mean_function = bmf.ZeroMeanFunction(1)
    data_input.set_mean_function(mean_function)

    if template_kernel is None:
        model_retrieval = \
            agr.GpmRetrieval(data_input, {"global_max_depth": 2, "default_window_size": win_size, "npo": 2},
                             met.MetricType.LL, met.MetricType.LL, mht.MatrixApproximations.NONE,
                             mht.NumericalMatrixHandlingType.CHOLESKY_BASED)

        model_retrieval.init_mean_function(mean_function)

        best_gps, _ = model_retrieval.execute_kernel_search(agr.AlgorithmType.CKS)

        template_kernel = best_gps[0].kernel

    x_features, features = get_covariance_fingerprint(
        template_kernel, x_data, y_data, win_size, stride=stride, normalize_per_window=False, test_ratio=0.5)

    epsilon = float(input("epsilon: "))
    limit_indices, limit_x_axis_values = get_matching_regions_from_fingerprint(
        features, x_features, win_size, stride, epsilon)

    # Create traces
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.reshape(denormalize(x_data, decimal_year_min, (decimal_year_max - decimal_year_min)), [-1, ]),
        y=np.reshape(y_data, [-1, ]), mode='lines', name='data'))

    fig.add_trace(go.Scatter(
        x=np.reshape(denormalize(x_features, decimal_year_min, (decimal_year_max - decimal_year_min)), [-1, ]),
        y=np.reshape(features, [-1, ]), mode='markers', name='fingerprint'))

    if limit_indices.shape[0] != 0:
        limit_x_axis_values = denormalize(limit_x_axis_values, decimal_year_min,
                                          (decimal_year_max - decimal_year_min))

        limit_x_axis_values_no_overlap = []

        last_start = None
        last_stop = None

        for row in limit_x_axis_values:
            row = tf.reshape(row, [-1, ]).numpy().tolist()
            if last_start is None:
                last_start = row[0]
                last_stop = row[1]
            elif row[0] <= last_stop:
                last_stop = row[1]
            else:
                limit_x_axis_values_no_overlap.append([last_start, last_stop])
                last_start = row[0]
                last_stop = row[1]

        if last_start is not None and last_stop is not None:
            limit_x_axis_values_no_overlap.append([last_start, last_stop])

        limit_x_axis_values_no_overlap = np.array(limit_x_axis_values_no_overlap)

        starts = np.reshape(limit_x_axis_values_no_overlap[:, 0], [-1, ]).tolist()

        stops = np.reshape(limit_x_axis_values_no_overlap[:, 1], [-1, ]).tolist()

        shape_dicts = []

        for i in range(0, len(starts)):
            shape_dicts.append(dict(
                type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=starts[i],
                y0=0,
                x1=stops[i],
                y1=1,
                fillcolor="LightSalmon",
                opacity=0.5,
                layer="below",
                line_width=0,
            ))

        fig.update_layout(
            shapes=shape_dicts,
            xaxis={
                'showgrid': False
            },
            yaxis={
                'showgrid': False
            }
        )

    fig.show()

    df_limits = pd.DataFrame({"start": starts, "stop": stops})

    df_limits.to_csv(path + name + '_matchingRegions.csv')

    df_base_data = \
        pd.DataFrame({"x_data": np.reshape(denormalize(x_data, decimal_year_min,
                                                       (decimal_year_max - decimal_year_min)), [-1, ]),
                      "y_data": np.reshape(y_data, [-1, ])})

    df_base_data.to_csv(path + name + '_baseData.csv')

    df_fingerprint_data = \
        pd.DataFrame({"x_data": np.reshape(denormalize(x_features, decimal_year_min,
                                                       (decimal_year_max - decimal_year_min)), [-1, ]),
                      "y_data": np.reshape(features, [-1, ])})

    df_fingerprint_data.to_csv(path + name + '_fingerprint.csv')

    hyparam_names = template_kernel.get_hyper_parameter_names()

    hyparam = template_kernel.get_last_hyper_parameter()

    named_hyparam = {hyparam_names[i]: float(hyparam[i].numpy()) for i in range(0, len(hyparam_names))}

    further_infos = {"min_x": min_x, "max_x": max_x, "epsilon": epsilon, "window_size": win_size, "stride": stride,
                     "decimal_year_min": decimal_year_min, "decimal_year_max": decimal_year_max,
                     "covariance_function": {"expression": template_kernel.get_string_representation(),
                                             "hyper_parameters": named_hyparam},
                     "exemplary_interval": exemplary_interval}

    with open(path + name + '_furtherInfo.json', 'w') as json_file:
        json.dump(further_infos, json_file)
