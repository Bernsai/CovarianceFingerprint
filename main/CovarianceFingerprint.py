from typing import Tuple

import tensorflow as tf

import gpbasics.DataHandling.DataInput as di
from gpbasics import global_parameters as global_param
from gpbasics.KernelBasics import Kernel as k

global_param.ensure_init()

import numpy as np


def get_matching_regions_from_fingerprint(fingerprint: tf.Tensor, x_fingerprint, window_size: int, stride: int,
                                          epsilon: float):
    assert len(fingerprint) == len(x_fingerprint)

    boolean_mask_threshold = tf.math.less(fingerprint, epsilon)

    boolean_mask_threshold_left_shift = tf.concat([tf.reshape([False], [1, ]), boolean_mask_threshold[:-1]], axis=0)

    boolean_mask_threshold_right_shift = tf.concat([boolean_mask_threshold[1:], tf.reshape([False], [1, ])], axis=0)

    start_indices = tf.where(tf.logical_and(
        boolean_mask_threshold, tf.logical_not(boolean_mask_threshold_left_shift)))

    stop_indices = tf.where(tf.logical_and(
        boolean_mask_threshold, tf.logical_not(boolean_mask_threshold_right_shift))) + \
                   tf.cast(tf.math.floor(window_size / stride), dtype=tf.int64)

    stop_indices = tf.clip_by_value(stop_indices, clip_value_min=0, clip_value_max=len(fingerprint) - 1)

    limit_indices = tf.concat([start_indices, stop_indices], axis=1)

    limit_x_axis_values = tf.concat([tf.gather(x_fingerprint, start_indices),
                                     tf.gather(x_fingerprint, stop_indices)], axis=1)

    return limit_indices, limit_x_axis_values


def get_covariance_fingerprint(
        query_covariance_structure: k.Kernel, data_x_train: np.ndarray, data_y_train: np.ndarray,
        window_size: int, stride: int = 1, test_ratio: float = 0.2, normalize_per_window: bool = True) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    n = len(data_x_train)

    # x has to be scaled to [0;1]
    x = tf.constant(data_x_train, dtype=global_param.p_dtype)

    y = tf.constant(data_y_train, dtype=global_param.p_dtype)

    is_equidistant = di.is_equidistant(data_x_train)

    fingerprint = start_template_query_tf_graph(
        query_covariance_structure, x, y, window_size, stride, n, is_equidistant,
        normalize_per_window, test_ratio)

    len_fingerprint = tf.cast(tf.math.floor(n / stride), dtype=tf.int32)

    len_strided_window = tf.cast(tf.math.floor(window_size / stride), dtype=tf.int32)

    fingerprint_sliced = fingerprint[0:int(len_fingerprint - len_strided_window)]

    x_fingerprint = x[0:n - (window_size - 1):stride]

    if len(fingerprint_sliced) > len(x_fingerprint):
        x_fingerprint = x[0:n - (window_size - 1):stride]

    if len(fingerprint_sliced) < len(x_fingerprint):
        fingerprint_sliced = fingerprint[0:int(len_fingerprint - (len_strided_window- 1))]

    assert len(fingerprint_sliced) == len(x_fingerprint)

    return x_fingerprint, fingerprint_sliced


@tf.function(experimental_relax_shapes=True)
def start_template_query_tf_graph(query_covariance_structure: k.Kernel, x: tf.Tensor, y: tf.Tensor,
                                  frame_length, frame_step, data_length, equidistant_x, normalize_per_window,
                                  test_ratio) -> tf.Tensor:
    if equidistant_x:
        x = tf.signal.frame(tf.reshape(tf.linspace(start=0.0, num=data_length, stop=1.0), [-1, 1]),
                            frame_length, frame_step, axis=0, pad_end=True)
        x = tf.cast(x, dtype=global_param.p_dtype)
    else:
        x = tf.signal.frame(x, frame_length, frame_step, axis=0, pad_end=True)

    y = tf.signal.frame(y, frame_length, frame_step, axis=0, pad_end=True)

    idx_test_samples = 0
    n_test_samples = int(np.round(frame_length * test_ratio))
    n_train_samples = int(frame_length - n_test_samples)

    # Endpoint false means
    indices = tf.cast(tf.random.shuffle(
        tf.linspace(start=0, num=frame_length, stop=frame_length - 1)), dtype=tf.int32)

    rmses = list()

    batches = tf.constant(int(np.ceil(data_length / frame_step)), tf.int32)

    if normalize_per_window:
        x = x - tf.reduce_min(x, axis=0)
        x = x / tf.reduce_max(x, axis=0)

    foldings = int(np.floor(1 / test_ratio))
    for i in range(foldings):
        stop_idx_test = (n_test_samples + idx_test_samples)
        test_indices = tf.sort(indices[idx_test_samples:stop_idx_test])
        train_indices = tf.sort(tf.concat([indices[:idx_test_samples], indices[stop_idx_test:]], axis=0))

        x_train = tf.gather(x, train_indices, axis=1)
        x_test = tf.gather(x, test_indices, axis=1)
        y_train = tf.gather(y, train_indices, axis=1)
        y_test = tf.gather(y, test_indices, axis=1)

        rmses.append(one_fold_mse(x_train, x_test, y_train, y_test, batches, equidistant_x, n_test_samples,
                                  n_train_samples, query_covariance_structure))

        idx_test_samples += n_test_samples
        if (idx_test_samples + n_test_samples) >= frame_length:
            idx_test_samples = frame_length - 1 - n_test_samples

    avg_rmse = tf.add_n(rmses) / tf.cast(foldings, global_param.p_dtype)

    return tf.reshape(avg_rmse, [-1, ])


@tf.function(experimental_relax_shapes=True)
def one_fold_mse(x_train, x_test, y_train, y_test, batches, equidistant_x,
                 n_test_samples, n_train_samples, query_covariance_structure):
    hyper_parameter = query_covariance_structure.get_last_hyper_parameter()

    def map_get_k(param_x):
        return query_covariance_structure.get_tf_tensor(hyper_parameter, param_x[0], param_x[1])

    stack = tf.stack([x_train, x_train], 1)
    if not equidistant_x:
        k = tf.map_fn(map_get_k, stack)

        k_noised = k + tf.eye(n_train_samples, batch_shape=[k.shape[0]], dtype=global_param.p_dtype) \
                   * global_param.p_cov_matrix_jitter
        l = tf.linalg.cholesky(k_noised)

        stack = (x_train, x_test)
        shape = [n_train_samples, n_test_samples]
        ks = tf.map_fn(map_get_k, stack,
                       fn_output_signature=tf.TensorSpec(shape=shape, dtype=global_param.p_dtype))
    else:
        single_x_train = tf.reshape(tf.gather(x_train, [0], axis=0), [-1, 1])
        single_k = query_covariance_structure.get_tf_tensor(hyper_parameter, single_x_train, single_x_train)

        single_k_noised = single_k + tf.eye(n_train_samples, dtype=global_param.p_dtype) \
                          * global_param.p_cov_matrix_jitter
        single_l = tf.linalg.cholesky(single_k_noised)

        l = tf.stack(tf.repeat(tf.reshape(single_l, [1, n_train_samples, n_train_samples]),
                               repeats=batches, axis=0), axis=0)

        single_x_test = tf.reshape(tf.gather(x_test, [0], axis=0), [-1, 1])

        single_ks = query_covariance_structure.get_tf_tensor(hyper_parameter, single_x_train, single_x_test)

        ks = tf.stack(tf.repeat(tf.reshape(single_ks, [1, n_train_samples, n_test_samples]),
                                repeats=batches, axis=0), axis=0)
    l_alpha = tf.linalg.triangular_solve(
        tf.transpose(l, perm=[0, 2, 1]), tf.linalg.triangular_solve(l, y_train, name="inner_tri"),
        lower=False, name="outer_tri")
    posterior_mu = tf.matmul(tf.transpose(ks, perm=[0, 2, 1]), l_alpha)
    rmse = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(posterior_mu, y_test), axis=1))
    return rmse
