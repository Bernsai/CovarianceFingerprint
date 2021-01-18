import tensorflow as tf

from gpbasics import global_parameters as global_param

global_param.ensure_init()

import gpbasics.Auxiliary.Unique2D as u2d


@tf.function(experimental_relax_shapes=True)
def all_combinations(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    rows_a = len(interval_set_a)
    rows_b = len(interval_set_b)
    repeated_a = tf.repeat(interval_set_a, rows_b, axis=0)
    repeated_b = tf.reshape(tf.repeat(tf.reshape(interval_set_b, [1, -1]), rows_a, axis=0), [-1, 2])
    cross = tf.concat([repeated_a, repeated_b], axis=1)
    return cross


def before(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(cross[:, 1] < cross[:, 2]), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def after(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(cross[:, 0] > cross[:, 3]), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def overlap_length(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices_overlap = tf.reshape(tf.where(
        tf.logical_and(cross[:, 1] > cross[:, 2],
                       tf.logical_and(cross[:, 1] <= cross[:, 3], cross[:, 0] <= cross[:, 2]))), [-1, ])

    indices_overlapped_by = tf.reshape(tf.where(
        tf.logical_and(cross[:, 3] > cross[:, 0],
                       tf.logical_and(cross[:, 3] <= cross[:, 1], cross[:, 2] <= cross[:, 0]))), [-1, ])

    results_overlap = tf.gather(cross, indices_overlap)

    length_overlap = tf.reduce_sum(tf.abs(results_overlap[:, 1] - results_overlap[:, 2]) + 1)

    results_overlapped_by = tf.gather(cross, indices_overlapped_by)

    length_overlapped_by = tf.reduce_sum(tf.abs(results_overlapped_by[:, 3] - results_overlapped_by[:, 0]) + 1)

    return length_overlap + length_overlapped_by


def overlaps(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(
        tf.logical_and(cross[:, 1] > cross[:, 2],
                       tf.logical_and(cross[:, 1] < cross[:, 3], cross[:, 0] < cross[:, 2]))), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def overlapped_by(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(
        tf.logical_and(cross[:, 3] > cross[:, 0],
                       tf.logical_and(cross[:, 3] < cross[:, 1], cross[:, 2] < cross[:, 0]))), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def contains(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 0] < cross[:, 2], cross[:, 1] > cross[:, 3])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def during(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 0] > cross[:, 2], cross[:, 1] < cross[:, 3])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def starts(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 0] == cross[:, 2], cross[:, 1] < cross[:, 3])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def started_by(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 0] == cross[:, 2], cross[:, 1] > cross[:, 3])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def finishes(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 1] == cross[:, 3], cross[:, 0] > cross[:, 2])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def finished_by(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 1] == cross[:, 3], cross[:, 0] < cross[:, 2])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def meets(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(cross[:, 1] == cross[:, 2]), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def met_by(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(cross[:, 0] == cross[:, 3]), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result


def equal(interval_set_a: tf.Tensor, interval_set_b: tf.Tensor) -> tf.Tensor:
    cross = all_combinations(interval_set_a, interval_set_b)

    indices = tf.reshape(tf.where(tf.logical_and(cross[:, 0] == cross[:, 2], cross[:, 1] == cross[:, 3])), [-1, ])

    result = tf.gather(cross, indices)[:, 0:2]

    if len(result) > 1:
        result = u2d.tf_unique_2d(result)

    return result
