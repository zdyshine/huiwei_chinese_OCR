import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

# def transconv(inputs,out_num):
#     print('out_num',out_num)
#     print('inputs.shape[3]:',inputs.shape[3])
#     filter = tf.get_variable("conv_1", [tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2, 1024, inputs.shape[3]],
#                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 1024)))
#
#     return tf.nn.conv2d_transpose(inputs, filter, output_shape=[inputs.shape[0],tf.shape(inputs)[1]*2,tf.shape(inputs)[2]*2,out_num],strides=[1,2,2,1], name=None)

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    resnet_v1_50/block1 (?, ?, ?, 256)
    resnet_v1_50/block2 (?, ?, ?, 512)
    resnet_v1_50/block3 (?, ?, ?, 1024)
    resnet_v1_50/block4 (?, ?, ?, 2048)
    Shape of f_0 (?, ?, ?, 2048)
    Shape of f_1 (?, ?, ?, 512)
    Shape of f_2 (?, ?, ?, 256)
    Shape of f_3 (?, ?, ?, 64)
    Shape of h_0 (?, ?, ?, 2048), g_0 (?, ?, ?, 2048)
    Shape of h_1 (?, ?, ?, 128), g_1 (?, ?, ?, 128)
    Shape of h_2 (?, ?, ?, 64), g_2 (?, ?, ?, 64)
    Shape of h_3 (?, ?, ?, 32), g_3 (?, ?, ?, 32)

    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        # logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            # num_outputs = [None, 128, 64, 32]
            num_outputs = [None, 2048, 1024, 512, 256]
            # num_outputs = [None, 1024, 512, 256, 128]

            for i in range(4):
                if i == 0: # 最底层h1
                    f[i] = slim.conv2d(f[i], 2048, 1)  # 最底层的1x1conv
                    h[i] = f[i]
                else:
                    f[i] = slim.conv2d(f[i], num_outputs[i], 1)  # f:1,2,3 的 1x1conv
                    # c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    # print('g[i-1]:',g[i-1].shape)
                    # print('f[i]:',f[i].shape)
                    c1_1 = g[i - 1] + f[i]
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 1)
                    if i == 1:
                        h[i] = slim.conv2d(h[i], num_outputs[i+1], 5)
                    else:
                        h[i] = slim.conv2d(h[i], num_outputs[i + 1], 3)
                if i <= 2: # 中间层 h2,h3,
                    g[i] = unpool(h[i])
                else: # 最高层 h4 对h4进行卷积预测
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # print('g[3]:', g[3].shape) # 512
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            # geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry
    # return F_score, geo_map, angle_map


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
