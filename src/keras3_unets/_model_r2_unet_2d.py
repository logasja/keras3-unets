# ruff: noqa: F401, F403
from keras import Model, Input, layers

from keras3_unets.activations import GELU, Snake
from keras3_unets.layer_utils import CONV_output, CONV_stack, encode_layer, decode_layer


def RR_CONV(
    X,
    channel,
    kernel_size=3,
    stack_num=2,
    recur_num=2,
    activation="ReLU",
    batch_norm=False,
    name="rr",
):
    """
    Recurrent convolutional layers with skip connection.

    RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    """

    activation_func = eval(activation)

    layer_skip = layers.Conv2D(channel, 1, name=f"{name}_conv")(X)
    layer_main = layer_skip

    for i in range(stack_num):
        layer_res = layers.Conv2D(
            channel, kernel_size, padding="same", name=f"{name}_conv{i}"
        )(layer_main)

        if batch_norm:
            layer_res = layers.BatchNormalization(name=f"{name}_bn{i}")(layer_res)

        layer_res = activation_func(name=f"{name}_activation{i}")(layer_res)

        for j in range(recur_num):
            layer_add = layers.Add([layer_res, layer_main], name=f"{name}_add{i}_{j}")

            layer_res = layers.Conv2D(
                channel,
                kernel_size,
                padding="same",
                name=f"{name}_conv{i}_{j}",
            )(layer_add)

            if batch_norm:
                layer_res = layers.BatchNormalization(name=f"{name}_bn{i}_{j}")(
                    layer_res
                )

            layer_res = activation_func(name=f"{name}_activation{i}_{j}")(layer_res)

        layer_main = layer_res

    out_layer = layers.Add([layer_main, layer_skip], name=f"{name}_add{i}")

    return out_layer


def UNET_RR_left(
    X,
    channel,
    kernel_size=3,
    stack_num=2,
    recur_num=2,
    activation="ReLU",
    pool=True,
    batch_norm=False,
    name="left0",
):
    """
    The encoder block of R2U-Net.

    UNET_RR_left(X, channel, kernel_size=3,
                 stack_num=2, recur_num=2, activation='ReLU',
                 pool=True, batch_norm=False, name='left0')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    *downsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    """
    pool_size = 2

    # maxpooling layer vs strided convolutional layers
    X = encode_layer(
        X,
        channel,
        pool_size,
        pool,
        activation=activation,
        batch_norm=batch_norm,
        name=f"{name}_encode",
    )

    # stack linear convolutional layers
    X = RR_CONV(
        X,
        channel,
        stack_num=stack_num,
        recur_num=recur_num,
        activation=activation,
        batch_norm=batch_norm,
        name=name,
    )
    return X


def UNET_RR_right(
    X,
    X_list,
    channel,
    kernel_size=3,
    stack_num=2,
    recur_num=2,
    activation="ReLU",
    unpool=True,
    batch_norm=False,
    name="right0",
):
    """
    The decoder block of R2U-Net.

    UNET_RR_right(X, X_list, channel, kernel_size=3,
                  stack_num=2, recur_num=2, activation='ReLU',
                  unpool=True, batch_norm=False, name='right0')

    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor

    """

    pool_size = 2

    X = decode_layer(
        X,
        channel,
        pool_size,
        unpool,
        activation=activation,
        batch_norm=batch_norm,
        name=f"{name}_decode",
    )

    # linear convolutional layers before concatenation
    X = CONV_stack(
        X,
        channel,
        kernel_size,
        stack_num=1,
        activation=activation,
        batch_norm=batch_norm,
        name=f"{name}_conv_before_concat",
    )

    # Tensor concatenation
    H = layers.Concatenate(
        [
            X,
        ]
        + X_list,
        axis=-1,
        name=f"{name}_concat",
    )

    # stacked linear convolutional layers after concatenation
    H = RR_CONV(
        H,
        channel,
        stack_num=stack_num,
        recur_num=recur_num,
        activation=activation,
        batch_norm=batch_norm,
        name=name,
    )

    return H


def r2_unet_2d_base(
    input_tensor,
    filter_num,
    stack_num_down=2,
    stack_num_up=2,
    recur_num=2,
    activation="ReLU",
    batch_norm=False,
    pool=True,
    unpool=True,
    name="res_unet",
):
    """
    The base of Recurrent Residual (R2) U-Net.
    
    r2_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, recur_num=2,
                    activation='ReLU', batch_norm=False, pool=True, unpool=True, name='res_unet')
    
    ----------
    Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network 
    based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of stacked recurrent convolutional layers per downsampling level/block.
        stack_num_down: number of stacked recurrent convolutional layers per upsampling level/block.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    """

    # activation_func = eval(activation)

    X = input_tensor
    X_skip = []

    # downsampling blocks
    X = RR_CONV(
        X,
        filter_num[0],
        stack_num=stack_num_down,
        recur_num=recur_num,
        activation=activation,
        batch_norm=batch_norm,
        name=f"{name}_down0",
    )
    X_skip.append(X)

    for i, f in enumerate(filter_num[1:]):
        X = UNET_RR_left(
            X,
            f,
            kernel_size=3,
            stack_num=stack_num_down,
            recur_num=recur_num,
            activation=activation,
            pool=pool,
            batch_norm=batch_norm,
            name=f"{name}_down{i + 1}",
        )
        X_skip.append(X)

    # upsampling blocks
    X_skip = X_skip[:-1][::-1]
    for i, f in enumerate(filter_num[:-1][::-1]):
        X = UNET_RR_right(
            X,
            [
                X_skip[i],
            ],
            f,
            stack_num=stack_num_up,
            recur_num=recur_num,
            activation=activation,
            unpool=unpool,
            batch_norm=batch_norm,
            name=f"{name}_up{i + 1}",
        )

    return X


def r2_unet_2d(
    input_size,
    filter_num,
    n_labels,
    stack_num_down=2,
    stack_num_up=2,
    recur_num=2,
    activation="ReLU",
    output_activation="Softmax",
    batch_norm=False,
    pool=True,
    unpool=True,
    name="r2_unet",
):
    """
    Recurrent Residual (R2) U-Net
    
    r2_unet_2d(input_size, filter_num, n_labels, 
               stack_num_down=2, stack_num_up=2, recur_num=2,
               activation='ReLU', output_activation='Softmax', 
               batch_norm=False, pool=True, unpool=True, name='r2_unet')
    
    ----------
    Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network 
    based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of stacked recurrent convolutional layers per downsampling level/block.
        stack_num_down: number of stacked recurrent convolutional layers per upsampling level/block.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.     
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                  
        name: prefix of the created keras layers.
        
    Output
    ----------
        model: a keras model.
    
    """

    # activation_func = eval(activation)

    IN = Input(input_size, name=f"{name}_input")

    # base
    X = r2_unet_2d_base(
        IN,
        filter_num,
        stack_num_down=stack_num_down,
        stack_num_up=stack_num_up,
        recur_num=recur_num,
        activation=activation,
        batch_norm=batch_norm,
        pool=pool,
        unpool=unpool,
        name=name,
    )
    # output layer
    OUT = CONV_output(
        X,
        n_labels,
        kernel_size=1,
        activation=output_activation,
        name=f"{name}_output",
    )

    # functional API model
    model = Model(inputs=[IN], outputs=[OUT], name=f"{name}_model")

    return model
