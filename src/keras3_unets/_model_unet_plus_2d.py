# ruff: noqa: F401, F403


import warnings

from keras.layers import Input
from keras.models import Model

from keras3_unets._backbone_zoo import bach_norm_checker, backbone_zoo
from keras3_unets._model_unet_2d import UNET_left, UNET_right
from keras3_unets.activations import GELU, Snake
from keras3_unets.layer_utils import CONV_output, CONV_stack, decode_layer


def unet_plus_2d_base(
    input_tensor,
    filter_num,
    stack_num_down=2,
    stack_num_up=2,
    activation="ReLU",
    batch_norm=False,
    pool=True,
    unpool=True,
    deep_supervision=False,
    backbone=None,
    weights="imagenet",
    freeze_backbone=True,
    freeze_batch_norm=True,
    name="xnet",
):
    """
    The base of U-net++ with an optional ImageNet-trained backbone
    
    unet_plus_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                      activation='ReLU', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet')
    
    ----------
    Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture 
    for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning 
    for Clinical Decision Support (pp. 3-11). Springer, Cham.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras3_unets.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.   
        deep_supervision: True for a model that supports deep supervision. Details see Zhou et al. (2018).
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        If deep_supervision = False; Then the output is a tensor.
        If deep_supervision = True; Then the output is a list of tensors
            with the first tensor obtained from the first downsampling level (for checking the input/output shapes only),
            the second to the `depth-1`-th tensors obtained from each intermediate upsampling levels (deep supervision tensors),
            and the last tensor obtained from the end of the base.
    
    """

    depth_ = len(filter_num)
    # allocate nested lists for collecting output tensors
    X_nest_skip = [[] for _ in range(depth_)]

    # no backbone cases
    if backbone is None:
        X = input_tensor

        # downsampling blocks (same as in 'unet_2d')
        X = CONV_stack(
            X,
            filter_num[0],
            stack_num=stack_num_down,
            activation=activation,
            batch_norm=batch_norm,
            name=f"{name}_down0",
        )
        X_nest_skip[0].append(X)
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(
                X,
                f,
                stack_num=stack_num_down,
                activation=activation,
                pool=pool,
                batch_norm=batch_norm,
                name=f"{name}_down{i + 1}",
            )
            X_nest_skip[0].append(X)

    # backbone cases
    else:
        # handling VGG16 and VGG19 separately
        if "VGG" in backbone:
            backbone_ = backbone_zoo(
                backbone,
                weights,
                input_tensor,
                depth_,
                freeze_backbone,
                freeze_batch_norm,
            )
            # collecting backbone feature maps
            X_nest_skip[0] += backbone_(
                [
                    input_tensor,
                ]
            )
            depth_encode = len(X_nest_skip[0])

        # for other backbones
        else:
            backbone_ = backbone_zoo(
                backbone,
                weights,
                input_tensor,
                depth_ - 1,
                freeze_backbone,
                freeze_batch_norm,
            )
            # collecting backbone feature maps
            X_nest_skip[0] += backbone_(
                [
                    input_tensor,
                ]
            )
            depth_encode = len(X_nest_skip[0]) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:
            # begins at the deepest available tensor
            X = X_nest_skip[0][-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(
                    X,
                    filter_num[i_real],
                    stack_num=stack_num_down,
                    activation=activation,
                    pool=pool,
                    batch_norm=batch_norm,
                    name=f"{name}_down{i_real + 1}",
                )
                X_nest_skip[0].append(X)

    X = X_nest_skip[0][-1]

    for nest_lev in range(1, depth_):
        # depth difference between the deepest nest skip and the current upsampling
        depth_lev = depth_ - nest_lev

        # number of available encoded tensors
        depth_decode = len(X_nest_skip[nest_lev - 1])

        # loop over individual upsamling levels
        for i in range(1, depth_decode):
            # collecting previous downsampling outputs
            previous_skip = []
            for previous_lev in range(nest_lev):
                previous_skip.append(X_nest_skip[previous_lev][i - 1])

            # upsamping block that concatenates all available (same feature map size) down-/upsampling outputs
            X_nest_skip[nest_lev].append(
                UNET_right(
                    X_nest_skip[nest_lev - 1][i],
                    previous_skip,
                    filter_num[i - 1],
                    stack_num=stack_num_up,
                    activation=activation,
                    unpool=unpool,
                    batch_norm=batch_norm,
                    concat=False,
                    name=f"{name}_up{nest_lev - 1}_from{i - 1}",
                )
            )

        if depth_decode < depth_lev + 1:
            X = X_nest_skip[nest_lev - 1][-1]

            for j in range(depth_lev - depth_decode + 1):
                j_real = j + depth_decode
                X = UNET_right(
                    X,
                    None,
                    filter_num[j_real - 1],
                    stack_num=stack_num_up,
                    activation=activation,
                    unpool=unpool,
                    batch_norm=batch_norm,
                    concat=False,
                    name=f"{name}_up{nest_lev - 1}_from{j_real - 1}",
                )
                X_nest_skip[nest_lev].append(X)

    # output
    if deep_supervision:
        X_list = []

        for i in range(depth_):
            X_list.append(X_nest_skip[i][0])

        return X_list

    else:
        return X_nest_skip[-1][0]


def unet_plus_2d(
    input_size,
    filter_num,
    n_labels,
    stack_num_down=2,
    stack_num_up=2,
    activation="ReLU",
    output_activation="Softmax",
    batch_norm=False,
    pool=True,
    unpool=True,
    deep_supervision=False,
    backbone=None,
    weights="imagenet",
    freeze_backbone=True,
    freeze_batch_norm=True,
    name="xnet",
):
    """
    U-net++ with an optional ImageNet-trained backbone.
    
    unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                 backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet')
    
    ----------
    Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture 
    for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning 
    for Clinical Decision Support (pp. 3-11). Springer, Cham.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
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
        deep_supervision: True for a model that supports deep supervision. Details see Zhou et al. (2018).
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras model.
    
    """

    depth_ = len(filter_num)

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)
    # base
    X = unet_plus_2d_base(
        IN,
        filter_num,
        stack_num_down=stack_num_down,
        stack_num_up=stack_num_up,
        activation=activation,
        batch_norm=batch_norm,
        pool=pool,
        unpool=unpool,
        deep_supervision=deep_supervision,
        backbone=backbone,
        weights=weights,
        freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm,
        name=name,
    )

    # output
    if deep_supervision:
        if (backbone is not None) and freeze_backbone:
            backbone_warn = "\n\nThe shallowest U-net++ deep supervision branch directly connects to a frozen backbone.\nTesting your configurations on `keras3_unets.base.unet_plus_2d_base` is recommended."
            warnings.warn(backbone_warn)

        # model base returns a list of tensors
        X_list = X
        OUT_list = []

        print(
            '----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n'
        )

        # no backbone or VGG backbones
        # depth_ > 2 is expected (a least two downsampling blocks)
        if (backbone is None) or "VGG" in backbone:
            for i in range(depth_ - 1):
                if output_activation is None:
                    print(f"\t{name}_output_sup{i}")
                else:
                    print(f"\t{name}_output_sup{i}_activation")

                OUT_list.append(
                    CONV_output(
                        X_list[i],
                        n_labels,
                        kernel_size=1,
                        activation=output_activation,
                        name=f"{name}_output_sup{i}",
                    )
                )
        # other backbones
        else:
            for i in range(1, depth_ - 1):
                if output_activation is None:
                    print(f"\t{name}_output_sup{i - 1}")
                else:
                    print(f"\t{name}_output_sup{i - 1}_activation")

                # an extra upsampling for creating full resolution feature maps
                X = decode_layer(
                    X_list[i],
                    filter_num[i],
                    2,
                    unpool,
                    activation=activation,
                    batch_norm=batch_norm,
                    name=f"{name}_sup{i - 1}_up",
                )

                X = CONV_output(
                    X,
                    n_labels,
                    kernel_size=1,
                    activation=output_activation,
                    name=f"{name}_output_sup{i - 1}",
                )
                OUT_list.append(X)

        if output_activation is None:
            print(f"\t{name}_output_final")
        else:
            print(f"\t{name}_output_final_activation")

        OUT_list.append(
            CONV_output(
                X_list[-1],
                n_labels,
                kernel_size=1,
                activation=output_activation,
                name=f"{name}_output_final",
            )
        )

    else:
        OUT = CONV_output(
            X,
            n_labels,
            kernel_size=1,
            activation=output_activation,
            name=f"{name}_output",
        )
        OUT_list = [
            OUT,
        ]

    # model
    model = Model(
        inputs=[
            IN,
        ],
        outputs=OUT_list,
        name=f"{name}_model",
    )

    return model
