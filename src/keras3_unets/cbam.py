# https://www.kaggle.com/code/ipythonx/keras-ranzcr-multi-attention-efficientnet-tpu/comments?scriptVersionId=88108908
# Apache license
from keras import ops, layers, activations, initializers, backend


class SpatialAttentionModule(layers.Layer):
    def __init__(self, kernel_size=3):
        """
        paper: https://arxiv.org/abs/1807.06521
        code: https://gist.github.com/innat/99888fa8065ecbf3ae2b297e5c10db70
        """
        super().__init__()
        self.conv1 = layers.Conv2D(
            64,
            kernel_size=kernel_size,
            use_bias=False,
            kernel_initializer="he_normal",
            strides=1,
            padding="same",
            activation=activations.relu,
        )
        self.conv2 = layers.Conv2D(
            32,
            kernel_size=kernel_size,
            use_bias=False,
            kernel_initializer="he_normal",
            strides=1,
            padding="same",
            activation=activations.relu,
        )
        self.conv3 = layers.Conv2D(
            16,
            kernel_size=kernel_size,
            use_bias=False,
            kernel_initializer="he_normal",
            strides=1,
            padding="same",
            activation=activations.relu,
        )
        self.conv4 = layers.Conv2D(
            1,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            strides=1,
            padding="same",
            activation=activations.sigmoid,
        )

    def call(self, inputs):
        avg_out = ops.mean(inputs, axis=3)
        max_out = ops.max(inputs, axis=3)
        x = ops.stack([avg_out, max_out], axis=3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)


class ChannelAttentionModule(layers.Layer):
    def __init__(self, ratio=1):
        """paper: https://arxiv.org/abs/1807.06521
        code: https://gist.github.com/innat/99888fa8065ecbf3ae2b297e5c10db70
        """
        super(ChannelAttentionModule, self).__init__()
        self.ratio = ratio
        self.gapavg = layers.GlobalAveragePooling2D()
        self.gmpmax = layers.GlobalMaxPooling2D()

    def build(self, input_shape):
        self.conv2 = layers.Conv2D(
            input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            activation=activations.relu,
        )
        super(ChannelAttentionModule, self).build(input_shape)

    def call(self, inputs):
        # compute gap and gmp pooling
        gapavg = self.gapavg(inputs)
        gmpmax = self.gmpmax(inputs)
        gapavg = layers.Reshape((1, 1, gapavg.shape[1]))(gapavg)
        gmpmax = layers.Reshape((1, 1, gmpmax.shape[1]))(gmpmax)
        # forward passing to the respected layers
        gapavg_out = self.conv2(gapavg)
        gmpmax_out = self.conv2(gmpmax)
        return activations.sigmoid(gapavg_out + gmpmax_out)


# Original Src: https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
class AttentionWeightedAverage2D(layers.Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get("uniform")
        super(AttentionWeightedAverage2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [layers.InputSpec(ndim=4)]
        assert len(input_shape) == 4
        self.W = self.add_weight(
            shape=(input_shape[3], 1),
            name="{}_W".format(self.name),
            initializer=self.init,
        )
        self._trainable_weights = [self.W]
        super(AttentionWeightedAverage2D, self).build(input_shape)

    def call(self, x):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 2-dimensional weights
        logits = ops.dot(x, self.W)
        x_shape = ops.shape(x)
        logits = ops.reshape(logits, (x_shape[0], x_shape[1], x_shape[2]))
        ai = ops.exp(logits - ops.max(logits, axis=[1, 2], keepdims=True))

        att_weights = ai / (ops.sum(ai, axis=[1, 2], keepdims=True) + backend.epsilon())
        weighted_input = x * ops.expand_dims(att_weights)
        result = ops.sum(weighted_input, axis=[1, 2])
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[3]
        return (input_shape[0], output_len)


def cbam_block(base_out, ratio=1):
    # Neck
    can_module = ChannelAttentionModule()
    san_module = StripPooling()
    # san_module_x = SpatialAttentionModule()
    # san_module_y = SpatialAttentionModule()
    # awn_module = AttentionWeightedAverage2D()

    # Attention Modules 1
    # Channel Attention + Spatial Attention
    canx = can_module(base_out) * base_out
    spnx = san_module(canx)
    # spny   = san_module_y(canx)

    # Global Weighted Average Pooling
    # gapx   = keras.layers.GlobalAveragePooling2D()(spnx)
    # wvgx   = keras.layers.GlobalAveragePooling2D()(spny)
    # gapavg = keras.layers.Average()([gapx, wvgx])

    # Attention Modules 2
    # Attention Weighted Average (AWG)
    # awgavg = awn_module(base_out)
    # Summation of Attentions
    # attns_adds = keras.layers.Add()([gapavg, awgavg])

    return spnx


# https://openaccess.thecvf.com/content_CVPR_2020/papers/Hou_Strip_Pooling_Rethinking_Spatial_Pooling_for_Scene_Parsing_CVPR_2020_paper.pdf
class StripPooling(layers.Layer):
    def __init(self, **kwargs):
        super(StripPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.expand_vertical = layers.Conv2D(
            input_shape[-1], (1, 1), use_bias=False, kernel_initializer="he_normal"
        )
        self.expand_horizontal = layers.Conv2D(
            input_shape[-1], (1, 1), use_bias=False, kernel_initializer="he_normal"
        )
        self.fuse = layers.Add()
        self.conv1x1 = layers.Conv2D(
            input_shape[-1], (1, 1), use_bias=False, kernel_initializer="he_normal"
        )
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, inputs):
        # Vertical pooling
        pooled_vertical = ops.mean(
            inputs, axis=2, keepdims=True
        )  # Shape: (batch_size, H, 1, channels)
        pooled_vertical = self.expand_vertical(pooled_vertical)

        # Horizontal pooling
        pooled_horizontal = ops.mean(
            inputs, axis=1, keepdims=True
        )  # Shape: (batch_size, 1, W, channels)
        pooled_horizontal = self.expand_horizontal(pooled_horizontal)

        # Fuse the pooled tensors
        fused = self.fuse([pooled_vertical, pooled_horizontal])

        fused = self.conv1x1(fused)
        fused = self.sigmoid(fused)

        # Multiply with the original input
        output = inputs * fused

        return output
