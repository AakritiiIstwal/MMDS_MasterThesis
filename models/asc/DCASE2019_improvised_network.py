import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer


class PadChannels(Layer):
    def __init__(self, desired_channels, **kwargs):
        super(PadChannels, self).__init__(**kwargs)
        self.desired_channels = desired_channels

    def call(self, inputs):
        input_shape = inputs.shape
        current_channels = input_shape[-1]
        padding_size = self.desired_channels - current_channels
        # print("Padding size: ", padding_size)
        if padding_size > 0:
            padding = tf.zeros_like(inputs)[:, :, :, :padding_size]
            return tf.concat([inputs, padding], axis=-1)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        if input_shape[-1] < self.desired_channels:
            # print("Output shape:", input_shape[:-1] + (self.desired_channels,))
            return input_shape[:-1] + (self.desired_channels,)
        return input_shape

    def get_config(self):
        config = super(PadChannels, self).get_config()
        config.update({"desired_channels": self.desired_channels})
        return config


class SliceLayer(Layer):
    def __init__(self, start, end, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[:, self.start : self.end, :, :]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({"start": self.start, "end": self.end})
        return config


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    learn_bn=True,
    wd=1e-4,
    use_relu=True,
):
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(inputs)
    if use_relu:
        x = Activation("relu")(x)
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(wd),
        use_bias=False,
    )(x)
    return x


def model_resnet_new(
    num_classes, input_shape, initial_filters=24, depth=4, num_blocks=2, wd=1e-3
):
    inputs = Input(shape=input_shape)
    # splits = [inputs[:,0:64,:,:], inputs[:,64:128,:,:]]
    split1 = SliceLayer(start=0, end=64)(inputs)
    split2 = SliceLayer(start=64, end=128)(inputs)
    splits = [split1, split2]

    outputs = []
    for split in splits:
        x = resnet_layer(
            split, num_filters=initial_filters, strides=[1, 2], wd=wd, use_relu=False
        )
        for stack in range(depth):
            for block in range(num_blocks):
                strides = [1, 2] if stack > 0 and block == 0 else 1
                y = resnet_layer(
                    x,
                    num_filters=initial_filters,
                    strides=strides,
                    wd=wd,
                    use_relu=True,
                )
                y = resnet_layer(
                    y, num_filters=initial_filters, strides=1, wd=wd, use_relu=True
                )
                if stack > 0 and block == 0:
                    x = AveragePooling2D(
                        pool_size=(3, 3), strides=[1, 2], padding="same"
                    )(x)
                    desired_channels = y.shape[-1]
                    x = PadChannels(desired_channels)(x)
                x = Add()([x, y])
            initial_filters *= 2
        outputs.append(x)

    combined = Concatenate(axis=-1)(outputs)
    x = resnet_layer(
        combined, num_filters=initial_filters * 2, kernel_size=1, wd=wd, use_relu=True
    )
    x = resnet_layer(x, num_filters=num_classes, kernel_size=1, wd=wd, use_relu=False) # Reducing the number of classes to 10
    x = BatchNormalization(center=False, scale=False)(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model
