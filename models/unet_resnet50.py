import segmentation_models_pytorch as smp


def build_model(encoder_name, encoder_weights, in_channels, num_classes):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
