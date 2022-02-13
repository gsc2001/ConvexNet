from dataclasses import dataclass


@dataclass
class BaseConfig(object):
    batch_size: int = 4096
    lr: float = 1e-3
    n_epochs: int = 100
    num_classes: int = 100


@dataclass
class MlpMixerConfig(BaseConfig):
    img_size: int = 224
    num_mixer_layers: int = 4
    hidden_size: int = 128
    mlp_seq_dim: int = 64
    mlp_channel_dim: int = 128
    patch_size: int = 16

    # regularization and augmentation
    weight_decay: float = 0.1

    rand_aug_num_ops: int = 2
    rand_aug_magnitude: int = 15

    mixup_strength: float = .5

    drop_path_s: float = 0.1


Mixer_B_config = MlpMixerConfig(n_epochs=300, num_classes=1000, num_mixer_layers=12, hidden_size=768,
                                mlp_channel_dim=3072, mlp_seq_dim=384)

Mixer_B_16_config = MlpMixerConfig(**Mixer_B_config.__dict__)
Mixer_B_16_config.patch_size = 16
Mixer_B_32_config = MlpMixerConfig(**Mixer_B_config.__dict__)
Mixer_B_32_config.patch_size = 32

Mixer_L_config = MlpMixerConfig(n_epochs=300, num_classes=1000, num_mixer_layers=24, hidden_size=1024,
                                mlp_channel_dim=4096, mlp_seq_dim=512)

Mixer_L_16_config = MlpMixerConfig(**Mixer_L_config.__dict__)
Mixer_L_16_config.patch_size = 16

Mixer_L_32_config = MlpMixerConfig(**Mixer_L_config.__dict__)
Mixer_L_32_config.patch_size = 32
