import jax
import jax.numpy as jnp
import flax.linen as nn
import typing as tp


def channel_shuffle(x,groups):
    n,h,w,c = x.shape
    x = x.reshape(n,h,w,groups,c//groups) # group
    x = x.transpose([0,1,2,4,3]).reshape(n,h,w,-1)
    return x


class RequiresTrainingFlag:
    pass

def sequantial_apply(models, x, training: bool):
    for model in models:
        if isinstance(model, RequiresTrainingFlag):
            x = model(x, training=training)
        else:
            x = model(x)
    return x


class ConvBnSiLu(nn.Module, RequiresTrainingFlag):
    out_channels:int
    kernel_size: tp.Tuple[int,int]

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(self.out_channels,kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        return x


class ResidualBottleneck(nn.Module, RequiresTrainingFlag):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    out_channels:int

    @nn.compact
    def __call__(self, x, training: bool):
        n,h,w,c = x.shape

        x_chunk = x.reshape(n,h,w,c//2,2)
        x1, x2 = x_chunk[:, :, :, :, 0], x_chunk[:, :, :, :, 1]

        branch1 = sequantial_apply([
            nn.Conv(c//2, kernel_size=(3, 3), feature_group_count=c//2),
            nn.BatchNorm(use_running_average=not training),
            ConvBnSiLu(self.out_channels//2, kernel_size=(1, 1))
        ], x1, training=training)
        branch2 = sequantial_apply([
            ConvBnSiLu(c//2, kernel_size=(1, 1)),
            nn.Conv(c//2, kernel_size=(3, 3), feature_group_count=c//2),
            nn.BatchNorm(use_running_average=not training),
            ConvBnSiLu(self.out_channels//2, kernel_size=(1, 1))
        ], x2, training=training)

        x = jnp.concatenate([branch1, branch2], axis=-1)
        x = channel_shuffle(x, 2)
        return x


class ResidualDownsample(nn.Module, RequiresTrainingFlag):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''

    out_channels:int

    @nn.compact
    def __call__(self, x, training: bool):
        n,h,w,c = x.shape
        branch1 = sequantial_apply([
            nn.Conv(c,kernel_size=(3, 3), strides=2, padding=1, feature_group_count=c),
            nn.BatchNorm(use_running_average=not training),
            ConvBnSiLu(self.out_channels//2, kernel_size=(1, 1))
        ], x, training=training)
        branch2 = sequantial_apply([
            ConvBnSiLu(self.out_channels//2, kernel_size=(1, 1)),
            nn.Conv(c,kernel_size=(3, 3), strides=2, padding=1, feature_group_count=self.out_channels//2),
            nn.BatchNorm(use_running_average=not training),
            ConvBnSiLu(self.out_channels//2, kernel_size=(1, 1))
        ], x, training=training)
        x = jnp.concatenate([branch1, branch2], axis=-1)
        x = channel_shuffle(x, 2)
        return x


class MLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self,x,t):
        t_emb = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.silu,
            nn.Dense(self.output_dim)
        ])(t)
        x = x + t_emb[:, None, None, :]
        x = nn.silu(x)
        return x

class EncoderBlock(nn.Module, RequiresTrainingFlag):
    out_channels:int

    @nn.compact
    def __call__(self, x, t, training: bool):
        n,h,w,c = x.shape
        x_shortcut = sequantial_apply([
            *[ResidualBottleneck(out_channels=c) for i in range(3)],
            ResidualBottleneck(self.out_channels//2)
        ], x, training=training)
        if t is not None:
            x = MLP(hidden_dim=self.out_channels, output_dim=self.out_channels//2)(
                x, t
            )
        x = ResidualDownsample(self.out_channels)(x, training=training)
        return x, x_shortcut


def upsample(img, HW):
    B, H, W, C = img.shape
    H, W = HW
    return jax.image.resize(img, (B, H, W, C), method='nearest')


class DecoderBlock(nn.Module, RequiresTrainingFlag):
    out_channels: int

    @nn.compact
    def __call__(self, x, x_shortcut, t, training: bool):
        n,h,w,c = x.shape
        N,H,W,C = x_shortcut.shape
        assert n == N and c == C
        x = upsample(x, (H, W))
        x = jnp.concatenate([x, x_shortcut], axis=-1)
        x = sequantial_apply([
            *[ResidualBottleneck(out_channels=c) for i in range(3)],
            ResidualBottleneck(c//2)
        ], x, training=training)
        if t is not None:
            x = MLP(hidden_dim=c, output_dim=c//2)(
                x, t
            )
        x = ResidualBottleneck(self.out_channels//2)(x, training=training)
        return x

class Unet(nn.Module, RequiresTrainingFlag):
    timestep_num: int
    timestep_dim: int
    out_channels: int
    dims: tp.Tuple[int] # = tuple(32 * x for x in [1, 2, 4, 8, 16])
    
    '''
    simple unet design without attention
    '''
    @nn.compact
    def __call__(self, x, t, training: bool):
        assert all(dim % 2 == 0 for dim in self.dims)

        x = ConvBnSiLu(out_channels=self.dims[0], kernel_size=(3, 3))(
            x, training=training
        )
        if t is not None:
            t = nn.Embed(num_embeddings=self.timestep_num, features=self.timestep_dim)(
                t
            )
        encoder_shortcuts = []
        for dim in self.dims[1:]:
            x, x_shortcut = EncoderBlock(out_channels=dim)(
                x, t, training=training
            )
            encoder_shortcuts.append(x_shortcut)

        # mid block
        x = sequantial_apply([
            *[ResidualBottleneck(self.dims[-1]) for i in range(2)],
            ResidualBottleneck(self.dims[-1]//2)
        ], x, training=training)

        for x_shortcut, dim in reversed(list(zip(encoder_shortcuts, self.dims[:-1]))):
            x = DecoderBlock(out_channels=dim)(
                x, x_shortcut, t, training=training
            )
        x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)
        return x


class ConditionalUnet(nn.Module, RequiresTrainingFlag):
    timestep_num: int
    timestep_dim: int
    out_channels: int
    dims: tp.Tuple[int] # = tuple(32 * x for x in [1, 2, 4, 8, 16])
    label_count: int
    label_dim: int

    '''
    simple unet design without attention
    '''
    @nn.compact
    def __call__(self, x, labels, t, training: bool):
        assert all(dim % 2 == 0 for dim in self.dims)

        assert labels.shape == t.shape

        x = ConvBnSiLu(out_channels=self.dims[0], kernel_size=(3, 3))(
            x, training=training
        )
        t = nn.Embed(num_embeddings=self.timestep_num, features=self.timestep_dim)(t)
        labels = nn.Embed(num_embeddings=self.label_count, features=self.label_dim)(labels)
        condition = jnp.concatenate([t, labels], axis=-1)

        encoder_shortcuts = []
        for dim in self.dims[1:]:
            x, x_shortcut = EncoderBlock(out_channels=dim)(
                x, condition, training=training
            )
            encoder_shortcuts.append(x_shortcut)

        # mid block
        x = sequantial_apply([
            *[ResidualBottleneck(self.dims[-1]) for i in range(2)],
            ResidualBottleneck(self.dims[-1]//2)
        ], x, training=training)

        for x_shortcut, dim in reversed(list(zip(encoder_shortcuts, self.dims[:-1]))):
            x = DecoderBlock(out_channels=dim)(
                x, x_shortcut, condition, training=training
            )
        x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)
        return x


if __name__ == '__main__':
    model = Unet(timestep_num=1000, timestep_dim=128, out_channels=1)
    x=jnp.zeros((10, 28, 28, 1)).astype(jnp.float32)
    t=jnp.arange(10)
    params = model.init(jax.random.PRNGKey(0), x=x, t=t, training=False)
    y = model.apply(params, x, t, training=False)
    print(y.shape)
    print("number of paramters:", sum(v.size for v in jax.tree_flatten(params)[0]))
    print("model", model)
