import torch.nn as nn

class ConvPoolPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn = nn.PReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2),  # 100
            nn.BatchNorm2d(out_channels),
            activ_fn()
        )

    def forward(self, x):
        return self.model(x)

    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2.0)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn=nn.PReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activ_fn(),
        )

    def forward(self, x):
        return self.model(x)

    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2.0)


class AutoEncoder(nn.Module):
    def __init__(self, hparams=None, input_channels=1):
        super().__init__()
        self.hparams = hparams

        ## Parameters
        f_channels = hparams["filter_channels"]
        f = hparams["filter_size"]
        activ_fn = nn.PReLU
        drop_out = hparams["dropout"]

        padding = 1 if f == 3 else 0
        padding = 2 if f == 5 else padding

        self.model = nn.Sequential(
            # Decoder
            ConvPoolPRelu(input_channels, f_channels, f, padding, activ_fn = activ_fn),
            ConvPoolPRelu(f_channels, f_channels*2, f, padding, activ_fn = activ_fn),
            ConvPoolPRelu(f_channels*2, f_channels*4, f, padding, activ_fn = activ_fn),
            ConvPoolPRelu(f_channels*4, f_channels*8, f, padding, activ_fn=activ_fn),

            # Encoder
            DecoderBlock(f_channels * 8, f_channels * 4, f, padding, activ_fn=activ_fn),
            DecoderBlock(f_channels * 4, f_channels * 2, f, padding, activ_fn = activ_fn),
            DecoderBlock(f_channels * 2, f_channels, f, padding, activ_fn = activ_fn),
            DecoderBlock(f_channels, input_channels, f, padding, activ_fn = activ_fn)
        )

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.to('cuda')
        return self.model(x)