class AHead(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        in_features = self.arg.d_model + self.arg.seq_len // 2 + 1
        out_features = self.arg.spectrum_size // 2 + 1
        self.amplitude_head = nn.Sequential(
            nn.Linear(in_features, self.arg.d_ff),
            nn.GELU(),
            nn.Dropout(self.arg.dropout),
            nn.Linear(self.arg.d_ff, out_features)
        )
        self.activation = ALU(w=0.5)
        # self._get_spectrum_prior()

    def _get_spectrum_prior(self):
        train_data = TSFactory(self.arg)('train')[0].x
        spectrum_prior = torch.zeros(1, self.arg.enc_in, self.arg.spectrum_size // 2 + 1)
        for i in range(len(train_data) - self.arg.spectrum_size):
            x = train_data[i:i + self.arg.spectrum_size].unsqueeze(0)
            if self.arg.revin:
                dim = tuple(range(1, x.ndim - 1))
                means = torch.mean(x, dim=dim, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(x, dim=dim, correction=0, keepdim=True) + self.arg.eps).detach()
                x = (x - means) / stdev
            spectrum_prior += torch.abs(torch.fft.rfft(x.permute(0, 2, 1), norm=self.arg.fourier_norm))
        self.spectrum_prior = spectrum_prior / (len(train_data) - self.arg.spectrum_size)

    def forward(self, x):
        # spectrum_prior = self.spectrum_prior.to(x.device).repeat(x.shape[0], 1, 1)
        amplitude = self.activation(self.amplitude_head(x))
        return amplitude


class PHead(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        in_features = self.arg.d_model + self.arg.seq_len // 2 + 1
        out_features = self.arg.spectrum_size // 2 + 1
        self.sin_head = nn.Sequential(
            nn.Linear(in_features, self.arg.d_ff),
            nn.GELU(),
            nn.Dropout(self.arg.dropout),
            nn.Linear(self.arg.d_ff, out_features),
            nn.Tanh()
        )
        self.cos_head = nn.Sequential(
            nn.Linear(in_features, self.arg.d_ff),
            nn.GELU(),
            nn.Dropout(self.arg.dropout),
            nn.Linear(self.arg.d_ff, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        sin = self.sin_head(x)
        cos = self.cos_head(x)
        phase = torch.atan2(sin, cos)
        return phase


class ImplicitForecaster(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.a_head = AHead(self.arg)
        self.p_head = PHead(self.arg)

    def forward(self, x_enc, x):
        fft_x = torch.fft.rfft(x.permute(0, 2, 1), norm=self.arg.fourier_norm)
        amp_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        amp_out = self.a_head(torch.cat((x_enc, amp_x), dim=-1))
        pha_out = self.p_head(torch.cat((x_enc, pha_x), dim=-1))
        x = torch.fft.irfft(amp_out * torch.exp(1j * pha_out), norm=self.arg.fourier_norm)
        return x.permute(0, 2, 1)
