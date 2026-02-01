import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return features, pooled

class Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.enc1 = EncoderBlock(input_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_conv = ConvBlock(1024, 1024)
        
    def forward(self, x):
        skip_features = []
        
        # Block 1: 3 -> 64
        features, x = self.enc1(x)  # x: [B, 64, H/2, W/2]
        skip_features.append(features)
        
        # Block 2: 64 -> 128
        features, x = self.enc2(x)  # x: [B, 128, H/4, W/4]
        skip_features.append(features)
        
        # Block 3: 128 -> 256
        features, x = self.enc3(x)  # x: [B, 256, H/8, W/8]
        skip_features.append(features)
        
        # Block 4: 256 -> 512
        features, x = self.enc4(x)  # x: [B, 512, H/16, W/16]
        skip_features.append(features)
        
        # Bottleneck: 512 -> 1024
        x = F.relu(self.bottleneck(x))  # x: [B, 1024, H/16, W/16]
        x = self.bottleneck_conv(x)
        
        return x, skip_features

class AttentionGate(nn.Module):
    """
    Attention Gate блок из статьи
    Аргументы:
        in_channels: число каналов входных признаков (xl)
        gating_channels: число каналов gating signal (g)
        inter_channels: промежуточная размерность
    """
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        # Theta_x: 1×1 conv для входных признаков
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        
        # Phi_g: 1×1 conv для gating signal
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        
        # Psi: 1×1 conv для attention коэффициентов
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.bn = nn.BatchNorm2d(inter_channels)
        self.inter_channels = inter_channels
    
    def forward(self, x, g):
        """
        x: [B, C_in, H, W] - признаки из skip connection (low level)
        g: [B, C_g, H/2, W/2] - gating signal (high level, coarser)
        """
        # 1. Downsampling x до размера g (bilinear)
        x_down = F.interpolate(x, size=g.shape[2:4], mode='bilinear', align_corners=False)
        
        # 2. Linear transformations
        theta_x = self.theta_x(x_down)
        phi_g = self.phi_g(g)
        
        # 3. Additive attention: sum + ReLU
        q = F.relu(theta_x + phi_g, inplace=True)  # q = ReLU(Wx*x + Wg*g + bg)
        q = self.bn(q)
        
        # 4. Compute attention coefficients α
        psi = self.psi(q)  # [B, 1, H/2, W/2]
        alpha = torch.sigmoid(psi)  # α ∈ [0,1]
        
        # 5. Resample α до размера x (bilinear)
        alpha_upsampled = F.interpolate(alpha, size=x.shape[2:4], mode='bilinear', align_corners=False)
        
        # 6. Apply attention: x * α
        out = x * alpha_upsampled
        
        return out

class DecoderBlock(nn.Module):
    """Блок декодера: ConvTranspose + ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # ConvTranspose увеличивает в 2 раза
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        
        # Два сверточных слоя
        self.conv_block = ConvBlock(2*out_channels, out_channels)
        self.attention = AttentionGate(out_channels, in_channels, math.ceil(out_channels/2))
        
    def forward(self, x, skip_features):
        # Увеличиваем размер
        up_x = self.upconv(x)                      # [B, out_channels, H*2, W*2]

        # Блок внимания
        att_out = self.attention(skip_features, x) # [B, out_channels, H*2, W*2]

        # --- выравниваем размеры по высоте и ширине (pad нулями) ---
        _, _, h_up, w_up   = up_x.shape
        _, _, h_att, w_att = att_out.shape

        pad_h = h_up - h_att
        pad_w = w_up - w_att

        if pad_h > 0 or pad_w > 0:
            # дополняем только справа и снизу
            att_out = F.pad(att_out, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        elif pad_h < 0 or pad_w < 0:
            # дополняем up_x, если att_out оказался больше
            up_x = F.pad(up_x, (0, -pad_w, 0, -pad_h), mode='constant', value=0.)

        # Skip-connection: конкатенация
        x = torch.cat([up_x, att_out], dim=1)
        x = self.conv_block(x)
        return x

class Decoder(nn.Module):
    """Декодер из 4 блоков"""
    def __init__(self):
        super().__init__()
        
        # Преобразование 1024 -> 512 перед блоками
        self.preconv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
        # 4 блока: 512 -> 256 -> 128 -> 64
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

    def forward(self, x, skip_features):
        # skip_features: [enc4, enc3, enc2, enc1] (в обратном порядке!)
        
        # Предобработка
        x = F.relu(self.preconv(x))
        
        # Декодер Block 1: 512 -> 256
        x = self.dec1(x, skip_features[3])  # skip от enc4
        
        # Декодер Block 2: 256 -> 128
        x = self.dec2(x, skip_features[2])  # skip от enc3
        
        # Декодер Block 3: 128 -> 64
        x = self.dec3(x, skip_features[1])  # skip от enc2
        
        # Декодер Block 4: 64 -> 64
        x = self.dec4(x, skip_features[0])  # skip от enc1
        
        return x

class SegmentationModel(nn.Module):
    """Полная модель: Encoder + Decoder + финальная свертка"""
    def __init__(self, input_channels=3):
        super().__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder()
        
        # Финальная свертка 64 -> 1 (бинарная маска)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        
        # Энкодер
        bottleneck, skip_features = self.encoder(x)  # skip_features: [enc1, enc2, enc3, enc4]
        # Декодер (skip-connections в обратном порядке!)
        decoded = self.decoder(bottleneck, skip_features)
        
        # Финальная свертка
        output = self.final_conv(decoded)  # [B, 1, H, W]
        
        return output, bottleneck, skip_features

class ScaleEstimator(nn.Module):
    """
    Предсказывает масштаб из bottleneck и skip-connections
    """
    def __init__(self, bottleneck_channels=1024, skip_channels=[64, 128, 256, 512]):
        super().__init__()
        
        # Bottleneck: 1024 -> 512 -> 256
        self.bottleneck_head = nn.Sequential(
                nn.Conv2d(bottleneck_channels, math.ceil(bottleneck_channels / 2), 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(math.ceil(bottleneck_channels / 2), 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((8, 8))
        )
        
        # Skip connections: каждый уровень -> 64 признака
        self.skip_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, math.ceil(ch / 2), 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(math.ceil(ch / 2), 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((8, 8))
            ) for ch in skip_channels
        ])
        
        # Итоговый регрессор: 256 + 4*64 = 512 -> 128 -> 1
        total_features = 256 + 4 * 64
        self.regressor = nn.Sequential(
            nn.Conv2d(total_features, math.ceil(total_features / 2), 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(math.ceil(total_features / 2), math.ceil(total_features / 4), 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(math.ceil(total_features / 4) * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
    def forward(self, bottleneck, skip_features):
        """
        bottleneck: [B, 1024, H/16, W/16]
        skip_features: list of 4 tensors [B, C, H, W] на разных масштабах
        """
        # Обработка bottleneck
        bottle_features = self.bottleneck_head(bottleneck)   # [B, 256]
        
        # Обработка skip connections
        skip_features_list = []
        for i, skip in enumerate(skip_features):
            skip_processed = self.skip_heads[i](skip)  # [B, 64]
            skip_features_list.append(skip_processed)
        
        # Конкатенация
        all_features = torch.cat([bottle_features] + skip_features_list, dim=1)  # [B, 512, 8, 8]

        # Предсказание масштаба
        scale = self.regressor(all_features)  # [B, 1]
        return scale.squeeze(-1)  # [B]

def load_models(device, seg_path: Path, scale_path: Path):
    # Segmentation
    seg = SegmentationModel().to(device)

    seg_ckpt = torch.load(seg_path, map_location=device)
    seg_sd = seg_ckpt["model_state_dict"]


    seg.load_state_dict(seg_sd, strict=True)
    seg.eval()

    # Scale estimator
    scale = ScaleEstimator().to(device)

    scale_ckpt = torch.load(scale_path, map_location=device)
    scale_sd = scale_ckpt["scale_estimator_state_dict"]

    scale.load_state_dict(scale_sd, strict=True)
    scale.eval()

    return seg, scale