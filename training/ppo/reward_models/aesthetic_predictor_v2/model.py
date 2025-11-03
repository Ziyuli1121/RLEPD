import clip
import torch
import torch.nn as nn


def torch_normalized(a, axis=-1, order=2):
    l2 = torch.norm(a, dim=axis, p=order, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


class AestheticV2Model(nn.Module):
    def __init__(self, clip_path=None, predictor_path=None):
        super(AestheticV2Model, self).__init__()

        self.clip_encoder, self.preprocessor = clip.load(clip_path) if clip_path else clip.load("ViT-L/14")
        state_dict = torch.load(predictor_path, map_location="cpu")
        modified_state_dict = {}
        for key in state_dict.keys():
            modified_state_dict[key[7:]] = state_dict[key]
        self.aesthetic_predictor = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.aesthetic_predictor.load_state_dict(modified_state_dict)

    def forward(self, x):
        x = torch.stack([self.preprocessor(image) for image in x], dim=0).cuda()
        x = self.clip_encoder.encode_image(x)
        x = torch_normalized(x).to(torch.float32)
        x = self.aesthetic_predictor(x)
        return x


if __name__ == "__main__":
    model = AestheticV2Model(
        clip_path="/mnt/sda/models/clip/ViT-L-14.pt",
        predictor_path="/mnt/sda/models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    ).cuda()
