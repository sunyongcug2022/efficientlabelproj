import copy

import torch
import torchvision
from torch import nn

from lightly.loss import PMSNLoss
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead
from lightly.transforms import MSNTransform

from lightly.transforms.utils import IMAGENET_NORMALIZE

class PMSN(nn.Module):
    def __init__(self, vit, mask_ratio=0.15, output_dim=512):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.projection_head = MSNProjectionHead(384, output_dim=output_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

    def forward(self, images):
        out = self.backbone(images=images)
        return self.projection_head(out)

    def forward_masked(self, images):
        batch_size, _, _, width = images.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        out = self.anchor_backbone(images=images, idx_keep=idx_keep)
        return self.anchor_projection_head(out)


def get_dataloader_ssl(
        train_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/train/',
        test_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/test/'
):
    preprocess = MSNTransform()

    train_dataset = torchvision.datasets.ImageFolder(
        train_data_root,
        transform=preprocess,
        target_transform=lambda t: 0,
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (configs['input_size'], configs['input_size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ])

    # create a lightly dataset for embedding
    dataset_test = data.LightlyDataset(input_dir=test_data_root,
                                       transform=test_transforms)

    # create a dataloader for embedding
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=configs['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return dataloader_train, dataloader_test


def pretraining_pmsn(configs, dataloader):
    # ViT small configuration (ViT-S/16)
    vit = torchvision.models.VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=384 * 4,
    )

    model = PMSN(vit)
    model.to(device)
    # # or use a torchvision ViT backbone:
    # vit = torchvision.models.vit_b_32(pretrained=False)
    # model = PMSN(vit)

    criterion = PMSNLoss()

    params = [
        *list(model.anchor_backbone.parameters()),
        *list(model.anchor_projection_head.parameters()),
        model.prototypes,
    ]
    optimizer = torch.optim.AdamW(params, lr=1.5e-4)

    print("Starting Training")
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            views = batch[0]
            utils.update_momentum(model.anchor_backbone, model.backbone, 0.996)
            utils.update_momentum(model.anchor_projection_head,
                                  model.projection_head, 0.996)

            views = [view.to(device, non_blocking=True) for view in views]
            targets = views[0]
            anchors = views[1]
            anchors_focal = torch.concat(views[2:], dim=0)

            targets_out = model.backbone(images=targets)
            targets_out = model.projection_head(targets_out)
            anchors_out = model.forward_masked(anchors)
            anchors_focal_out = model.forward_masked(anchors_focal)
            anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

            loss = criterion(anchors_out, targets_out, model.prototypes.data)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
