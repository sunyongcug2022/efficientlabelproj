
import copy

import lightly.data as data
import pytorch_lightning as pl
import torch
import torchvision
from lightly.loss import PMSNLoss
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead
from lightly.transforms import MSNTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch import nn


class PMSN(pl.LightningModule):
    def __init__(self, mask_ratio=0.15, output_dim=512, prototype_dim=512):
        super().__init__()

        # ViT small configuration (ViT-S/16)
        self.mask_ratio = mask_ratio
        vit = torchvision.models.VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        # or use a torchvision ViT backbone:
        # vit = torchvision.models.vit_b_32(pretrained=False)
        # self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(384, output_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(output_dim, prototype_dim, bias=False).weight
        self.criterion = PMSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views = batch[0]
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(images=targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(images=anchors, idx_keep=idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=1.5e-4)
        return optim



def get_dataloader_ssl(
        configs,
        train_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/train/',
        test_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/test/'):

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


def pretraining_pmsn(configs, dataloader, train_iter=10):
    model = PMSN()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=train_iter, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)
    return model
