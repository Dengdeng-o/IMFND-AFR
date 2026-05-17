import os

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from cn_clip.clip import load_from_name

import models_mae
from utils.utils import clipdata2gpu_missing, Averager, Recorder, metricsTrueFalse

from .layers import (
    cnn_extractor,
    MaskAttention,
    TokenAttention,
    MLP_fusion,
    clip_fusion,
    ExpertGate,
    MissingModalityHandler,
    InfoNCELoss,
    get_feature_kernel,
)


MODEL_NAME = "IMFND-AFR"
PARAMETER_FILE_NAME = "parameter_imfnd_afr.pkl"


class IMFND_AFR(nn.Module):
    """IMFND-AFR main model."""

    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert,
        dropout,
        num_expert=6,
        num_domains=9,
    ):
        super().__init__()

        self.num_expert = num_expert
        self.num_domains = num_domains
        self.emb_dim = emb_dim
        self.text_dim = 768
        self.image_dim = 768
        self.clip_dim = 512
        self.expert_out_dim = 320
        self.domain_emb_dim = 64

        feature_kernel = get_feature_kernel()

        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        self.image_model = models_mae.__dict__["mae_vit_base_patch16"](
            norm_pix_loss=False
        )
        checkpoint = torch.load("./mae_pretrain_vit_base.pth", map_location="cpu")
        self.image_model.load_state_dict(checkpoint["model"], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False

        self.clip_model, _ = load_from_name(
            "ViT-B-16",
            device="cpu",
            download_root="./",
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.text_attention = MaskAttention(self.emb_dim)
        self.image_attention = TokenAttention(self.emb_dim)

        self.shared_general_experts = nn.ModuleList(
            [
                cnn_extractor(self.text_dim, feature_kernel)
                for _ in range(self.num_expert)
            ]
        )
        self.text_general_gate = ExpertGate(self.emb_dim, self.num_expert, dropout)
        self.image_general_gate = ExpertGate(self.emb_dim, self.num_expert, dropout)

        self.text_specific_experts = nn.ModuleList(
            [
                cnn_extractor(self.text_dim, feature_kernel)
                for _ in range(self.num_expert)
            ]
        )
        self.text_specific_gate = ExpertGate(self.emb_dim, self.num_expert, dropout)

        self.image_specific_experts = nn.ModuleList(
            [
                cnn_extractor(self.image_dim, feature_kernel)
                for _ in range(self.num_expert)
            ]
        )
        self.image_specific_gate = ExpertGate(self.emb_dim, self.num_expert, dropout)

        self.missing_handler = MissingModalityHandler(
            expert_out_dim=self.expert_out_dim,
            num_domains=self.num_domains,
            domain_emb_dim=self.domain_emb_dim,
            clip_dim=self.clip_dim,
            dropout=dropout,
        )

        self.clip_fusion_layer = clip_fusion(
            input_dim=self.clip_dim * 2,
            output_dim=self.expert_out_dim,
            hidden_dims=[384],
            dropout=dropout,
        )

        self.fusion_layer = MLP_fusion(
            input_dim=self.expert_out_dim * 3,
            output_dim=self.expert_out_dim,
            hidden_dims=[512],
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.expert_out_dim * 3, mlp_dims[0]),
            nn.BatchNorm1d(mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims[0], 1),
            nn.Sigmoid(),
        )

        self.infonce_loss = InfoNCELoss(temperature=0.07)
        self.lambda_align = 0.1
        self.lambda_rec = 0.2

    def extract_general_features(self, feature, gate_input, is_text=True):
        gate_weights = (
            self.text_general_gate(gate_input)
            if is_text
            else self.image_general_gate(gate_input)
        )

        e_gen = 0
        for expert_idx in range(self.num_expert):
            expert_out = self.shared_general_experts[expert_idx](feature)
            expert_weight = gate_weights[:, expert_idx].unsqueeze(1)
            e_gen = e_gen + expert_out * expert_weight

        return e_gen

    def extract_specific_features(self, feature, gate_input, is_text=True):
        if is_text:
            gate_weights = self.text_specific_gate(gate_input)
            experts = self.text_specific_experts
        else:
            gate_weights = self.image_specific_gate(gate_input)
            experts = self.image_specific_experts

        e_spe = 0
        for expert_idx in range(self.num_expert):
            expert_out = experts[expert_idx](feature)
            expert_weight = gate_weights[:, expert_idx].unsqueeze(1)
            e_spe = e_spe + expert_out * expert_weight

        return e_spe

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        image = kwargs["image"]
        clip_text_input = kwargs["clip_text"]
        clip_image_input = kwargs["clip_image"]
        missing = kwargs["missing"]
        domain_label = kwargs["category"]

        complete_mask = missing == 1

        text_feature = self.bert(content, attention_mask=content_masks)[0]
        image_feature = self.image_model.forward_ying(image)

        with torch.no_grad():
            clip_text_feature = self.clip_model.encode_text(clip_text_input)
            clip_image_feature = self.clip_model.encode_image(clip_image_input)
            clip_text_feature = clip_text_feature / clip_text_feature.norm(
                dim=-1, keepdim=True
            )
            clip_image_feature = clip_image_feature / clip_image_feature.norm(
                dim=-1, keepdim=True
            )

        text_atn = self.text_attention(text_feature, content_masks)
        image_atn, _ = self.image_attention(image_feature)

        e_text_gen = self.extract_general_features(text_feature, text_atn, is_text=True)
        e_text_spe = self.extract_specific_features(text_feature, text_atn, is_text=True)
        e_image_gen = self.extract_general_features(
            image_feature, image_atn, is_text=False
        )
        e_image_spe = self.extract_specific_features(
            image_feature, image_atn, is_text=False
        )

        (
            e_image_gen_final,
            e_image_spe_final,
            clip_image_final,
            rec_image_gen,
            rec_image_spe,
            rec_clip_image,
        ) = self.missing_handler(
            e_text_gen=e_text_gen,
            e_text_spe=e_text_spe,
            e_image_gen=e_image_gen,
            e_image_spe=e_image_spe,
            clip_text=clip_text_feature.float(),
            clip_image=clip_image_feature.float(),
            domain_label=domain_label,
            missing=missing,
        )

        text_enhanced = e_text_gen + e_text_spe
        image_enhanced = e_image_gen_final + e_image_spe_final

        clip_combined = torch.cat(
            [clip_text_feature.float(), clip_image_final],
            dim=-1,
        )
        clip_fused = self.clip_fusion_layer(clip_combined)

        fusion_input = torch.cat(
            [text_enhanced, image_enhanced, clip_fused],
            dim=-1,
        )
        fusion_feature = self.fusion_layer(fusion_input)

        final_concat = torch.cat(
            [text_enhanced, image_enhanced, fusion_feature],
            dim=-1,
        )
        pred = self.classifier(final_concat).squeeze(-1)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        return (
            pred,
            e_text_gen,
            e_text_spe,
            e_image_gen,
            e_image_spe,
            rec_image_gen,
            rec_image_spe,
            rec_clip_image,
            clip_image_feature,
            complete_mask,
        )


class Trainer:
    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert,
        use_cuda,
        lr,
        dropout,
        train_loader,
        val_loader,
        test_loader,
        category_dict,
        weight_decay,
        save_param_dir,
        early_stop=5,
        epochs=100,
        lambda_align=0.1,
        lambda_rec=0.2,
        lambda_ortho=0.01,
        num_domains=9,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epochs = epochs
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.lambda_align = lambda_align
        self.lambda_rec = lambda_rec
        self.lambda_ortho = lambda_ortho

        os.makedirs(save_param_dir, exist_ok=True)
        self.save_param_dir = save_param_dir
        self.parameter_path = os.path.join(self.save_param_dir, PARAMETER_FILE_NAME)

        self.model = IMFND_AFR(
            emb_dim=emb_dim,
            mlp_dims=mlp_dims,
            bert=bert,
            dropout=dropout,
            num_domains=num_domains,
        )
        if use_cuda:
            self.model = self.model.cuda()

        self.model.lambda_align = lambda_align
        self.model.lambda_rec = lambda_rec

        self.loss_fn = nn.BCELoss()
        self.infonce_loss = InfoNCELoss(temperature=0.07)

    def train(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.98,
        )
        recorder = Recorder(self.early_stop)

        for epoch in range(self.epochs):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for _, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu_missing(batch)
                label = batch_data["label"]

                (
                    pred,
                    e_text_gen,
                    e_text_spe,
                    e_image_gen,
                    e_image_spe,
                    rec_image_gen,
                    rec_image_spe,
                    rec_clip_image,
                    clip_image,
                    complete_mask,
                ) = self.model(**batch_data)

                loss_cls = self.loss_fn(pred, label.float())

                if complete_mask.sum() > 1:
                    loss_align_complete = self.infonce_loss(
                        e_text_gen[complete_mask],
                        e_image_gen[complete_mask],
                    )
                else:
                    loss_align_complete = torch.tensor(0.0, device=label.device)

                missing_mask = ~complete_mask
                if missing_mask.sum() > 1:
                    loss_align_missing = self.infonce_loss(
                        e_text_gen[missing_mask],
                        rec_image_gen[missing_mask],
                    )
                else:
                    loss_align_missing = torch.tensor(0.0, device=label.device)

                loss_align = loss_align_complete + loss_align_missing

                if complete_mask.sum() > 0:
                    loss_rec_general = F.mse_loss(
                        rec_image_gen[complete_mask],
                        e_image_gen[complete_mask].detach(),
                    )
                    loss_rec_specific = F.mse_loss(
                        rec_image_spe[complete_mask],
                        e_image_spe[complete_mask].detach(),
                    )
                    loss_rec_clip = F.mse_loss(
                        rec_clip_image[complete_mask],
                        clip_image[complete_mask].detach(),
                    )
                    loss_rec = loss_rec_general + loss_rec_specific + loss_rec_clip
                else:
                    loss_rec = torch.tensor(0.0, device=label.device)

                text_gen_norm = F.normalize(e_text_gen, dim=1)
                text_spe_norm = F.normalize(e_text_spe, dim=1)
                loss_ortho_text = (text_gen_norm * text_spe_norm).sum(dim=1).abs().mean()

                if complete_mask.sum() > 0:
                    image_gen_norm = F.normalize(e_image_gen[complete_mask], dim=1)
                    image_spe_norm = F.normalize(e_image_spe[complete_mask], dim=1)
                    loss_ortho_image = (
                        image_gen_norm * image_spe_norm
                    ).sum(dim=1).abs().mean()
                else:
                    loss_ortho_image = torch.tensor(0.0, device=label.device)

                loss_ortho = loss_ortho_text + loss_ortho_image

                loss = (
                    loss_cls
                    + self.lambda_align * loss_align
                    + self.lambda_rec * loss_rec
                    + self.lambda_ortho * loss_ortho
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                avg_loss.add(loss.item())

            print(f"Training Epoch {epoch + 1}; Loss {avg_loss.item()};")
            print("----- self.save_param_dir", self.save_param_dir)

            results = self.test(self.val_loader)
            mark = recorder.add(results)

            if mark == "save":
                torch.save(self.model.state_dict(), self.parameter_path)
            elif mark == "esc":
                break

        self.model.load_state_dict(torch.load(self.parameter_path))
        print("开始进行最后的测试")
        results = self.test(self.test_loader)
        print("final: ", results)

        return results, self.parameter_path

    def test(self, dataloader):
        pred = []
        label = []
        category = []

        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for _, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu_missing(batch)
                batch_label = batch_data["label"]
                batch_category = batch_data["category"]

                batch_label_pred, _, _, _, _, _, _, _, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res
