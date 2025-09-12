import torch
import torch.nn as nn

from src.model.ecg_encoder import ProbXResNet1D, prob_xresnet1d101
from src.model.text_encoder import ProbEncoderTextBert

MODALITY_KEYS: list[str] = ["ecg", "text"]


class EchoingECG(nn.Module):
    def __init__(self, model_cfg: dict, **kwargs: dict) -> None:
        super().__init__()
        ecgencoder_cfg = model_cfg.get("ecg_encoder")
        textencoder_cfg = model_cfg.get("text_encoder")

        assert ecgencoder_cfg is not None, "make sure ecg encoder cfg is present"
        assert textencoder_cfg is not None, "make sure text encoder cfg is present"
        self.ecg_encoder: ProbXResNet1D = prob_xresnet1d101(**ecgencoder_cfg)
        self.text_encoder: ProbEncoderTextBert = ProbEncoderTextBert(**textencoder_cfg)
        self.embed_size = model_cfg.get("embed_size")
        assert self.embed_size == self.ecg_encoder.embed_size == self.text_encoder.embed_size, (
            "make sure embeddings are consistent between encoders!"
        )

        self.encode_dict = {"ecg": self.encode_ecg, "text": self.encode_text}

    @property
    def get_embed_size(self) -> int:
        return self.embed_size

    def encode_ecg(self, ecg, **kwargs) -> dict[str, torch.Tensor]:
        return self.ecg_encoder(ecg)

    def encode_text(self, text, attention_mask, **kwargs) -> dict[str, torch.Tensor]:
        try:
            lengths = attention_mask.squeeze().sum(dim=1)
        except:
            lengths = attention_mask.sum(dim=1)
        assert lengths is not None, "make sure to pass attention_mask in dict"
        return self.text_encoder(text, attention_mask=attention_mask, lengths=lengths)

    def pass_embeddings(self, modality, **kwargs) -> dict[str, torch.Tensor]:
        ENCODE_FN = self.encode_dict[modality]
        return ENCODE_FN(**kwargs)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = {}
        assert isinstance(inputs, dict), "Inputs must be a dictionary of modality tensors"
        for modality_key, modality_value in inputs.items():
            if modality_key in MODALITY_KEYS:
                output = self.pass_embeddings(modality_key, **inputs)
                outputs[modality_key] = output
        return outputs
