from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.selfies_ted.load import SELFIES
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from huggingface_hub import PyTorchModelHubMixin, snapshot_download

class Projector(nn.Module):
    def __init__(self, config):
        super(Projector, self).__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config["input_dim"][0]*self.config["input_dim"][1], self.config["hidden_dim"])
        self.layer2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.layer3 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.layer4 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"])
        self.activation = F.rrelu

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)

        return x

class InverseProjector(nn.Module):
    def __init__(self, config):
        super(InverseProjector, self).__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.layer2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.layer3 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.layer4 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"][0]*self.config["output_dim"][1])
        self.activation = F.rrelu

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        x = x.reshape(len(x), self.config["output_dim"][0], self.config["output_dim"][1])
        if x.shape[2] == 1:
            x = x.reshape(x.shape[0], x.shape[1])
        return x

'''
SELFIES-TED model with projections to a single 128-dimensional latent vector for a SELFIES string
'''    
class SELFIESForEmbeddingExploration(SELFIES, PyTorchModelHubMixin):
    
    def __init__(self) -> None:
        super().__init__()
        config_proj = {"input_dim": [64, 256],
                         "hidden_dim":  512,
                         "output_dim":128}
        self.projector = Projector(config_proj)
        
        config_invproj = {"input_dim": 128,
                            "hidden_dim": 512,
                            "output_dim": [64, 256]}
        self.invprojector = InverseProjector(config_invproj)

    def load(self, checkpoint="ibm/materials.selfies-ted2m"):
        """
            inputs :
                   checkpoint - HuggingFace repo or path to directory containing checkpoints
        """

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.lm = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model = self.lm.model

    def load_projectors(self, checkpoint_dir: str) -> None:
        self.projector.load_state_dict(torch.load(os.path.join(checkpoint_dir, "projector_2023-05-25.pth")))
        self.invprojector.load_state_dict(torch.load(os.path.join(checkpoint_dir, "invprojector_2023-05-25.pth")))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
       
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model.load(model_id)
            model.load_projectors(model_id)
        else:
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            model.load(model_dir)
            model.load_projectors(model_dir)
           
        return model

    def hidden_state_to_latent_vector(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_state).cpu().detach()
    
    def latent_vector_to_hidden_state(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.invprojector(latent_vector).cpu().detach()
    
    def get_encoder_last_hidden_state(self, selfies):
        encoding = self.tokenizer(selfies, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        outputs = self.model.encoder(input_ids=input_ids)
        return outputs.last_hidden_state
    
    def get_latent_vector(self, selfies: List[str]) -> torch.Tensor:
        return self.hidden_state_to_latent_vector(self.get_encoder_last_hidden_state(selfies))

    def decode(self, embedding):
        hidden_state = self.latent_vector_to_hidden_state(embedding)
        encoder_outputs = BaseModelOutput(hidden_state)
        decoder_output = self.lm.generate(encoder_outputs=encoder_outputs, max_new_tokens=64, do_sample=True,  top_k=1, top_p=0.95, num_return_sequences=1)
        return self.tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
        