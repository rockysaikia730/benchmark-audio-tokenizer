import os
import sys
import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, Optional, Any, Union
import logging
import yaml
import time

import torch._dynamo._trace_wrapped_higher_order_op as trace_module

class DummyTransformGetItemToIndex:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs):
        if args and callable(args[0]): return args[0]
        return self
trace_module.TransformGetItemToIndex = DummyTransformGetItemToIndex


import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

hcodec_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repos', 'unified-audio-masked', 'QuarkAudio-HCodec','HCodec-1.5')
sys.path.insert(0, hcodec_path)

from audio_tokenizer import HCodecTokenizer

from ..base import BaseAudioTokenizer
logger = logging.getLogger(__name__)

class HCodecWrapper(BaseAudioTokenizer):
    name = "hcodec"

    def _load_model(self):
        try:
            cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/audio_codec/hcodec/"
            os.makedirs(cache_dir, exist_ok=True)

            model_name = "hcode_1.5_adaptive_4+4.pt"
            model_path = os.path.join(cache_dir,model_name)
            if not os.path.exists(model_path):
                logger.info(f"Downloading HCodec_1.5 from HuggingFace...")
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id="QuarkAudio/HCodec-1.5-adaptive",
                    filename=model_name,
                    cache_dir=cache_dir,
                    local_dir=cache_dir
                )

            config_path = os.path.join(hcodec_path,"conf","config_adaptive_v3.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            config["ckpt_path"] = model_path
            self.model = HCodecTokenizer(config=config).to(self.device)
        except Exception as e:
            logger.error(f"Error loading HCodec Tokenizer : {e}")
            raise     

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate for the decoder."""
        return 16000  # WavTokenizer outputs at 24kHz

    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 16000:
            logger.warning(f"WavTokenizer uses fixed 24kHz output rate, ignoring requested {value}Hz")

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return (4*1024+4*512)  # Acoustic uses 1024 size codebook of 4 layers
                               # Semantic uses 512 size codebook of 4 layers

    @property
    def downsample_rate(self) -> int:
        """Downsampling rate from audio samples to tokens."""
        return 16000/19.54  # (~19.54 frames per second per layer)


    def encode_audio(self, audio: torch.Tensor, wav_lengths: torch.Tensor = None) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        audio = audio.to(self.device)
        
        codes = self.model.tokenize(audio, wav_lengths=wav_lengths)
        # merged_codes = torch.cat((codes['acoustic_codes'], codes['semantic_codes']), dim=0)

        merged_codes = torch.cat((codes['acoustic_codes'].unsqueeze(1), 
                                codes['semantic_codes'].unsqueeze(1)), dim=1)

        return merged_codes
    
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        
        tokens = tokens.to(self.device) 
        acoustic_codes, semantic_codes = torch.chunk(tokens, chunks=2, dim=1)

        # codes = {"acoustic_codes": acoustic_codes,
        #         "semantic_codes": semantic_codes}
        codes = {"acoustic_codes": acoustic_codes.squeeze(dim=1),
                "semantic_codes": semantic_codes.squeeze(dim=1)}
        audio = self.model.detokenize(**codes)
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        return audio
    
    def encode(self,
              audio: Union[np.ndarray, torch.Tensor, str],
              sr: Optional[int] = None, 
              wav_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full encoding pipeline with preprocessing.
        
        Returns:
            tokens: Encoded tokens
            info: Encoding information
        """
        # Preprocess
        audio_tensor = self.preprocess_audio(audio, sr)
        
        # Encode
        start_time = time.time()
        with torch.no_grad():
            tokens = self.encode_audio(audio_tensor, wav_lengths=wav_lengths)
        encode_time = time.time() - start_time
        
        # Info
        info = {
            "encode_time": encode_time,
            "input_shape": list(audio_tensor.shape),
            "token_shape": list(tokens.shape),
            "num_tokens": tokens.numel(),
        }
        
        return tokens, info