import os

import torch
from transformers import pipeline
from transformers import AutoModel
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from peft import PeftModel, PeftConfig

from src.audio_utils import save_audio_to_file

from .asr_interface import ASRInterface


class AdmedVoiceASR(ASRInterface):


    def __init__(self, **kwargs):
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda:0")

        peft_model_id = kwargs.get("peft_model", "/workspace/PEFT/checkpoint-1100")
        language = "Polish"
        task = "transcribe"
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path #, load_in_8bit=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, peft_model_id)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        #forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            tokenizer=tokenizer, 
            feature_extractor=feature_extractor
        )

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        if client.config["language"] is not None:
            to_return = self.asr_pipeline(
                file_path,
                generate_kwargs={"language": client.config["language"]},
            )["text"]
        else:
            to_return = self.asr_pipeline(file_path)["text"]

        os.remove(file_path)

        to_return = {
            "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            "language_probability": None,
            "text": to_return.strip(),
            "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return
