# Prediction interface for Cog ⚙️
# https://cog.run/python
from argparse import Namespace

from cog import BasePredictor, Input, Path

from tasks.eval.eval_utils import conv_templates, ChatPllava
from tasks.eval.model_utils import load_pllava


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        print('Initializing PLLaVA...')
        args = Namespace(
            pretrained_model_name_or_path='llava-v1.6-34b-hf',
            num_frames=4,
            use_lora=False,
            weight_dir=None,
            use_multi_gpus=False,
            lora_alpha=None,
            conv_mode='plain'
        )
        self.model, self.processor = load_pllava(
            args.pretrained_model_name_or_path,
            args.num_frames,
            use_lora=args.use_lora,
            weight_dir=args.weight_dir,
            lora_alpha=args.lora_alpha,
            use_multi_gpus=args.use_multi_gpus
        )
        if not args.use_multi_gpus:
            self.model = self.model.to('cuda')
        self.chat = ChatPllava(self.model, self.processor)
        self.conv = conv_templates[args.conv_mode].copy()
        self.img_list = []

    def predict(
            self,
            video: Path = Input(description="Input video file"),
            query: str = Input(description="User query"),
            num_beams: int = Input(description="Beam search numbers", default=1, ge=1, le=5),
            temperature: float = Input(description="Prediction temperature", default=1.0, ge=0.1, le=2.0)
    ) -> str:
        """Run a single prediction on the model"""
        system_prompt = ("You are a powerful Video Magic ChatBot, a large vision-language assistant. You are able to "
                         "understand the video content that the user provides and assist the user in a video-language"
                         " related task.")
        system_prompt += ("The user might provide you with the video and maybe some extra noisy information to help "
                          "you out or ask you a question. Make use of the information in a proper way to be competent "
                          "for the job. ### INSTRUCTIONS: 1. Follow the user's instruction. 2. Be critical yet "
                          "believe in yourself.")

        msg, self.img_list, self.conv = self.chat.upload_video(video, self.conv, self.img_list)

        self.conv = self.chat.ask(query, self.conv, system_prompt)

        output_text, _, self.conv = self.chat.answer(
            conv=self.conv,
            img_list=self.img_list,
            max_new_tokens=200,
            num_beams=num_beams,
            temperature=temperature
        )

        return output_text
