import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import BlipPreTrainedModel, BlipTextModel, BlipVisionModel

@dataclass
class BlipAttributeExtractionModelOutput(ModelOutput):
    preds: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    attr_loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    question_embeds: Optional[torch.FloatTensor] = None

class BlipForAttributeExtraction(BlipPreTrainedModel):
    def __init__(self, config, tokenizer, label_encoder):
        super().__init__(config)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(tokenizer))
        
        embed_size = config.text_config.hidden_size
        self.heads = {}
        for cat in label_encoder.vocab.index:
            self.heads[cat] = [nn.Linear(embed_size, n) for n in label_encoder.num_classes(cat)]
        self.post_init()

    def forward(self, batch):
        image_embeds = self.vision_model(pixel_values=batch.pixel_values,
                                         output_attentions=self.config.output_attentions,
                                         output_hidden_states=self.config.output_hidden_states,
                                         interpolate_pos_encoding=False)[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
        
        question_embeds = self.text_encoder(input_ids=batch.input_ids,
                                            attention_mask=batch.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts)[0]
        attr_embeds = question_embeds[:, batch.special_token_mask, :]

        attr_loss = torch.ones(attr_embeds.size()[1], dtype=torch.float32)
        preds = torch.ones(attr_embeds.size()[:-1], dtype=torch.int8)
        
        for i, head in enumerate(self.heads[batch.category]):
            logits = head(attr_embeds[:, i, :])
            preds[:, i] = logits.argmax(dim=1)
            if batch.labels is not None:
                attr_loss[i] = nn.CrossEntropyLoss()(logits.cpu(), batch.labels[:,i])
        loss = attr_loss.mean() if batch.labels is not None else None

        return BlipAttributeExtractionModelOutput(loss=loss, attr_loss=attr_loss,
                                                  preds=preds, image_embeds=image_embeds,
                                                  question_embeds=question_embeds)