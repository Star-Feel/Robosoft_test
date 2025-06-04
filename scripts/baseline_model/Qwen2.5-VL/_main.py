import copy
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, QWEN2_5_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss


# @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
# @replace_return_docstrings(
#     output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
# )
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """wefsef
    sdfsdf"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id
                              ).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            n_video_tokens = (input_ids == self.config.video_token_id
                              ).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (
        attention_mask is None or attention_mask.ndim == 2
    ):
        # calculate RoPE index once per generation in the pre-fill stage only
        if ((cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None or
            (past_key_values is None
             or past_key_values.get_seq_length() == 0)):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            if cache_position is not None:
                offset = cache_position[0].to(self.rope_deltas)
            elif past_key_values is not None:
                offset = torch.tensor(past_key_values.get_seq_length()
                                      ).to(self.rope_deltas)
            else:
                offset = 0
            delta = offset + self.rope_deltas

            position_ids, new_rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                None,
            )
            self.rope_deltas = new_rope_deltas
            # position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None or past_key_values is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(
                    batch_size // delta.shape[0], dim=0
                )
            position_ids = position_ids.to(delta)
            position_ids = position_ids.add(delta)
            # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits, ) + outputs[1:]
        return (loss, ) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


# Qwen2_5_VLForConditionalGeneration.forward = forward
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/data/zyw/Checkpoints/Qwen2.5-VL-3B-Instruct",
#     torch_dtype="auto",
#     device_map="auto",
# )
model = AutoModelForCausalLM.from_pretrained(
    "/archive/pretrained/qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "/data/zyw/Checkpoints/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

all_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can understand and process both text and visual information. Now, please complete a navigation task for a soft robot. I will provide you with the image of the scene where the soft robot is currently located, along with the target position it needs to navigate to. Please provide me with the corresponding travel plan in the format: {time_step: action, time_step: action}, where time_step is the moment the action is taken, and action specifies the action taken at that time. The actions are defined as follows: 0 represents moving straight ahead, 1 represents a left turn, and 2 represents a right turn. I will give you some examples to help you understand the task better. "
    },
    {
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Here are two examples for your reference:\n"
        }, {
            "type": "text",
            "text": "Example 1: \n Task: Navigate to the gray pillow.\n"
        }, {
            "type": "image",
            "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data/visual/0/visual/top/frame_00000.png",
        }, {
            "type": "text",
            "text": "Solution: {20000: 2, 260000: 0, 550000: 1, 750000: 2, 860000: 0, 930000: 2}\n"
        }, {
            "type": "text",
            "text": "Example 2: \n Task: Navigate to the basketball.\n",
        }, {
            "type": "image",
            "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data/visual/1/visual/top/frame_00000.png"
        }, {
            "type": "text",
            "text": "Solution: {50000: 2, 120000: 1, 530000: 2, 840000: 0, 870000: 0, 920000: 2}\n"
        }, {
            "type": "text",
            "text": "Now please you solve this task. Task: Navigate to the football.\n",
        }, {
            "type": "image",
            "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data/visual/2/visual/top/frame_00000.png",
        }],
    },
]

short_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Here are two examples for your reference:\n"
            },
            {
                "type": "text",
                "text": "Example 1: \n Task: Navigate to the gray pillow.\n"
            },
            {
                "type": "image",
                "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data/visual/0/visual/top/frame_00000.png",
            },
        ],
    },
]

SPLIT_LEN = 100

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def split_text_with_vision_markers(text, max_length=1000):
    """
    将文本分割成多个部分，确保每部分不超过max_length，
    并且保持视觉标记（<|vision_start|><|image_pad|><|vision_end|>）的完整性。

    Args:
        text: 要分割的文本
        max_length: 每部分的最大长度

    Returns:
        list: 分割后的文本部分列表
    """
    # 视觉标记
    vision_marker = "<|vision_start|><|image_pad|><|vision_end|>"

    # 分割文本
    result = []
    current_chunk = ""

    # 处理文本中的视觉标记
    while text:
        # 检查接下来的文本是否以视觉标记开头
        if text.startswith(vision_marker):
            # 如果当前块不为空，先保存它
            if current_chunk:
                result.append(current_chunk)
                current_chunk = ""

            # 将视觉标记作为单独的块
            result.append(vision_marker)

            # 移除已处理的视觉标记
            text = text[len(vision_marker):]
        else:
            # 找到下一个视觉标记的位置
            next_marker_pos = text.find(vision_marker)

            if next_marker_pos == -1:
                # 如果没有更多视觉标记，处理剩余文本
                if len(current_chunk) + len(text) <= max_length:
                    # 如果剩余文本加上当前块不超过最大长度，直接添加
                    current_chunk += text
                    text = ""
                else:
                    # 需要进一步分割文本
                    space_left = max_length - len(current_chunk)
                    if space_left > 0:
                        current_chunk += text[:space_left]
                        text = text[space_left:]

                    # 保存当前块并重置
                    result.append(current_chunk)
                    current_chunk = ""

                    # 如果剩余文本仍然很长，继续分割
                    while len(text) > max_length:
                        result.append(text[:max_length])
                        text = text[max_length:]

                    # 剩余文本作为新的当前块
                    current_chunk = text
                    text = ""
            else:
                # 处理视觉标记之前的文本
                text_before_marker = text[:next_marker_pos]

                if len(current_chunk) + len(text_before_marker) <= max_length:
                    # 如果标记前的文本加上当前块不超过最大长度
                    current_chunk += text_before_marker
                    text = text[next_marker_pos:]  # 保留视觉标记供下一轮处理
                else:
                    # 需要分割标记前的文本
                    space_left = max_length - len(current_chunk)
                    if space_left > 0:
                        current_chunk += text_before_marker[:space_left]
                        text_before_marker = text_before_marker[space_left:]

                    # 保存当前块并重置
                    result.append(current_chunk)
                    current_chunk = ""

                    # 处理剩余的标记前文本
                    while len(text_before_marker) > max_length:
                        result.append(text_before_marker[:max_length])
                        text_before_marker = text_before_marker[max_length:]

                    # 剩余的标记前文本作为新的当前块
                    current_chunk = text_before_marker
                    text = text[next_marker_pos:]  # 保留视觉标记供下一轮处理

    # 添加最后一个块（如果有）
    if current_chunk:
        result.append(current_chunk)

    return result


def split_text_by_vision_markers(text):
    text_parts = text.split(VISION_MARKER)
    result = []

    # Add text parts and vision markers in proper order
    for i, part in enumerate(text_parts):
        if part:
            result.append(part)
        # Add the vision marker after each part except the last one
        if i < len(text_parts) - 1:
            result.append(VISION_MARKER)

    return result


class ListDict:

    def __init__(self):
        self.data = {}

    def append(self, new_data: dict):
        for key, value in new_data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getattr__(self, item):
        return self.data[item]


# 分批处理函数
def process_in_batches(messages_list, batch_size=1):
    text = processor.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages_list)

    split_texts = split_text_by_vision_markers(text)
    past_key_values = None
    prompt_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=10240,
        # device="cuda",
        dtype=torch.bfloat16,
    )
    rope_deltas = None
    cache_position = None
    attention_mask = torch.tensor([])
    hidden_states = []
    inputs_all = ListDict()
    for idx, sub_text in enumerate(split_texts):
        if VISION_MARKER in sub_text:
            # 处理视觉信息
            batch_image_inputs = [image_inputs.pop(0)]
        else:
            batch_image_inputs = None
        inputs = processor(
            text=sub_text,
            images=batch_image_inputs,
            padding=False,
            return_tensors="pt",
        )
        mask = inputs.pop(
            "attention_mask", None
        )  # Remove attention mask if sit exists
        attention_mask = torch.cat([attention_mask, mask],
                                   dim=1) if attention_mask.numel() else mask
        inputs = inputs.to(model.device)
        inputs_all.append(inputs)
        if idx == len(split_texts) - 1:
            inputs["input_ids"] = inputs.input_ids[:, :-1]
        with torch.no_grad():

            outputs = model(
                input_ids=inputs.input_ids,
                # attention_mask=attention_mask,
                # past_key_values=past_key_values,
                # rope_deltas=rope_deltas,
                # cache_position=cache_position,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            # rope_deltas = outputs.rope_deltas

        hidden_states.append(outputs.hidden_states[-1])
        if idx == 1:
            break
    hidden_states = torch.cat(hidden_states, dim=1)
    inputs_all["input_ids"] = torch.cat(inputs_all["input_ids"], dim=1)
    inputs_all["pixel_values"] = torch.cat(inputs_all["pixel_values"], dim=0)
    inputs_all["image_grid_thw"] = torch.cat(
        inputs_all["image_grid_thw"], dim=0
    )
    hidden_states_in_one = model(
        **inputs_all.data,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-1]

    history = copy.deepcopy(past_key_values)
    with torch.no_grad():
        outputs = model.generate(
            **inputs_all.data,
            # past_key_values=history,
            max_length=20480,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs_all.input_ids, outputs)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


generated_texts = process_in_batches(all_messages)
print(generated_texts)
