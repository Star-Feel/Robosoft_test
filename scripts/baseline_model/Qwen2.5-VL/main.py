import copy
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, QWEN2_5_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/zyw/Checkpoints/Qwen2.5-VL-3B-Instruct",
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

pre_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can understand and process both text and visual information. Now, please complete a navigation task for a soft robot. I will provide you with the image of the scene where the soft robot is currently located, along with the target position it needs to navigate to. Please provide me with the corresponding travel plan in the format: {time_step: action, time_step: action}, where time_step is the moment the action is taken, and action specifies the action taken at that time. The actions are defined as follows: 0 represents moving straight ahead, 1 represents a left turn, and 2 represents a right turn. I will give you some examples to help you understand the task better. "
    },
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
                "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data_new/visual/0/visual/top/frame_00000.png",
            },
            {
                "type": "text",
                "text": "Solution: {2000: 2, 26000: 0, 86000: 1}\n"
            },
            {
                "type": "text",
                "text": "Example 2: \n Task: Navigate to the basketball.\n",
            },
            {
                "type": "image",
                "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data_new/visual/1/visual/top/frame_00000.png"
            },
            {
                "type": "text",
                "text": "Solution: {18000: 0, 33000: 1, 45000: 0}\n"
            },
        ],
    },
]
questions = [{
    "type": "text",
    "text": "Now please you solve this task. Task: Navigate to the football.\n",
}, {
    "type": "image",
    "image": "/data/zyw/workshop/attempt/work_dirs/navigation_data_new/visual/2/visual/top/frame_00000.png",
}]


def single_inference(messages_list):
    messages_list[1]["content"].extend(questions)
    text = processor.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages_list)

    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=20480,
            max_new_tokens=1024,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text

# def multi_inference(messages_list, num_inferences=10):
generated_texts = single_inference(pre_messages)
print(generated_texts)
