"""
Minimal vision input parser for Qwen/Cosmos-style chat messages.
Extracts image and video inputs from message content.
"""
from typing import List, Dict, Tuple


def process_vision_info(messages: List[Dict]) -> Tuple[List, List]:
    """
    Extract image and video inputs from a list of chat messages.
    Returns (image_inputs, video_inputs).
    """
    image_inputs = []
    video_inputs = []

    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and "image" in item:
                image_inputs.append(item["image"])
            elif item_type == "video" and "video" in item:
                video_inputs.append(item["video"])

    return image_inputs, video_inputs
