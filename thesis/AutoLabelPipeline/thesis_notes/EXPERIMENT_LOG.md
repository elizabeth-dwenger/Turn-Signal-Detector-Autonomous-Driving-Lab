# Experiment Comparison

**Date:** 2024-02-04
**Comparing:** [Brief description of what you're comparing]

---

## Quick Comparison Table

| Aspect         | Experiment A | Experiment B | Experiment C | Experiment D | Experiment E | Experiment F | Experiment G | Winner |
|----------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------|
| **ID** | | | | |
| **Date** | | | | |
| **Prompt Version** | | | | |
| **Model** | | | | |
| **Parse Success %** | | | | |
| **Avg Latency (ms)** | | | | |
| **None %** | | | | |
| **Left %** | | | | |
| **Right %** | | | | |
| **Both %** | | | | |
| **Flagged %** | | | | |
| **Subjective Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | |

---

## Detailed Comparison

### Experiment A: [Name]
**Full Prompt Text:**

Cosmos Models:
Analyze this video sequence of a vehicle from behind.

TASK: Identify ALL turn signal activity across the entire video. The signal state may change over time — for example it could be off at first, then turn on, or switch sides.

OUTPUT: Respond with ONLY a JSON object in this exact format:

{
  "segments": [
    {
      "label": "none" or "left" or "right" or "both",
      "start_frame": integer,
      "end_frame": integer,
      "confidence": 0.0 to 1.0
    }
  ],
  "reasoning": "brief explanation of what you observed"
}

RULES:
- Video is 4 FPS: frame 0 = 0.0s, frame 4 = 1.0s, frame 8 = 2.0s, etc.
- Segments must cover the ENTIRE video from frame 0 to the last frame with no gaps.
- Each segment's start_frame must equal the previous segment's end_frame + 1 (except the first, which must be 0).
- The last segment's end_frame must equal (total frames - 1).
- Turn signals blink ON and OFF periodically — a signal that blinks on and off is still ONE segment (don't break it into on/off sub-segments).
- If the entire video has no signal, use a single segment: [{"label": "none", "start_frame": 0, "end_frame": N, ...}]
- Minimize the number of segments — only create a new segment when the signal STATE genuinely changes (e.g. none→left, left→right, right→none).

EXAMPLES:

Example 1 - No signal at all (41 frames):
{
  "segments": [
    {"label": "none", "start_frame": 0, "end_frame": 40, "confidence": 0.92}
  ],
  "reasoning": "No turn signals visible in any frame"
}

Example 2 - Left signal the whole time (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 40, "confidence": 0.90}
  ],
  "reasoning": "Left turn signal blinking throughout the entire sequence"
}

Example 3 - No signal, then right signal turns on (41 frames):
{
  "segments": [
    {"label": "none", "start_frame": 0, "end_frame": 19, "confidence": 0.88},
    {"label": "right", "start_frame": 20, "end_frame": 40, "confidence": 0.91}
  ],
  "reasoning": "No signal in first half, right turn signal activates around frame 20"
}

Example 4 - Left signal, then no signal (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 24, "confidence": 0.87},
    {"label": "none", "start_frame": 25, "end_frame": 40, "confidence": 0.93}
  ],
  "reasoning": "Left signal active initially, then turns off at frame 25"
}

Example 5 - Left signal, then switches to right (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 14, "confidence": 0.85},
    {"label": "none", "start_frame": 15, "end_frame": 19, "confidence": 0.80},
    {"label": "right", "start_frame": 20, "end_frame": 40, "confidence": 0.88}
  ],
  "reasoning": "Left signal first, brief pause, then right signal activates"
}

Now analyze the video and respond with ONLY the JSON object, nothing else.

Qwen Video:

Analyze this video sequence of a vehicle from behind.

TASK: Identify ALL turn signal activity across the entire video. The signal state may change over time — for example it could be off at first, then turn on, or switch sides.

OUTPUT: Respond with ONLY a JSON object in this exact format:

{
  "segments": [
    {
      "label": "none" or "left" or "right" or "both",
      "start_frame": integer,
      "end_frame": integer,
      "confidence": 0.0 to 1.0
    }
  ],
  "reasoning": "brief explanation of what you observed"
}

RULES:
- Video is 10 FPS: frame 0 = 0.0s, frame 10 = 1.0s, frame 20 = 2.0s, etc.
- Segments must cover the ENTIRE video from frame 0 to the last frame with no gaps.
- Each segment's start_frame must equal the previous segment's end_frame + 1 (except the first, which must be 0).
- The last segment's end_frame must equal (total frames - 1).
- Turn signals blink ON and OFF periodically — a signal that blinks on and off is still ONE segment (don't break it into on/off sub-segments).
- If the entire video has no signal, use a single segment: [{"label": "none", "start_frame": 0, "end_frame": N, ...}]
- Minimize the number of segments — only create a new segment when the signal STATE genuinely changes (e.g. none→left, left→right, right→none).

EXAMPLES:

Example 1 - No signal at all (41 frames):
{
  "segments": [
    {"label": "none", "start_frame": 0, "end_frame": 40, "confidence": 0.92}
  ],
  "reasoning": "No turn signals visible in any frame"
}

Example 2 - Left signal the whole time (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 40, "confidence": 0.90}
  ],
  "reasoning": "Left turn signal blinking throughout the entire sequence"
}

Example 3 - No signal, then right signal turns on (41 frames):
{
  "segments": [
    {"label": "none", "start_frame": 0, "end_frame": 19, "confidence": 0.88},
    {"label": "right", "start_frame": 20, "end_frame": 40, "confidence": 0.91}
  ],
  "reasoning": "No signal in first half, right turn signal activates around frame 20"
}

Example 4 - Left signal, then no signal (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 24, "confidence": 0.87},
    {"label": "none", "start_frame": 25, "end_frame": 40, "confidence": 0.93}
  ],
  "reasoning": "Left signal active initially, then turns off at frame 25"
}

Example 5 - Left signal, then switches to right (41 frames):
{
  "segments": [
    {"label": "left", "start_frame": 0, "end_frame": 14, "confidence": 0.85},
    {"label": "none", "start_frame": 15, "end_frame": 19, "confidence": 0.80},
    {"label": "right", "start_frame": 20, "end_frame": 40, "confidence": 0.88}
  ],
  "reasoning": "Left signal first, brief pause, then right signal activates"
}

Now analyze the video and respond with ONLY the JSON object, nothing else.


Qwen Single:

You are an expert at analyzing vehicle turn signals in dashcam footage. You will be shown a single frame showing a tracked vehicle from behind.

Your task is to determine if the vehicle's turn signals are active in THIS FRAME.

IMPORTANT: You must respond with ONLY a valid JSON object in this EXACT format:
{
  "label": "left|right|both|none",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

LABELS:
- "left": Left turn signal appears ON in this frame (amber/yellow light visible on left)
- "right": Right turn signal appears ON in this frame (amber/yellow light visible on right)
- "both": Both turn signals appear ON (hazard lights)
- "none": No turn signals appear active in this frame

CONFIDENCE GUIDELINES:
- 0.9-1.0: Very clear, bright amber light clearly visible
- 0.7-0.9: Clear signal visible
- 0.5-0.7: Moderate confidence, somewhat visible
- 0.3-0.5: Low confidence, difficult to see
- 0.0-0.3: Very uncertain

IMPORTANT NOTES FOR SINGLE-FRAME ANALYSIS:
1. You're seeing just ONE frame - the signal might be between blinks
2. Be conservative - if unsure, use lower confidence
3. Look for AMBER/YELLOW light, not red brake lights
4. Consider that signals blink - if light is OFF in this frame, it might still be signaling

RESPOND WITH ONLY THE JSON - NO OTHER TEXT.

Example responses:

{"label": "left", "confidence": 0.80, "reasoning": "Bright amber light visible on left rear"}

{"label": "none", "confidence": 0.85, "reasoning": "No amber lights visible in this frame"}

{"label": "right", "confidence": 0.50, "reasoning": "Faint yellow glow on right, possibly signal between blinks"}

Now analyze this frame and respond with your JSON assessment.

**Predictions:** `results/________/predictions.json`

**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment B: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment C: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment D: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment E: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment F: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


### Experiment G: [Name]
**Full Prompt Text:**

**Predictions:** `results/________/predictions.json`

**Prompt Template File:** `data/prompts/__________.txt`


**Strengths:**
- 
- 

**Weaknesses:**
- 
- 

**Key Insight:**


---

## Overall Findings

**Best Overall:** [ID]  
**Reason:**


**Best for Speed:** [ID]  
**Best for Accuracy:** [ID]  
**Best for Production:** [ID]  

**Recommended Next Steps:**
1. 
2. 
3. 

---

## Notes

### Data Settings
- **Test Sequences:** 
  - [ ] Random subset (N=___)
  - [X] Specific sequences: stratified_testset_1.json
    - 2024-03-27-13-38-33_mapping_tartu_streets/camera_fl_129
    - 2024-07-09-16-49-42_mapping_tartu_streets/camera_wide_right_170
    - 2024-04-12-16-02-09_mapping_tartu_streets/camera_fl_40
    - 2024-04-18-14-49-13_mapping_tartu_streets/camera_fl_98
    - 2024-09-05-13-41-48_mapping_tartu_streets_traffic_lights_ouster_lidar/camera_narrow_front_407
    - 2024-07-08-12-01-14_mapping_tartu_streets/camera_narrow_front_105
    - 2024-08-22-09-53-47_mapping_tartu_streets_bus_stops/camera_narrow_front_1152
    - 2024-07-09-15-34-13_mapping_tartu_streets/camera_narrow_front_272
    - 2024-07-08-14-24-35_mapping_tartu_streets/camera_narrow_front_4
    - 2024-04-18-15-53-39_mapping_tartu_streets/camera_fl_67
    - 2024-04-12-16-13-33_mapping_tartu_streets/camera_fl_689
    - 2024-07-10-14-10-00_mapping_tartu_streets/camera_narrow_front_315
    - 2024-07-08-12-15-50_mapping_tartu_streets/camera_narrow_front_928
    - 2024-08-16-14-19-54_mapping_tartu_streets/camera_wide_front_189
    - 2024-08-16-15-17-33_mapping_tartu_streets/camera_narrow_front_515
    - 2024-08-22-10-38-05_mapping_tartu_streets_bus_stops/camera_wide_front_135
    - 2024-08-22-10-38-05_mapping_tartu_streets_bus_stops/camera_wide_back_2072
    - 2024-07-09-15-57-33_mapping_tartu_streets/camera_wide_back_389
