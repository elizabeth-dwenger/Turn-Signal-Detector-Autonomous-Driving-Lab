TURN_SIGNAL_LABELS = ["left", "right", "none", "both"]
#TAIL_LIGHT_LABELS = ["on", "off", "not_visible"]

# Mappings
TURN_SIGNAL_TO_IDX = {label: idx for idx, label in enumerate(TURN_SIGNAL_LABELS)}
IDX_TO_TURN_SIGNAL = {idx: label for label, idx in TURN_SIGNAL_TO_IDX.items()}

#TAIL_LIGHT_TO_IDX = {label: idx for idx, label in enumerate(TAIL_LIGHT_LABELS)}
#IDX_TO_TAIL_LIGHT = {idx: label for label, idx in TAIL_LIGHT_TO_IDX.items()}
