from transformers import DetrForObjectDetection, DetrImageProcessor

# Define Labels consistent with VIRAT dataset
LABELS = {
    0: "unknown",
    1: "person",
    2: "car",
    3: "other_vehicle",
    4: "other_object",
    5: "bike",
}

def load_processor():
    """
    Helper to load the processor (used in train.py for collate_fn)
    """
    return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def apply_freeze_mode(model, mode):
    """
    Implements the 'Fine-tuning methods' from the Assignment PDF.
    """
    mode = mode.lower()
    print(f"Applying freeze mode: {mode}")

    def freeze_module(module, train_it=False):
        for p in module.parameters():
            p.requires_grad = train_it

    if mode == "all":
        # 1. Fine-tune all parameters
        freeze_module(model, train_it=True)
        return model

    if mode == "backbone_only":
        # 2. Fine-tune only the convolutional block
        freeze_module(model.model.backbone, train_it=True)
        freeze_module(model.model.encoder, train_it=False)
        freeze_module(model.model.decoder, train_it=False)
        freeze_module(model.class_labels_classifier, train_it=False)
        freeze_module(model.bbox_predictor, train_it=False)
        return model

    if mode == "class_head_only":
        # 3. Fine-tune only the transformer classification head
        freeze_module(model.model.backbone, train_it=False)
        freeze_module(model.model.encoder, train_it=False)
        freeze_module(model.model.decoder, train_it=False)
        freeze_module(model.class_labels_classifier, train_it=True)
        freeze_module(model.bbox_predictor, train_it=True)
        return model

    if mode == "transformer_only":
        # 4. Finetune only the transformer block
        freeze_module(model.model.backbone, train_it=False)
        freeze_module(model.model.encoder, train_it=True)
        freeze_module(model.model.decoder, train_it=True)
        freeze_module(model.class_labels_classifier, train_it=True)
        freeze_module(model.bbox_predictor, train_it=True)
        return model

    # Fallback / Error
    raise ValueError(f"Unknown freeze mode: {mode}. Check your spelling.")

def build_model(num_classes=6, freeze_mode="all"):
    """
    Builds the DETR model with HYPERPARAMETER TUNING for small datasets.
    """
    # Load Pretrained DETR
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    
    # --- PRO UPGRADE: TUNE HYPERPARAMETERS ---
    # These settings help the model learn faster on small datasets (VIRAT)
    config = model.config
    config.class_cost = 2.0             # Penalize wrong class harder
    config.bbox_loss_coefficient = 7.0  # Prioritize exact box location
    config.giou_loss_coefficient = 4.0  # Prioritize box overlap
    config.eos_coef = 0.9               # Aggressively punish "no object" predictions
    
    # Apply the specific rubric requirement logic
    apply_freeze_mode(model, freeze_mode)
    
    return model