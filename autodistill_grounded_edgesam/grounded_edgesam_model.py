import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
import urllib
from autodistill.helpers import load_image
from autodistill_grounding_dino import GroundingDINO
import numpy as np
import subprocess

from typing import Any

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_dependencies():
    # Create the ~/.cache/autodistill directory if it doesn't exist
    original_dir = os.getcwd()
    autodistill_dir = os.path.expanduser("~/.cache/autodistill")
    os.makedirs(autodistill_dir, exist_ok=True)

    os.chdir(autodistill_dir)

    # git clone  https://github.com/autodistill/EdgeSAM
    if not os.path.isdir("EdgeSAM"):
        print("Cloning EdgeSAM...")
        subprocess.run(["git", "clone", "https://github.com/autodistill/EdgeSAM"])

        os.chdir("EdgeSAM")

        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        subprocess.run(["pip3", "install", "-e", "."])

    from segment_anything import SamPredictor
    from segment_anything.build_sam import sam_model_registry

    print("Loading EdgeSAM...")
    # Check if segment-anything library is already installed

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "edge_sam.pth")

    url = "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        print("Downloading EdgeSAM checkpoint...")
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "edge_sam"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamPredictor(sam)

    os.chdir(original_dir)

    return sam_predictor


@dataclass
class GroundedEdgeSAM(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(
        self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25
    ):
        self.ontology = ontology
        self.predictor = check_dependencies()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.grounding_dino_model = GroundingDINO(ontology)

    def predict(self, input: Any) -> sv.Detections:
        input = load_image(input, return_format="cv2")
        detections = self.grounding_dino_model.predict(input)

        xyxy = detections.xyxy

        self.predictor.set_image(input)
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.predictor.predict(
                box=box
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        
        detections.mask = np.array(result_masks)

        return detections