<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Grounded EdgeSAM Module

This repository contains the code supporting the Grounded EdgeSAM base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[EdgeSAM](https://github.com/chongzhou96/EdgeSAM), introduced in the "EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM" paper, is a faster version of the Segment Anything model.

Grounded EdgeSAM combines [Grounding DINO](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/) and EdgeSAM, allowing you to identify objects and generate segmentation masks for them.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/) to learn more about Autodistill.

## Installation

To use Grounded EdgeSAM with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-grounded-edgesam
```

## Quickstart

```python
from autodistill_clip import CLIP

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
from autodistill_grounded_edgesam import GroundedEdgeSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedEdgeSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "forklift": "forklift",
        }
    )
)

# run inference on a single image
results = base_model.predict("logistics.jpeg")

plot(
    image=cv2.imread("logistics.jpeg"),
    classes=base_model.ontology.classes(),
    detections=results
)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

## License

This repository is released under an [S-Lab License 1.0](LICENSE) license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
