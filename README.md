# ImageSegmentation-Using-GraoundingDINO-SegmentAnythingModel

GroundingDINO and Segment Anything Model Image Segmentation

Welcome to the GroundingDINO and Segment Anything Model Image Segmentation repository! This powerful tool empowers you to perform accurate and efficient image segmentation using region-of-interest definitions, giving you the ability to precisely identify objects within images.
Installation and Setup

Follow these simple steps to set up the repository and start using the image segmentation capabilities:
Prerequisites

    Python: Make sure you have Python installed on your system. You can download it from python.org.

Step 1: Clone the Repository

bash

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

Step 2: Install Dependencies

bash

pip install -q -e .
pip install -q roboflow

Step 3: Download Pre-trained Weights

bash

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Step 4: Install Segment Anything Model

bash

pip install git+https://github.com/facebookresearch/segment-anything.git
pip uninstall -y supervision
pip install -q -U supervision==0.6.0

Usage

    Initialize the Environment:

    Set up the necessary environment variables in your preferred way. Ensure HOME_DIR, GD_DIR, and weights_dir are correctly configured.

    Run Image Segmentation:

    Use the prediction function from image_segment.py to perform image segmentation. Provide the initial image and specify the object you want to detect (defined by word_mask). The function will return the segmented image.

    python

    from image_segment import prediction

    # Load your initial image
    init_image = ...

    # Define the object to detect (e.g., "shoes")
    word_mask = "shoes"

    # Perform image segmentation
    segmented_image = prediction(init_image, word_mask)

Directory Structure

    GroundingDINO/: Contains the GroundingDINO codebase.
    weights/: Directory for storing pre-trained model weights.
    image_segment.py: Module for image segmentation functions.
    tools.py: Utility functions for logging and other purposes.
