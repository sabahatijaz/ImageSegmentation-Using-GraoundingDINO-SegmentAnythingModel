# ImageSegmentation-Using-GraoundingDINO-SegmentAnythingModel

Image Segmentation Using GroundingDINO and SegmentAnything Model

Image Segmentation Using GroundingDINO and SegmentAnything Model is a comprehensive image processing repository that combines the power of GroundingDINO for image cropping and outpainting and the SegmentAnything model for advanced image segmentation. This repository enables users to perform accurate and efficient image segmentation tasks, providing precise delineation of objects within images.
Features

    Image Cropping and Outpainting: Utilize GroundingDINO to crop images around specific regions of interest and seamlessly extend smaller images using generative AI.

    Advanced Image Segmentation: Harness the SegmentAnything model to achieve high-quality image segmentation results, accurately identifying and delineating objects within images.

Setup Instructions

Ensure you have Python installed on your system before proceeding with the following steps.
Step 1: Clone the Repository

bash

git clone https://github.com/USERNAME/ImageSegmentation-Using-GroundingDINO-SegmentAnythingModel.git
cd ImageSegmentation-Using-GroundingDINO-SegmentAnythingModel

Step 2: Install Dependencies

bash

pip install -q -e .
pip install -q roboflow
pip install git+https://github.com/facebookresearch/segment-anything.git
pip uninstall -y supervision
pip install -q -U supervision==0.6.0

Step 3: Download Pre-trained Weights

bash

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage

    Image Segmentation: Utilize the SegmentAnything model by following the usage instructions provided in its documentation. Ensure you provide the appropriate input images, and the model will output accurate segmentation masks.
