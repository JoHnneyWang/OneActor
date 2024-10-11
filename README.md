# Some Information
This demo is OneActor simple implemention. As an alpha version, this demo only supports consistent generation of single subject. We promise that we will release the full version as soon as we finish the necessary preparations. Hope you have fun in this demo.
# Prepare Environment
This demo is designed to drive on NVIDIA A100 and the python version is 3.10.0.
Use this code to install the required packages:
`pip install -r requirements.txt`
# Specify PATH
Please edit the <mark>PATH.json</mark> to specify the paths to the code and the StableDiffusionXL model.
# Generate Target Image
Please change the prompt and other settings in <mark>./config/gen_tune.yaml</mark> or you can leave it to the default setting.
Use this code to generate the target image and the auxiliary images:
`python generate_data.py`
The images will be saved at <mark>./data/demo</mark>.
# Tune the Projector
`python tune.py`
# Inference for Consistent Images
Please change the prompt and other settings in <mark>./config/inference.yaml</mark> or you can leave it to the default setting.
Use this code to generate the consistent image:
`python inference.py`
The images will be saved at <mark>./output/demo/inference</mark>.