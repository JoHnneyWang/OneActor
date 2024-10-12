from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os
import argparse
import yaml
import pickle
import json
import shutil
import sys

sys.path.append("./diffusers")

def cat_images(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_image

def decode_latent(latent, pipeline):
    with torch.no_grad():
        pipeline.upcast_vae()
        latent = latent.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
        image = pipeline.vae.decode(latent / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type='pil')[0]
    return image

def decode_and_cat(latent_list, pipeline):
    images = []
    for i in latent_list:
        images.append(decode_latent(i, pipeline))
    image = cat_images(images)
    return image
        

if __name__ == '__main__':
    # get environment configs
    with open("PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    # get user configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/gen_tune.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    device = config['device']
    pipeline = DiffusionPipeline.from_pretrained(ENV_CONFIGS['paths']['sdxl_path']).to(device)
    data_root = config['data_root']
    prompt = config['source_prompt']
    guidance_scale = config['guidance_scale']
    steps = config['steps']
    generator = torch.manual_seed(config['g_seed'])
    # make data dirs
    os.makedirs(data_root, exist_ok=True)
    # copy the config
    shutil.copyfile(opt.config_path, data_root+'/config.yaml')
    # use original pipeline to prepare data
    image, xt_list_, prompt_embeds, mid_ = pipeline(prompt, neg_prompt=config['neg_prompt'], 
                                                    num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
                                                    oneactor_save=True)
    # save the target image
    image = image.images[0]
    image.save(data_root+'/source.jpg')
    # save feature h and latent z
    xt_list = [_.cpu() for _ in xt_list_]
    mid = [_.cpu() for _ in mid_]
    save_pkl = {'xt':xt_list, 'prompt_embed':prompt_embeds.cpu(), 'h_mid': mid}
    with open(data_root+'/xt_list.pkl', 'wb') as f:
        pickle.dump(save_pkl, f)
    # generate auxiliary images and data
    if config['gen_base'] > 0:
        num_base = config['gen_base']
        os.makedirs(data_root+'/base', exist_ok=True)
        mid_last_base = []
        for i in range(num_base):
            image, xt_list_, prompt_embeds, mid_ = pipeline(prompt, neg_prompt=config['neg_prompt'],
                                                            num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
                                                            oneactor_save=True)
            image = image.images[0]
            image.save(data_root+f'/base/base{i}.jpg')
            mid_last_base.append(mid_[-1].cpu())
        with open(data_root+'/base/mid_list.pkl', 'wb') as f:
            pickle.dump(mid_last_base, f)
