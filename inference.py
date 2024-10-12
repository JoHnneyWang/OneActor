import os
import pickle
import torch
import sys
sys.path.append('./diffusers')
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import yaml
import argparse
import shutil
import json

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def find_token_ids(tokenizer, prompt, words):
    tokens = tokenizer.encode(prompt)
    ids = []
    for word in words:
        for i, token in enumerate(tokens):
            if tokenizer.decode(token) == word:
                ids.append(i)
                break
    assert len(ids) != 0 , 'Cannot find the word in the prompt.'
    return ids

def projector_inference(projector_path, h_target, h_base, device):
    with torch.no_grad():
        projector = torch.load(projector_path).to(device)
        mid_base_target = h_base + [h_target[-1]]
        mid_base_all = torch.stack(mid_base_target)
        delta_emb_all = projector(mid_base_all[:,-1].to(device))
    return delta_emb_all

def pipeline_inference(pipeline, config, oneactor_extra_config, generator=None):
    if generator is None:
        generator = torch.manual_seed(config['seed'])
    return pipeline(config['prompt'], negative_prompt=config['neg_prompt'], num_inference_steps=config['inference_steps'], guidance_scale=config['eta_1'], \
            generator=generator, oneactor_extra_config=oneactor_extra_config)

def main():
    with open("PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/inference.yaml')
    args = parser.parse_args()
    # load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # make dir and initialize
    out_root = config['pretrain_root'] + config['out_file']
    os.makedirs(out_root, exist_ok=True)

    shutil.copyfile(args.config_path, out_root+'/gen_config.yaml')

    # load sd pipeline
    pipeline = DiffusionPipeline.from_pretrained(ENV_CONFIGS['paths']['sdxl_path']).to(config['device'])
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load cluster information
    xt_dic = load_pickle(config['data_root']+'/xt_list.pkl')
    h_base = load_pickle(config['data_root']+'/base/mid_list.pkl')
    h_tar = xt_dic['h_mid']

    # original output by SDXL
    generator = torch.manual_seed(config['seed'])
    image = pipeline(config['prompt'], negative_prompt=config['neg_prompt'], num_inference_steps=config['inference_steps'], guidance_scale=config['eta_1'], generator=generator)
    image = image.images[0]
    image.save(out_root+'/original_sdxl.jpg')

    # perform step-wise guidance
    select_steps = config['select_steps']
    if select_steps is not False:
        assert (len(select_steps) % 2) == 0
        select_list = []
        for _ in range(len(select_steps) // 2):
            a = select_steps[2*_]
            b = select_steps[2*_ + 1]
            select_list = select_list + list(range(a-1,b))
    else:
        select_list = None

    # locate the base token id
    token_id = find_token_ids(pipeline.tokenizer, config['prompt'], [config['base']])
    generator = torch.manual_seed(config['seed'])
    config['generator'] = generator

    if config['only_step'] is False:
        for i in range(50):
            steps = config['step_from']+config['step']*(i)
            
            projector_path = config['pretrain_root'] + f'/weight/learned-projector-steps-{steps}.pth'
            delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

            delta_emb_aver = delta_emb_all[:-1].mean(dim=0)
            delta_emb_tar = config['v'] * delta_emb_all[-1]

            oneactor_extra_config = {
                'token_ids': token_id,
                'delta_embs': delta_emb_tar,
                'delta_steps': select_list,
                'eta_2': config['eta_2'],
                'delta_emb_aver': delta_emb_aver
            }

            image = pipeline_inference(pipeline, config, oneactor_extra_config)
            image = image.images[0]
            image.save(out_root+f'OneActor.jpg')
    elif config['only_step'] == 'best':
        projector_path = config['pretrain_root'] + f'/weight/best-learned-projector.pth'
        delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

        delta_emb_aver = delta_emb_all[:-1].mean(dim=0) # [2048]
        delta_emb_tar = config['v'] * delta_emb_all[-1] # [2048]
        
        oneactor_extra_config = {
            'token_ids': token_id,
            'delta_embs': delta_emb_tar,
            'delta_steps': select_list,
            'eta_2': config['eta_2'],
            'delta_emb_aver': delta_emb_aver
        }
        image = pipeline_inference(pipeline, config, oneactor_extra_config)
        image = image.images[0]
        image.save(out_root+f'/OneActor.jpg')
    else:
        steps = config['only_step']
            
        with torch.no_grad():
            projector_path = config['pretrain_root'] + f'/weight/learned-projector-steps-{steps}.pth'
            delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

        delta_emb_aver = delta_emb_all[:-1].mean(dim=0) # [2048]
        delta_emb_tar = config['v'] * delta_emb_all[-1] # [2048]

        oneactor_extra_config = {
            'token_ids': token_id,
            'delta_embs': delta_emb_tar,
            'delta_steps': select_list,
            'eta_2': config['eta_2'],
            'delta_emb_aver': delta_emb_aver
        }
        image = pipeline_inference(pipeline, config, oneactor_extra_config)
        image = image.images[0]
        image.save(out_root+f'/OneActor.jpg')

if __name__ == '__main__':
    main()