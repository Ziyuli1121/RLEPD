import os
import re
import csv
import tqdm
import dnnlib
import pickle
import PIL.Image
import scipy
import numpy as np
import torch
from torch import autocast
from torch_utils.download_util import check_file_by_key
from torchvision.utils import make_grid, save_image
import utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from IPython.display import HTML
import solvers

plt.rcParams['mathtext.fontset'] = 'cm'

solver_kwargs = {}

# Options

# General options
grid = True
outdir = "./outputs"                            # Where to save the results
stat_path = None                                # If not None, load pre-computed statistics to draw the figures

device = torch.device('cuda')
seeds = utils.parse_int_list('0-63')            # One seed for one image

# Sampling options
solver_kwargs['dataset_name'] = 'cifar10'       # Name of the dataset, one in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64', 'lsun_bedroom', 'imagenet256', 'lsun_bedroom_ldm', 'ffhq_ldm', 'ms_coco']
solver_kwargs['max_batch_size'] = 64            # Maximum batch size
solver_kwargs['solver'] = 'euler'               # Name of the Solver, one in ['euler', 'heun', 'dpm', 'dpmpp', 'unipc', 'deis', 'ipndm', 'ipndm_v']
solver_kwargs['num_steps'] = 21                 # Number of timestamps. When num_steps=N, there will be N-1 sampling steps.
solver_kwargs['afs'] = False                    # Whether to use AFS which saves the first model evaluation
solver_kwargs['denoise_to_zero'] = False        # Whether to denoise from the last timestamp (>0) to 0.
solver_kwargs['return_inters'] = True           # Whether to return intermediate samples
solver_kwargs['return_denoised'] = True         # Whether to return intermediate denoised samples
solver_kwargs['return_eps'] = True              # Whether to return intermediate noises
solver_kwargs['guidance_type'] = 'cfg'          # Only useful for ADM and LDM models
solver_kwargs['guidance_rate'] = 7.5            # Only useful for ADM and LDM models
solver_kwargs['prompt'] = None                  # Only useful for SD models

# Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, iPNDM_v and DEIS, 1<=max_order<=3 for DPM-Solver++
solver_kwargs['max_order'] = 2
# Additional options for DPM-Solver++
solver_kwargs['predict_x0'] = False
solver_kwargs['lower_order_final'] = True
# Additional options for DEIS
solver_kwargs['deis_mode'] = 'tab'

# Schedule options
solver_kwargs['t_steps'] = None                 # Support custom timestamps (list of floats)
solver_kwargs['schedule_type'] = 'polynomial'   # One in ['polynomial', 'logsnr', 'time_uniform', 'discrete']
solver_kwargs['schedule_rho'] = 7               # Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform', 'discrete']

save_images = False

save_grid = False

# Calculate the magnitude of the L2 norm of the intermediate samples, denoised samples and intermediate noises.
cal_magnitude = True

# Calculate the perpendicular L2 distance of the intermediate samples (say, x_t) to the line (x_T - x_0).
cal_deviation = True

# Calculate the L2 distance of the intermediate samples to the final sample, i.e., ||x_t - x_0||.
cal_distance = True

# Calculate the cosine similarity between the intermediate gradients (say, \epsilon_t) and the line (x_0 - x_t).
cal_cos = True

# Calculate the difference between the samples and the theoretically optimal samples. (Only for CIFAR-10)
cal_opt_difference = False                       # Only for CIFAR-10
path_to_cifar10 = "/path/to/cifar10-32x32.zip"  # processed cifar10 dataset
KNN_size = 5

# Calculate the FID
cal_FID = False
inceptionV3_path = "/path/to/inception-2015-12-05.pkl"
ref_stat_path = "/path/to/cifar10-32x32.npz"    # FID reference statistics


# Load

if cal_FID:
    mu, sigma, mu_ref, sigma_ref, detector_net = utils.fid_prepare(ref_stat_path, inceptionV3_path, device)

# Load Dataset
cifar10_dataset = None
if cal_opt_difference:
    assert solver_kwargs['dataset_name'] == 'cifar10'
    cifar10_dataset = utils.cifar10_prepare(path_to_cifar10, device)

# Load Prompt
if solver_kwargs['dataset_name'] in ['ms_coco'] and solver_kwargs['prompt'] is None:
    # Loading MS-COCO captions for FID-30k evaluaion
    # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
    prompt_path, _ = check_file_by_key('prompts')
    sample_captions = []
    with open(prompt_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['text']
            sample_captions.append(text)

# Load Model
net, solver_kwargs['model_source'] = utils.create_model(solver_kwargs['dataset_name'], solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
print('Finished.')

# Loop over batches.

stat = {}
num_batches = ((len(seeds) - 1) // (solver_kwargs['max_batch_size']) + 1)
all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
rank_batches = all_batches
if stat_path is None:
    print(f'Generating {len(seeds)} images...')
    loop_count = 0
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch'):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = utils.StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = c = uc = None
        if net.label_dim:
            if solver_kwargs['model_source'] == 'adm':
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            elif solver_kwargs['model_source'] == 'ldm' and solver_kwargs['dataset_name'] == 'ms_coco':
                if solver_kwargs['prompt'] is None:
                    prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                else:
                    prompts = [solver_kwargs['prompt'] for i in range(batch_size)]
                if solver_kwargs['guidance_rate'] != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
            else:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

        # Generate images.
        with torch.no_grad():
            if solver_kwargs['model_source'] == 'ldm':
                with autocast("cuda"):
                    with net.model.ema_scope():
                        inter_xt, inter_denoised, inter_eps = sampler_fn(net, latents, condition=c, unconditional_condition=uc, **solver_kwargs)
                        images = net.model.decode_first_stage(inter_xt[-1])
            else:
                inter_xt, inter_denoised, inter_eps = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)
                images = inter_xt[-1]

        if save_images:
            outdir_img = os.path.join(f"./samples/{solver_kwargs['dataset_name']}", f"{solver_kwargs['solver']}_nfe{solver_kwargs['nfe']}")
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir_img, f'{seed-seed%1000:06d}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        if save_grid:
            outdir_img = os.path.join(f"./samples/grids/{solver_kwargs['dataset_name']}", f"{solver_kwargs['solver']}_nfe{solver_kwargs['nfe']}")
            image_grid = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir_img, exist_ok=True)
            nrows = image_grid.shape[0] // int(image_grid.shape[0] ** 0.5)
            image_grid = make_grid(image_grid, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir_img, "grid.png"))

        # Calculate the magnitude of the L2 norm of the intermediate samples, denoised samples and intermediate noises.
        if cal_magnitude:
            if loop_count == 0:
                mag_xt = torch.norm(inter_xt, p=2, dim=(2,3,4))                     # (num_steps, batch_size)
                mag_denoised = torch.norm(inter_denoised, p=2, dim=(2,3,4))         # (num_steps-1, batch_size)
                mag_eps = torch.norm(inter_eps, p=2, dim=(2,3,4))                   # (num_steps-1, batch_size)
            else:
                mag_xt = torch.cat((mag_xt, torch.norm(inter_xt, p=2, dim=(2,3,4))), dim=0)
                mag_denoised = torch.cat((mag_denoised, torch.norm(inter_denoised, p=2, dim=(2,3,4))), dim=0)
                mag_eps = torch.cat((mag_eps, torch.norm(inter_eps, p=2, dim=(2,3,4))), dim=0)

        # Calculate the perpendicular L2 distance of the intermediate samples (say, x_t) to the line (x_T - x_0).
        if cal_deviation:
            if loop_count == 0:
                dev_xt = utils.cal_deviation(inter_xt, net.img_channels, net.img_resolution, batch_size).transpose(0,1)                # (num_steps-2, batch_size)
                dev_denoised = utils.cal_deviation(inter_denoised, net.img_channels, net.img_resolution, batch_size).transpose(0,1)    # (num_steps-3, batch_size)
            else:
                dev_xt = torch.cat((dev_xt, utils.cal_deviation(inter_xt, net.img_channels, net.img_resolution, batch_size).transpose(0,1)), dim=0)
                mag_denoised = torch.cat((dev_denoised, utils.cal_deviation(inter_denoised, net.img_channels, net.img_resolution, batch_size).transpose(0,1)), dim=0)

        # Calculate the L2 distance of the intermediate samples to the final sample, i.e., ||x_t - x_0||.
        if cal_distance:
            if loop_count == 0:
                dist_xt = torch.norm(inter_xt - inter_xt[-1].unsqueeze(0), p=2, dim=(2,3,4))                        # (num_steps, batch_size)
                dist_denoised = torch.norm(inter_denoised - inter_denoised[-1].unsqueeze(0), p=2, dim=(2,3,4))      # (num_steps-1, batch_size)
            else:
                dist_xt = torch.cat((dist_xt, torch.norm(inter_xt - inter_xt[-1].unsqueeze(0), p=2, dim=(2,3,4))), dim=0)
                dist_denoised = torch.cat((dist_denoised, torch.norm(inter_denoised - inter_denoised[-1].unsqueeze(0), p=2, dim=(2,3,4))), dim=0)

        # Calculate the cosine similarity between the intermediate gradients (say, \epsilon_t) and the line (x_0 - x_t).
        if cal_cos:
            a = inter_eps.reshape(inter_eps.shape[0],batch_size,-1)
            b = (inter_xt[:-1]-inter_xt[-1].unsqueeze(0)).reshape(inter_eps.shape[0],batch_size,-1)
            if loop_count == 0:
                cos_xt = torch.nn.functional.cosine_similarity(a, b, dim=2)                                         # (num_steps-1, batch_size)
            else:
                cos_xt = torch.cat((cos_xt, torch.nn.functional.cosine_similarity(a, b, dim=2)), dim=0)

        # Calculate the difference between the samples and the theoretically optimal samples. (Only for CIFAR-10)
        if cal_opt_difference:
            assert cifar10_dataset is not None
            inter_xt_opt, inter_denoised_opt, inter_eps_opt = solvers.optimal_sampler(net, latents, cifar10_dataset, **solver_kwargs)
            images_opt = inter_xt_opt[-1]
            diff_sample_traj = torch.norm(inter_xt_opt - inter_xt, p=2, dim=(2,3,4))                                    # (num_steps, batch_size)
            diff_denoised_traj = torch.norm(inter_denoised_opt - inter_denoised, p=2, dim=(2,3,4))                      # (num_steps-1, batch_size)
            
            for i in range(inter_denoised.shape[0]):
                if i == 0:
                    opt_denoised_traj = solvers.get_denoised_opt(inter_xt[i], solver_kwargs['t_steps'][i], cifar10_dataset).unsqueeze(0)
                else:
                    opt_denoised_traj = torch.cat((opt_denoised_traj, solvers.get_denoised_opt(inter_xt[i], solver_kwargs['t_steps'][i], cifar10_dataset).unsqueeze(0)), dim=0)
            if loop_count == 0:
                diff_traj = torch.norm(opt_denoised_traj - inter_denoised, p=2, dim=(2,3,4))
            else:
                diff_traj = torch.cat((diff_traj, torch.norm(opt_denoised_traj - inter_denoised, p=2, dim=(2,3,4))), dim=0)
            
            for i in range(inter_denoised.shape[0]):
                if i == 0:
                    denoised_opt_traj = solvers.get_denoised(net, inter_xt_opt[i], solver_kwargs['t_steps'][i]).unsqueeze(0)
                else:
                    denoised_opt_traj = torch.cat((denoised_opt_traj, solvers.get_denoised(net, inter_xt_opt[i], solver_kwargs['t_steps'][i]).unsqueeze(0)), dim=0)
            if loop_count == 0:
                diff_opt_traj = torch.norm(denoised_opt_traj - inter_denoised_opt, p=2, dim=(2,3,4))
            else:
                diff_opt_traj = torch.cat((diff_opt_traj, torch.norm(denoised_opt_traj - inter_denoised_opt, p=2, dim=(2,3,4))), dim=0)

        # Calculate FID features
        if cal_FID:
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
            features = detector_net(images, return_features=True).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features

        loop_count += 1

    # Calculate grand totals.
    if cal_FID:
        print(f'Calculating FID...')
        mu /= len(seeds)
        sigma -= mu.ger(mu) * len(seeds)
        sigma /= len(seeds) - 1
        mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()

        m = np.square(mu - mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
        fid = m + np.trace(sigma + sigma_ref - s * 2)

        print(f'FID: {float(np.real(fid))}')

    # Collect all the statistics
    if cal_magnitude:
        stat['mag_xt'] = mag_xt.cpu().numpy()
        stat['mag_denoised'] = mag_denoised.cpu().numpy()
        stat['mag_eps'] = mag_eps.cpu().numpy()
    if cal_deviation:
        stat['dev_xt'] = dev_xt.cpu().numpy()
        stat['dev_denoised'] = dev_denoised.cpu().numpy()
    if cal_distance:
        stat['dist_xt'] = dist_xt.cpu().numpy()
        stat['dist_denoised'] = dist_denoised.cpu().numpy()
    if cal_cos:
        stat['cos_xt'] = cos_xt.cpu().numpy()
    if cal_opt_difference:
        stat['diff_sample_traj'] = diff_sample_traj.cpu().numpy()
        stat['diff_denoised_traj'] = diff_denoised_traj.cpu().numpy()
        stat['diff_traj'] = diff_traj.cpu().numpy()
        stat['diff_opt_traj'] = diff_opt_traj.cpu().numpy()

    # Save the computed statistics
    for key, value in solver_kwargs.items():
        if key == 't_steps':
            stat[key] = value.cpu().numpy()
        elif value is not None:
            stat[key] = value
    # prev_run_dirs = []
    # if os.path.isdir(outdir):
    #     prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    # prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    # prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    # cur_run_id = max(prev_run_ids, default=-1) + 1
    # desc = f"{solver_kwargs['dataset_name']:s}-{solver_kwargs['solver']}-steps{solver_kwargs['num_steps']}-batch{len(seeds)}"
    # run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    # np.savez(os.path.join(run_dir, 'stat.npz'), **stat)
    
else:
    stat = np.load(stat_path)
    # Print solver settings.
    print("Solver settings:")
    for key, value in stat.items():
        if value is None:
            continue
        elif key in ['dataset_name', 'solver', 'num_steps', 't_steps']:
            print(f"\t{key}: {value}")