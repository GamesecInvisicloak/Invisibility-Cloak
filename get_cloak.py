import os
import torch
import numpy as np
from PIL import Image
from utils.dataset import load_dataset
import time
import shutil
from tqdm import tqdm
from loguru import logger
import argparse
from pytorch_msssim import ssim
import imageio

parser = argparse.ArgumentParser(description='Invisibility Cloak')
parser.add_argument('--game', default='cs2', type=str,
                    help='Dataset')
parser.add_argument('--n_iter', default=5, type=int,
                    help='n_iter')
parser.add_argument('--lr', default=0.005, type=float,
                    help='epsilon')
parser.add_argument('--epsilon', default=8, type=int,
                    help='attack strength')
parser.add_argument('--local_model', default='yolov5n', type=str, choices=['yolov5n','yolov5s','yolov5m'],
                    help='local_model')
parser.add_argument('--target_model', default='yolov5n', type=str, choices=['yolov5n','yolov5s','yolov5m'],
                    help='target_model')
parser.add_argument('--gpu', default='0', type=str,
                    help='device')
parser.add_argument('--use_universal_cloak', default=1, type=int,
                    help='global')
parser.add_argument('--visualize_gif', default=0, type=int,
                    help='visualize_gif') 
parser.add_argument('--scenario', default='cover', type=str, 
                    choices=['2people','back','cover','fire','flash','football',
                    'halfbody','hited','jump','knife','op','props','reload','run',
                    'side','smoke','stand','usingprop'],
                    help='cs2 demo scenario')

args = parser.parse_args()
device = 'cuda:' + args.gpu
BCEobj = torch.nn.BCELoss()
BCElogits = torch.nn.BCEWithLogitsLoss()
gt_conf = torch.zeros((1, 63000, 1),device=device)
gt_conf_nano = torch.zeros((1, 21250),device=device)
global_noise = None
if args.game == 'cf':
    global_noise_path = 'universal_cloak/cf/yolov5n_yolov5s_yolov5m/Best-Succ-0.9143-BS-16-LR-0.001.pt'
elif args.game == 'cs2':
    global_noise_path = 'universal_cloak/cs2/yolov5n_yolov5s_yolov5m/Best-Succ-0.84-BS-16-LR-0.001.pt'
if args.use_universal_cloak == 1:
    print("use_universal_cloak")
    global_noise = torch.load(global_noise_path).to(device)

class FindNoise(torch.nn.Module):
    def __init__(self, model,epsilon,noise):
        super(FindNoise,self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.noise = noise
        self.noise.requires_grad = True

    def forward(self,input):
        noise = torch.clamp(self.noise, -self.epsilon, self.epsilon) 
        adv_img = torch.clamp(input + noise, 0.0, 1.0) 
        preds = self.model(adv_img) 
        return preds 
    def get_adv(self,input):
        with torch.no_grad():
            noise = torch.clamp(self.noise, -self.epsilon, self.epsilon) 
            adv_img = torch.clamp(input + noise, 0.0, 1.0) 
        return adv_img


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom rightog_vanishingt x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def attack_success(preds,conf_thres=0.4):
    preds = preds.detach()
    xc = preds[..., 4] > 0.25  # candidates

    x = preds[0][xc[0]]  # confidence
    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    # Box/Mask
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    mask = x[:, 85:]  # zero columns if no masks
    # Detections matrix nx6 (xyxy, conf, cls)
    conf, j = x[:, 5:85].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
    # Filter by class
    x = x[(x[:, 5:6] == torch.tensor(0, device=x.device)).any(1)]
    count = x.shape[0]
    return count

def tog_vanishing(x,model,n_iter=3, eps=8/255., lr=0.001):
    t_start = time.time()
    succ = 0
    query = 0
    x = x.clone()
    # test global noise
    with torch.no_grad():
        if not global_noise is None:
            eta = torch.clamp(global_noise,-eps,eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0)
            preds = model(x_adv)
            query += 1
            result = attack_success(preds)
            if result == 0: # attack success
                succ = 1
                t = time.time() - t_start
                return x_adv,query,succ,t
            else:
                eta = global_noise.clone()
        else:
            eta = torch.rand((x.shape),device=device).mul(2*eps).sub(eps)

    # create find noise model 
    noise_model = FindNoise(model, eps,eta).to(device)
    optimizer = torch.optim.Adam([{'params': noise_model.noise}],lr=lr)

    for k in range(n_iter):
        compute_loss = 0
        x.requires_grad = True
        preds = noise_model(x)
        query += 1
        result = attack_success(preds)
        if result == 0:
            succ = 1
            break
        # compute loss and backward and update eta
        compute_loss = BCEobj(preds[...,4:5], gt_conf[:,:preds.shape[1],:]) 
        compute_loss.backward()
        logger.info('compute_loss {:.6f} '.format(compute_loss.item()))
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

    x_adv = noise_model.get_adv(x)
    t = time.time() - t_start
    return x_adv,query,succ,t

def create_gif(demo_save_dir):
    gt_folder = os.path.join(demo_save_dir,'gt')
    attack_folder = os.path.join(demo_save_dir,'attack')
    gt_files = os.listdir(gt_folder)
    gt_num_list = [int(f) for f in gt_files ]
    gt_sorted_files = [f for _, f in sorted(zip(gt_num_list, gt_files))]
    output_folder = os.path.join(demo_save_dir,'gif')
    os.makedirs(output_folder, exist_ok=True)

    gt_files = os.listdir(gt_folder)
    attack_files = os.listdir(attack_folder)


    num_list = [int(f.split('.')[0]) for f in attack_files ]
    sorted_files = [f for _, f in sorted(zip(num_list, attack_files))]
    frame_duration = 60

    gt_images = []
    attack_images = []
    gif_num = 0
    for idx,dir_idx in enumerate(sorted_files):
        if idx % 100 == 0 and idx!=0:
            gt_images = []
            attack_images = []

        attack_image_path = os.path.join(attack_folder, dir_idx,'image0.jpg')
        attack_image = Image.open(attack_image_path)
        attack_images.append(attack_image)
        gt_image_path = os.path.join(gt_folder, dir_idx,'image0.jpg')
        gt_image = Image.open(gt_image_path)
        gt_images.append(gt_image)
        if idx % 99 == 0 and idx != 0 :
            gif_num += 1
            imageio.mimsave(os.path.join(output_folder,f'{gif_num}-gt.gif'), gt_images, duration=frame_duration, loop=0)
            imageio.mimsave(os.path.join(output_folder,f'{gif_num}-attack.gif'), attack_images, duration=frame_duration, loop=0)

def main():

    game = args.game
    n_iter = args.n_iter
    epsilon = args.epsilon/255 
    log_dir = os.path.join('result','log','{}'.format(game))
    os.makedirs(log_dir,exist_ok=True)
    log_file = '{}-localM_{}-targetM_{}-lr{}-eps{}-iter{}-universal{}.log'.format(game,args.local_model,args.target_model,args.lr,args.epsilon,args.n_iter,args.use_universal_cloak)
    # logger
    with open(os.path.join(log_dir,log_file),'w') as log:
        logger.add(os.path.join(log_dir,log_file))

    scenario = 'dust2_' + args.scenario
    # Dataset 
    data_loader = load_dataset('cs2_demo',1,scenario=scenario,demo=True)

    # Local Proxy Model
    weights_path = 'pretrained_models/{}.pt'.format(args.local_model)
    from models.common import DetectMultiBackend,AutoShape
    model = DetectMultiBackend(weights_path,device=torch.device(device),fuse=True)
    model = AutoShape(model)  

    # Target Model
    predict_model = torch.hub.load('./', 'custom','pretrained_models/{}.pt'.format(args.target_model),source='local').to(device)
    predict_model.classes = 0
    predict_model.conf = 0.4

    # metrics
    succ_num = .0
    total_num = .0
    fps= 0
    total_time = 0
    total_ssim = []
    total_query = []

    # Start
    logger.info("Attack Begin")
    if not global_noise is None:
        logger.info("{}".format(global_noise_path))
    else:
        logger.info("No Global Noise")

    logger.info("len dataset {}".format(len(data_loader)))

    len_dataset = len(data_loader)
    #  Attack
    total = 0
    demo_save_dir = os.path.join('result','visualization','cs2_demo',args.scenario)
    os.makedirs(demo_save_dir,exist_ok=True)
    for i,(img_path,_,_) in enumerate(data_loader):

        img = np.array(Image.open(img_path[0]), np.uint8)
        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
        imgs = img.to(device, non_blocking=True).float() / 255

        adv_imgs,k,local_suc,t = tog_vanishing(imgs,model,n_iter=n_iter,eps=epsilon,lr=args.lr)
        total_time += t
        
        target_succ = 0
        if 'yolov5' in args.target_model:
            x = adv_imgs[0].detach().permute(1,2,0).cpu()*255
            predict_results = predict_model(np.array(x),size=320)
            result = predict_results.xyxy[0].shape[0]
            if result == 0:
                target_succ = 1
                succ_num += 1.0
        elif 'yolov8' in args.target_model:
            attack_results = predict_model.predict(source=adv_imgs,conf=0.4,verbose=False,classes=0)
            if len(attack_results[0].boxes.conf) == 0:
                target_succ = 1
                succ_num += 1.0      

        ssim_val = ssim( (imgs*255).cpu(), (adv_imgs*255).cpu(), data_range=255, size_average=False)   
        total_ssim.append(ssim_val.item())    
        total_query.append(k)
        total_num += 1.0
        fps = total_num / total_time
        logger.info("Index:{} LSucc:{} TSucc:{} Single_Time:{:.3f} Query:{} SSIM: {:.3f} ".
            format(i,local_suc,target_succ,t,k,ssim_val.item()))
        logger.info("Total:{} DSR:{:.3f} AvgTime:{:.3f} FPS:{:.3f} AvgSSIM: {:.3f} ".
            format(len_dataset,succ_num/total_num,total_time/total_num,fps,np.mean(total_ssim)))

        # visualize inter results
        attack_file_name = os.path.join(demo_save_dir,'attack','{}'.format(i))
        attack_x = adv_imgs[0].detach().permute(1,2,0).cpu()*255
        attack_results = predict_model(np.array(attack_x),size=320)
        attack_results.save(labels=True, save_dir=attack_file_name, exist_ok=True) 
        clean_file_name = os.path.join(demo_save_dir,'gt','{}'.format(i))
        clean_x = imgs[0].detach().permute(1,2,0).cpu()*255
        clean_results = predict_model(np.array(clean_x),size=320)
        clean_results.save(labels=True, save_dir=clean_file_name, exist_ok=True) 
        

    if args.visualize_gif == 1:
        logger.info("Createing Gif every 100 frames")
        create_gif(demo_save_dir)

if __name__ == "__main__":
    main()