import os, shutil
import torch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import lpips


def save_model_samples(name, net, data_iterator, batch_size, num_samples, device, data2_iterator):
    if os.path.exists(name):
        shutil.rmtree(name)

    os.makedirs(name, exist_ok=True)
    count = 0
    l2_sum = 0
    lpips_sum = 0

    loss_fn = lpips.LPIPS(net='vgg').to(device)

    with tqdm(range(num_samples)):
        while count < num_samples:
            cur_batch_size = min(num_samples - count, batch_size)
            batch = next(data_iterator)
            X = batch[0][:cur_batch_size].to(device).to(torch.float32) / 127.5 - 1
            with torch.no_grad():
                latent_z = torch.randn(X.shape[0], net.model.nz, device=X.device)*0.1
            T_X = net(X, latent_z)
            l2_sum += F.mse_loss(X, T_X, reduction='sum')
            lpips_sum += loss_fn(X.clamp(-1,1), T_X.clamp(-1,1)).sum()
            out = (T_X.clamp(-1,1) * 127.5 + 128).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(out.shape[0]):
                img = Image.fromarray(out[i])
                n_digits = len(str(count))
                img_name = (6 - n_digits) * '0' + str(count) + '.png'
                img.save(os.path.join(name, img_name))
                count += 1
                
    batch = next(data_iterator)
    X = batch[0][:8].to(device).to(torch.float32) / 127.5 - 1
    with torch.no_grad():
        latent_z = torch.randn(X.shape[0], net.model.nz, device=X.device)*0.1
    T_X = net(X, latent_z)
    batch = next(data2_iterator)
    Y = batch[0][:8].to(device).to(torch.float32) / 127.5 - 1
    
    X = (X.clamp(-1,1) * 127.5 + 128).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    Y = (Y.clamp(-1,1) * 127.5 + 128).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    T_X = (T_X.clamp(-1,1) * 127.5 + 128).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    
    X_array = []
    Y_array = []
    T_X_array = []
    for i in range(X.shape[0]):
        X_array.append(Image.fromarray(X[i]))
        Y_array.append(Image.fromarray(Y[i]))
        T_X_array.append(Image.fromarray(T_X[i]))
        
    # wandb.log({"examples": [wandb.Image(image) for image in images]})
    return l2_sum/count, lpips_sum/count, X_array, T_X_array, Y_array