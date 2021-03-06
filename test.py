import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import logger, pdb
import numpy as np
import math

run_num = 7
# tb_path = '/media/hdd1/shashant/super_slomo/tb_logs/'
# model_path = os.path.join('/media/hdd1/shashant/super_slomo/baselines/')
model_path = os.path.join('/media/hdd1/shashant/super_slomo/models',str(run_num))

# if not os.path.exists(tb_path):
# 	os.makedirs(tb_path)
# if not os.listdir(tb_path):
#     run_num = 1
# else:    
# 	run_num = max([int(x) for x in os.listdir(tb_path)]) + 1
global_step = 0

class FlowWarper(nn.Module):
	def __init__(self, w, h):
		super(FlowWarper, self).__init__()
		x = np.arange(0,w)
		y = np.arange(0,h)
		gx, gy = np.meshgrid(x,y)
		self.w = w
		self.h = h
		self.grid_x = torch.autograd.Variable(torch.Tensor(gx), requires_grad=False).cuda()
		self.grid_y = torch.autograd.Variable(torch.Tensor(gy), requires_grad=False).cuda()

	def forward(self, img, uv):
		u = uv[:,0,:,:]
		v = uv[:,1,:,:]
		X = self.grid_x.unsqueeze(0).expand_as(u) + u # what's happening here?
		Y = self.grid_y.unsqueeze(0).expand_as(v) + v
		X = 2*(X/self.w - 0.5)  # normalize and mean = 0
		Y = 2*(Y/self.h - 0.5)
		grid_tf = torch.stack((X,Y), dim=3)
		img_tf = torch.nn.functional.grid_sample(img, grid_tf) # bilinear interp
		return img_tf

def test():
	flowModel = model.UNet_flow().cuda()
	interpolationModel = model.UNet_refine().cuda()
	# Load pretrained flowModel and interpolationModel
	flow_model = os.path.join(model_path, 'checkpoint_flow_14_1999.pt')
	interp_model = os.path.join(model_path, 'checkpoint_interp_14_1999.pt')
	flow_chkpt = torch.load(flow_model)
	interp_chkpt = torch.load(interp_model)
	flowModel.load_state_dict(flow_chkpt)
	interpolationModel.load_state_dict(interp_chkpt)

	# ### ResNet for Perceptual Loss
	# res50_model = torchvision.models.resnet18(pretrained=True)
	# res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
	# res50_conv.cuda()

	# for param in res50_conv.parameters():

	# 	param.requires_grad = False

	dataFeeder = dataloader.testLoader('/media/hdd1/datasets/Adobe240-fps/my_high_fps_frames/slomo_rgb/000055', mode='test')
	test_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=1, 
											  shuffle=False, num_workers=1,
											  pin_memory=True)

	flowModel.eval()
	interpolationModel.eval()

	warper = FlowWarper(352,352)

	count = 0
	for i, (imageList) in enumerate(test_loader):
		I0_var = torch.autograd.Variable(imageList[0]).cuda()
		I1_var = torch.autograd.Variable(imageList[-1]).cuda()
		flow_out_var = flowModel(I0_var, I1_var)
		F_0_1 = flow_out_var[:,:2,:,:]
		F_1_0 = flow_out_var[:,2:,:,:]
		np.save('./opt_01_0.npy', F_0_1.detach().cpu().numpy())
		np.save('./opt_10_0.npy', F_1_0.detach().cpu().numpy())
		image_collector = []
		for t_ in range(1,8):
			t = t_/8

			F_t_0 = -(1-t)*t*F_0_1 + t*t*F_1_0
			
			F_t_1 = (1-t)*(1-t)*F_0_1 - t*(1-t)*(F_1_0)

			np.save('./opt_t0_'+str(t_)+'.npy', F_t_0.detach().cpu().numpy())
			np.save('./opt_t1_'+str(t_)+'.npy', F_t_1.detach().cpu().numpy())
			
			
			g_I0_F_t_0 = warper(I0_var, F_t_0)
			g_I1_F_t_1 = warper(I1_var, F_t_1)

			interp_out_var = interpolationModel(I0_var, I1_var, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1)
			F_t_0_final = interp_out_var[:,:2,:,:] + F_t_0
			F_t_1_final = interp_out_var[:,2:4,:,:] + F_t_1
			V_t_0 = torch.unsqueeze(interp_out_var[:,4,:,:],1)
			V_t_1 = 1 - V_t_0

			g_I0_F_t_0_final = warper(I0_var, F_t_0_final)
			g_I0_F_t_1_final = warper(I1_var, F_t_1_final)

			normalization = (1-t)*V_t_0 + t*V_t_1
			interpolated_image_t_pre = (1-t)*V_t_0*g_I0_F_t_0_final + t*V_t_1*g_I0_F_t_1_final
			interpolated_image_t = interpolated_image_t_pre / normalization
			image_collector.append(interpolated_image_t)
		save_path = os.path.join('/media/hdd1/datasets/Adobe240-fps/pred_high_fps_frames',str(run_num))
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		torchvision.utils.save_image((I0_var),os.path.join(save_path, str(count).zfill(4) +'.jpg'),normalize=True)
		count += 1
		for jj,image in enumerate(image_collector):
			torchvision.utils.save_image((image),os.path.join(save_path, str(count).zfill(4)+'.jpg'),normalize=True)
			count += 1
		torchvision.utils.save_image((I1_var),os.path.join(save_path, str(count).zfill(4) +'.jpg'),normalize=True)
		count += 1
	return save_path
						
def make_vid(save_path):
	frame_path = save_path
	vid_path = save_path.replace('frames/', 'videos/')
	if not os.path.exists(vid_path):
		os.makedirs(vid_path)
	os_cmd = 'ffmpeg -framerate 60 -i '+frame_path +'/%04d.jpg '+os.path.join(vid_path)+'/test_sim1.mp4 '
	# print(os_cmd)
	os.system(os_cmd)




if __name__ == '__main__':
	# train_val()
	# save_path = os.path.join('/media/hdd1/datasets/Adobe240-fps/pred_high_fps_frames',str(run_num))
	save_path = test()
	make_vid(save_path)

