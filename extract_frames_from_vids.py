# Extract frames from videos

import os.path as osp
import os

vid_path = '/media/hdd1/datasets/Adobe240-fps/original_high_fps_videos'
out_path = '/media/hdd1/datasets/Adobe240-fps/original_high_fps_frames'
if not osp.exists(out_path):
	os.makedirs(out_path)

vid_list = os.listdir(vid_path)
for vid in vid_list:
	# print(osp.splitext(vid)[0])
	folder_path = osp.join(out_path,osp.splitext(vid)[0])
	if not osp.exists(folder_path):
		os.makedirs(folder_path)
	os_cmd = 'ffmpeg -i '+osp.join(vid_path,vid)+' '+osp.join(folder_path, osp.splitext(vid)[0])+'_%04d.jpg -hide_banner'
	os.system(os_cmd)
	# print(os_cmd)
	# break
