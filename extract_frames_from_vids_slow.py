# Extract frames from videos

import os.path as osp
import os, pdb

vid_path = '/media/hdd1/datasets/Adobe240-fps/test_high_fps_videos'
out_path = '/media/hdd1/datasets/Adobe240-fps/test_high_fps_frames'
if not osp.exists(out_path):
	os.makedirs(out_path)

vid_list = os.listdir(vid_path)
for i, vid in enumerate(vid_list):
	# print(osp.splitext(vid)[0])
	folder_path = osp.join(out_path,osp.splitext(vid)[0])
	slow_path = osp.join(out_path.replace('high', 'low'),osp.splitext(vid)[0].replace('240', '30'))
	if not osp.exists(folder_path):
		os.makedirs(folder_path)
	if not osp.exists(slow_path):
		os.makedirs(slow_path)
	os_cmd = 'ffmpeg -i '+osp.join(vid_path,vid)+' '+osp.join(folder_path, osp.splitext(vid)[0])+'_%04d.jpg -hide_banner'
	os.system(os_cmd)
	os_cmd_slow = 'ffmpeg -i '+osp.join(vid_path,vid)+' -vf \"select=not(mod(n\\,8))\" -vsync vfr -q:v 2 '+osp.join(slow_path, osp.splitext(vid)[0].replace('240', '30'))+'_%04d.jpg -hide_banner'
	os.system(os_cmd_slow)
	# print(os_cmd)
	# break