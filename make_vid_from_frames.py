# Extract frames from videos
import os.path as osp
import os, pdb

frame_path = '/media/hdd1/datasets/Adobe240-fps/test_high_fps_frames'
vid_path = '/media/hdd1/datasets/Adobe240-fps/pred_high_fps_videos'
if not osp.exists(vid_path):
	os.makedirs(vid_path)

folder_list = os.listdir(frame_path)
# folder_list = [x for x in folder_list if '30' in x]
for i, folder in enumerate(folder_list):
	# print(osp.splitext(vid)[0])
	# f_path = osp.join(frame_path, osp.splitext(folder)[0])
	pdb.set_trace()
	print(osp.join(frame_path,osp.splitext(folder)[0]))
	os_cmd = 'ffmpeg -framerate 30 -i '+osp.join(frame_path,osp.splitext(folder)[0],osp.splitext(folder)[0]) +'_%04d.jpg '+osp.join(vid_path, osp.splitext(folder)[0])+'.mp4 '
	os.system(os_cmd)
	# print(os_cmd)
	# break