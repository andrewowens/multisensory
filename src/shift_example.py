import shift_net, aolib.sound as sound, aolib.img as ig, aolib.util as ut, sep_video, shift_params, numpy as np

# Example using the pretarined audio-visual net. This example predicts
# a class activation map (CAM) for an input video, then saves a
# visualization of the CAM.

pr = shift_params.shift_v1()
model_file = '../results/nets/shift/net.tf-650000'
gpu = None

# uncomment for higher-resolution CAM (like the ones in the paper)
# pr = shift_params.cam_v1()
# model_file = '../results/nets/cam/net.tf-675000'

with ut.VidFrames('../data/crossfire.mp4', sound = True, 
                  start_time = 0., end_time = pr.vid_dur + 2./30, fps = 29.97) \
                  as (im_files, snd_file):
  ims = np.array(map(ig.load, im_files))
  ims = ims[:pr.sampled_frames]
  snd = sound.load_sound(snd_file).normalized()
  samples = snd.samples[:pr.num_samples]
  # make a version of the net using the pretrained weights
  # (i.e. learned through self-supervision)
  clf = shift_net.NetClf(pr, model_file, gpu = gpu)
  # use the audio-visual net to compute a class activation map
  [cam] = clf.predict_cam_resize(ims[np.newaxis], samples[np.newaxis])
  # average the CAM over time and overlay it on the middle frame
  cam = np.abs(cam[0, :, :, :, 0])
  cam = np.mean(cam, 0)
  vis = sep_video.heatmap(ims[len(ims)/2][np.newaxis], cam[np.newaxis], adapt = True)
  ig.save('../results/cam_example.png', vis[0])

