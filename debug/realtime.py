from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

import time
import pyrealsense2 as rs

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = '../pretrained_models/wilor_final.ckpt' , cfg_path= '../pretrained_models/model_config.yaml')
    detector = YOLO('../pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()


    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution and framerate as needed

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            img_cv2 = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            start = time.time()
            detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
            print("detection time ", time.time() - start)
            bboxes    = []
            is_right  = []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if len(bboxes) == 0:
                continue
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            all_joints= []
            all_kpts  = []

            for batch in dataloader:
                batch = recursive_to(batch, device)

                start = time.time()
                with torch.no_grad():
                    out = model(batch)
                print("hand time: ", time.time() - start)

                multiplier    = (2*batch['right']-1)
                pred_cam      = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center    = batch["box_center"].float()
                box_size      = batch["box_size"].float()
                img_size      = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()


                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    verts  = out['pred_vertices'][n].detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                    is_right    = batch['right'][n].cpu().numpy()
                    verts[:,0]  = (2*is_right-1)*verts[:,0]
                    joints[:,0] = (2*is_right-1)*joints[:,0]
                    cam_t = pred_cam_t_full[n]
                    kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])

                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    all_joints.append(joints)
                    all_kpts.append(kpts_2d)

            # Render front view
            if len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_PURPLE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                # Display the detection results
                cv2.imshow("YOLO Hand Detection", input_img_overlay)

            # Exit the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming and release resources
        pipeline.stop()
        cv2.destroyAllWindows()


def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()
