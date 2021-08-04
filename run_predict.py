import os
import argparse
import torch

from models.regressor import SingleInputRegressor
from models.smpl_official import SMPL
from predict.predict_3D import predict_3D, setup_detectron2_predictors
import config

from tqdm import tqdm

def main(input_path, checkpoint_path, device, silhouettes_from, output_dir):
    
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)

    # print("Regressor loaded. Weights from:", checkpoint_path)
    regressor.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    regressor.eval()

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    os.makedirs(output_dir, exist_ok=True)
    video_files = os.listdir(input_path)
    video_paths = [os.path.join(input_path, video_file) for video_file in video_files]
    for video_path in tqdm(video_paths, desc='video'):
        basename = os.path.basename(video_path)
        videoname = basename.split(".")[0]
        predict_3D(video_path, 
                    regressor, 
                    device, 
                    silhouettes_from=silhouettes_from, 
                    joints2D_predictor=joints2D_predictor, 
                    silhouette_predictor=silhouette_predictor,
                    smpl=smpl,
                    save_proxy_vis=True, 
                    render_vis=True, 
                    output_dir=output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--output', type=str, help='Path to output image/folder of images.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--silh_from', choices=['densepose', 'pointrend'])
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Regarding body mesh visualisation using pyrender:
    # If you are running this script on a remote machine via ssh, you might need to use EGL
    # to create an OpenGL context. If EGL is installed on the remote machine, uncommenting the
    # following line should work.
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # If this still doesn't work, just disable rendering visualisation by setting render_vis
    # argument in predict_3D to False.

    main(args.input, args.checkpoint, device, args.silh_from, args.output)
