import time
import os
import argparse

import torch
import torchvision
import tensorrt as trt
# from torch2trt import torch2trt
import torch_tensorrt
torch_tensorrt.dynamo._compiler.CompilationSettings(truncate_long_and_double=True)
import matplotlib.pyplot as plt

from data import datasets
from model import loader
from metrics import AverageMeter, Result
# from data import transforms
from torchvision import transforms

max_depths = {
    'kitti': 80.0,
    'nyu' : 10.0,
    'nyu_reduced' : 10.0,
}
nyu_res = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}
kitti_res = {
    'full' : (384, 1280),
    'half' : (192, 640)}
resolutions = {
    'nyu' : nyu_res,
    'nyu_reduced' : nyu_res,
    'kitti' : kitti_res}
crops = {
    'kitti' : [128, 381, 45, 1196],
    'nyu' : [20, 460, 24, 616],
    'nyu_reduced' : [20, 460, 24, 616]}


def get_args():
    parser = argparse.ArgumentParser(description='Nano Inference for Monocular Depth Estimation')

    #Mode
    parser.set_defaults(evaluate=False)
    parser.add_argument('--eval',
                        dest='evaluate',
                        action='store_true')

    #Data
    parser.add_argument('--test_path',
                        type=str,
                        help='path to test data')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset for training',
                        choices=['kitti', 'nyu', 'nyu_reduced'],
                        default='kitti')
    parser.add_argument('--resolution',
                        type=str,
                        help='Resolution of the images for training',
                        choices=['full', 'half'],
                        default='half')


    #Model
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to be trained',
                        default='GuideDepth')
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to model weights')
    parser.add_argument('--save_results',
                        type=str,
                        help='path to save results to',
                        default='./results')

    #System
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of dataloader workers',
                        default=1)


    return parser.parse_args()


def inverse_depth_norm(args, depth):
    maxDepth = max_depths[args.dataset]
    depth = maxDepth / depth
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    return depth


if __name__ == '__main__':
    args = get_args()
    print(args)

    # engine = Inference_Engine(args)
    def load_model(model_path):
        """加载模型"""
        model = loader.load_model('GuideDepth', model_path)
        model = model.eval().cuda()
        return model

    def infer_single_image(args, image_path, model, save_dir):
        from PIL import Image
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        img_size = resolutions[args.dataset][args.resolution]
        print(img_size)
    
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(img_size),  # 调整图像大小
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).cuda()
        # img_tensor = img_tensor.half()
        # model = model.half()
        # 推理
        with torch.no_grad():
            pred = model(img_tensor)
            if 1:
                # 测速：推理100次
                start_time = time.time()
                for _ in range(1000):
                    _ = model(img_tensor)
                torch.cuda.synchronize()  # 确保GPU运算完成
                end_time = time.time()
                
                # 计算平均推理时间
                total_time = end_time - start_time
                avg_time = total_time / 1000
                print(f"Average inference time over 1000 runs: {avg_time*1000:.2f} ms")
                
        # 将预测结果转换为numpy数组
        pred_depth = inverse_depth_norm(args, pred)
        pred_depth = pred_depth.squeeze().cpu().numpy()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制原图和深度图
        save_path = os.path.join(save_dir, os.path.basename(image_path).split('.')[0] + '_depth_result.png')
        plt.figure(figsize=(15, 7))
        
        # 绘制原图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # 绘制深度图
        plt.subplot(1, 2, 2)
        depth_plot = plt.imshow(pred_depth, cmap='viridis')
        plt.colorbar(depth_plot, label='Depth')
        plt.title('Depth Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_image_paths(input_path):
        """
        解析输入路径，返回所有图片文件的完整路径列表
        Args:
            input_path: 单个图片路径或图片文件夹路径
        Returns:
            image_paths: 图片完整路径列表
        """
        image_paths = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
        
        if os.path.isfile(input_path):
            # 单个文件，检查是否为图片
            if input_path.lower().endswith(image_extensions):
                image_paths.append(input_path)
        elif os.path.isdir(input_path):
            # 文件夹，遍历所有图片文件
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"Warning: No valid images found in {input_path}")
            
        return sorted(image_paths)

    # 使用示例
    if args.weights_path and args.test_path:
        # 加载模型
        model = load_model(args.weights_path)
        # 获取图片路径
        image_paths = get_image_paths(args.test_path)
        # 遍历处理每张图片
        for image_path in image_paths:
            infer_single_image(args, image_path, model, args.save_results)
