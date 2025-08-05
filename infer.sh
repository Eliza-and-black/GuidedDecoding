python3 inference.py \
    --eval \
    --resolution half \
    --dataset nyu_reduced \
    --weights_path /cv_project/GuidedDecoding/weights/NYU_Half_GuideDepth.pth \
    --save_results /cv_project/GuidedDecoding/results \
    --test_path /cv_project/GuidedDecoding/test_data \
    --model GuideDepth 