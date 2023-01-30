import os

def get_cmds(dir):
    checkpoints = {
        'split0': '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-0/best_mDice_iter_960.pth',
        'split1': '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-1/best_mDice_iter_910.pth',
        'split2': '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-2/best_mDice_iter_980.pth',
        'split3': '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-3/best_mDice_iter_920.pth',
        'split4': '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-4/best_mDice_iter_880.pth'
        }
    cmds = []
    for i, config_file in enumerate(sorted(os.listdir(dir))):
        config_file = os.path.join(dir, config_file)
        checkpoint = checkpoints[f'split{i}']
        cmds.append(f"python tools/train.py {config_file} --load-from {checkpoint}")
    return cmds

def main():
    dir = "/home/ws/oc9627/mmsegmentation/configs/segformer/2d_ms_org_seg_inference_run4"
    cmds =get_cmds(dir)
    for cmd in cmds:
        print(cmd)
        os.system(cmd)
        # break

main()