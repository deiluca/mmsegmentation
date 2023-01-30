import os

def get_cmds(dir):
    cmds = []
    for config_file in sorted(os.listdir(dir)):
        config_file = os.path.join(dir, config_file)
        cmds.append(f"python tools/train.py {config_file}")
    return cmds

def main():
    dir = "/home/ws/oc9627/mmsegmentation/configs/segformer/ventricle_seg_configs_larger_arch"
    cmds =get_cmds(dir)
    for cmd in cmds:
        print(cmd)
        os.system(cmd)
        # break

main()