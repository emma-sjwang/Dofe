import os
import random
from shutil import copyfile

if __name__ == '__main__':

    root_path = "/home/sjwang/ImsightMed/MIDL2018/domain_generalization/data/"

    for i in range(4):

        test_folder = os.path.join(root_path, "Domain%d" % (i+1))

        train_root_path = os.path.join(root_path, "disc_cup_split/train%d" % (i+1))
        test_root_path = os.path.join(root_path, "disc_cup_split/test%d" % (i+1))
        val_root_path = os.path.join(root_path, "disc_cup_split/val%d" % (i+1))
        if not os.path.exists(train_root_path):
            os.makedirs(train_root_path)
        if not os.path.exists(test_root_path):
            os.makedirs(test_root_path)
        if not os.path.exists(val_root_path):
            os.makedirs(val_root_path)

        image_folder = os.path.join(test_folder, "disc_cup_ROIs/image")
        mask_folder = os.path.join(test_folder, "disc_cup_ROIs/mask")
        filelist = os.listdir(image_folder)
        for name in filelist:
            img_path = os.path.join(image_folder, name)
            mask_path = os.path.join(mask_folder, name)
            dst_img = os.path.join(test_root_path, 'image', name)
            dst_mask = os.path.join(test_root_path, 'mask', name)
            if not os.path.exists(os.path.dirname(dst_img)):
                os.makedirs(os.path.dirname(dst_img))
            if not os.path.exists(os.path.dirname(dst_mask)):
                os.makedirs(os.path.dirname(dst_mask))
            copyfile(img_path, dst_img)
            copyfile(mask_path, dst_mask)

        for j in range(4):
            if i == j:  # test folder
                continue
            # train or val
            image_folder = os.path.join(root_path, "Domain%d" % (j+1), "disc_cup_ROIs/image")
            mask_folder = os.path.join(root_path, "Domain%d" % (j+1), "disc_cup_ROIs/mask")

            filelist = os.listdir(image_folder)
            for name in filelist:
                img_path = os.path.join(image_folder, name)
                mask_path = os.path.join(mask_folder, name)
                seed = random.random()
                if seed > 0.8:
                    dst_img = os.path.join(val_root_path, 'image', name)
                    dst_mask = os.path.join(val_root_path, 'mask', name)
                else:
                    dst_img = os.path.join(train_root_path, 'image', name)
                    dst_mask = os.path.join(train_root_path, 'mask', name)

                if not os.path.exists(os.path.dirname(dst_img)):
                    os.makedirs(os.path.dirname(dst_img))
                if not os.path.exists(os.path.dirname(dst_mask)):
                    os.makedirs(os.path.dirname(dst_mask))
                copyfile(img_path, dst_img)
                copyfile(mask_path, dst_mask)