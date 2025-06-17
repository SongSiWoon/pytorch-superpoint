"""This is the frontend interface for training
base class: inherited by other Train_model_*.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging
import os
import glob

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path


def thd_img(img, thd=0.015):
    """
    thresholding the image.
    :param img:
    :param thd:
    :return:
    """
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_frontend(object):
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.
    
    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 10000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]
        
        :param config:
            dense_loss, sparse_loss (default)
            
        :param save_path:
        :param device:
        :param verbose:
        """
        # config
        print("Load Train_model_frontend!!")
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False
        self.loss = 0

        self.max_iter = config["train_iter"]

        if self.config["model"]["dense_loss"]["enable"]:
            ## original superpoint paper uses dense loss
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        if self.config["model"]["subpixel"]["enable"]:
            ## deprecated: only for testing subpixel prediction
            self.subpixel = True

            def get_func(path, name):
                logging.info("=> from %s import %s", path, name)
                mod = __import__("{}".format(path), fromlist=[""])
                return getattr(mod, name)

            self.subpixel_loss_func = get_func(
                "utils.losses", self.config["model"]["subpixel"]["loss_func"]
            )

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()

        pass

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(
            self.net, lr=self.config["model"]["learning_rate"]
        )
        pass

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("adam optimizer")
        import torch.optim as optim

        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        # SuperPointNet_gauss2는 config 매개변수를 받지 않으므로 조건부로 처리
        if model != "SuperPointNet_gauss2":
            params["config"] = self.config  # config를 params에 추가
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")
        optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
            pass
        else:
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode=mode, full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)
        pass


    @property
    def writer(self):
        """
        # writer for tensorboard
        :return:
        """
        # print("get writer")
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **options):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        running_losses = []
        epoch = 0
        best_loss = float('inf')
        
        # Train one epoch
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # train one sample
                loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                self.n_iter += 1
                running_losses.append(loss_out)
                
                # run validation
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    val_losses = []
                    for j, sample_val in enumerate(self.val_loader):
                        val_loss = self.train_val_sample(sample_val, self.n_iter + j, False)
                        val_losses.append(val_loss)
                        if j > self.config.get("validation_size", 3):
                            break
                    
                    # validation loss의 평균 계산
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    
                    # best 모델 저장
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        logging.info("New best model! Loss: %f", best_loss)
                        self.saveModel(is_best=True)
                
                # ending condition
                if self.n_iter > self.max_iter:
                    # end training
                    logging.info("End training: %d", self.n_iter)
                    break

        pass

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        """
        ## deprecated: loss function
        :param semi:
        :param labels3D_in_loss:
        :param mask_3D_flattened:
        :param device:
        :return:
        """
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        # if self.config['data']['gaussian_label']['enable']:
        #     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
        #     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
        # else:
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # deprecated: default train_val_sample
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]

        losses = {}
        ## get the inputs
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        img_warp, labels_warp_2D, mask_warp_2D = (
            sample["warped_img"],
            sample["warped_labels"],
            sample["warped_valid_mask"],
        )

        # homographies
        mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            outs, outs_warp = (
                self.net(img.to(self.device)),
                self.net(img_warp.to(self.device), subpixel=self.subpixel),
            )
            semi, coarse_desc = outs[0], outs[1]
            semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        else:
            with torch.no_grad():
                outs, outs_warp = (
                    self.net(img.to(self.device)),
                    self.net(img_warp.to(self.device), subpixel=self.subpixel),
                )
                semi, coarse_desc = outs[0], outs[1]
                semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
                pass

        # detector loss
        ## get labels, masks, loss for detection
        labels3D_in_loss = self.getLabels(labels_2D, self.cell_size, device=self.device)
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.get_loss(
            semi, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        ## warping
        labels3D_in_loss = self.getLabels(
            labels_warp_2D, self.cell_size, device=self.device
        )
        mask_3D_flattened = self.getMasks(
            mask_warp_2D, self.cell_size, device=self.device
        )
        loss_det_warp = self.get_loss(
            semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        mask_desc = mask_3D_flattened.unsqueeze(1)

        # descriptor loss
        loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
            coarse_desc,
            coarse_desc_warp,
            mat_H,
            mask_valid=mask_desc,
            device=self.device,
            **self.desc_params
        )

        loss = (
            loss_det + loss_det_warp + self.config["model"]["lambda_loss"] * loss_desc
        )

        if self.subpixel:
            # coarse to dense descriptor
            # work on warped level
            dense_map = flattenDetection(semi_warp)  # tensor [batch, 1, H, W]
            # concat image and dense_desc
            concat_features = torch.cat(
                (img_warp.to(self.device), dense_map), dim=1
            )  # tensor [batch, n, H, W]
            # prediction
            pred_heatmap = outs_warp[2]  # tensor [batch, 1, H, W]

            # loss
            labels_warped_res = sample["warped_res"]
            subpix_loss = self.subpixel_loss_func(
                labels_warp_2D.to(self.device),
                labels_warped_res.to(self.device),
                pred_heatmap.to(self.device),
                patch_size=11,
            )

            # extract the patches from labels
            label_idx = labels_2D[...].nonzero()
            from utils.losses import extract_patches

            patch_size = 32
            patches = extract_patches(
                label_idx.to(self.device),
                img_warp.to(self.device),
                patch_size=patch_size,
            )  # tensor [N, patch_size, patch_size]

            def label_to_points(labels_res, points):
                labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
                points_res = labels_res[
                    points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
                ]  # tensor [N, 2]
                return points_res

            points_res = label_to_points(labels_warped_res, label_idx)

            num_patches_max = 500
            # feed into the network
            pred_res = self.subnet(
                patches[:num_patches_max, ...].to(self.device)
            )  # tensor [1, N, 2]

            # loss function
            def get_loss(points_res, pred_res):
                loss = points_res - pred_res
                loss = torch.norm(loss, p=2, dim=-1).mean()
                return loss

            loss = get_loss(points_res[:num_patches_max, ...].to(self.device), pred_res)

            losses.update({"subpix_loss": subpix_loss})

        self.loss = loss

        losses.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        if train:
            loss.backward()
            self.optimizer.step()

        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )
            self.printLosses(losses, task)

        return loss.item()

    def saveModel(self, is_best=False):
        if not is_best:
            return

        model_state_dict = self.net.module.state_dict()
        best_filename = "superPointNet_best_checkpoint.pth.tar"
        best_saved_path = save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            "best"
        )

        # === 백본만 따로 저장 ===
        if hasattr(self.net.module, 'save_backbone'):
            self.net.module.save_backbone(self.save_path)
        else:
            print("[경고] save_backbone 메서드가 없습니다. 백본 저장이 생략됩니다.")

        if self.config.get("wandb", {}).get("enable", False):
            upload_interval = 10000  # 원하는 업로드 간격(예: 10000)
            if self.n_iter % upload_interval == 0 or is_best:
                import wandb
                wandb.save(best_saved_path)
                # 필요하면 백본도 업로드
                # wandb.save(backbone_path)

        # 기존 체크포인트 삭제 (best만 남김)
        for f in glob.glob(os.path.join(self.save_path, "*.pth*")):
            if not f.endswith(best_filename) and f != best_saved_path:
                os.remove(f)

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        # TensorBoard 관련 코드 제거
        pass

    # tensorboard
    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        # TensorBoard 관련 코드 제거
        pass

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        # TensorBoard 관련 코드 제거
        pass

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        # TensorBoard 관련 코드 제거
        pass

    def tb_hist_dict(self, task, tb_dict):
        # TensorBoard 관련 코드 제거
        pass

    def printLosses(self, losses, task="training"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            print(task, "-", element, ": ", losses[element].item())

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        # TensorBoard 관련 코드 제거
        pass

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        # for e in list(sample):
        #     print("sample[e]", sample[e].shape)
        #     if (sample[e]).dim() == 4:
        #         tb_images_dict[e] = sample[e]
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_test.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_frontend(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()
    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
