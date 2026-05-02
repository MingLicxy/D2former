
# train
CUDA_VISIBLE_DEVICES=1 python /home/caoxinyu/UNet-based/Xformer-main/basicsr/train.py -opt /home/caoxinyu/UNet-based/Xformer-main/options/INRUNet/GaussianGrayscaleDenoising_SRNOSwinUNetSigma15_4668.yml


# test
CUDA_VISIBLE_DEVICES=3 python /home/caoxinyu/UNet-based/Xformer-main/basicsr/train.py -opt /home/caoxinyu/UNet-based/Xformer-main/options/INRUNet/GaussianGrayscaleDenoising_INRSwinUNetSigma15.yml