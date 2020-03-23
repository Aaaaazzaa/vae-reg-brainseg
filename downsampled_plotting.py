# from nibabel.processing import resample_to_output
import nibabel as nib
# from nilearn.image import resample_img
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

src_img = nib.load('./scratch/test_out.src.nii.gz')
src_img_data = src_img.get_fdata()

etgt_img = nib.load('./scratch/test_out.tc_gt.nii.gz')
etgt_img_data = etgt_img.get_fdata()

wtgt_img = nib.load('./scratch/test_out.et_gt.nii.gz')
wtgt_img_data = etgt_img.get_fdata()

fig, axs = plt.subplots(1, 2)
ims = []
for i in range(128):
    blend_list = []
    blend_list.append(axs[0].imshow(src_img_data[:,:, i].T, cmap="gray", origin="lower", animated=True))
    blend_list.append(axs[0].set_title("test src"))
    blend_list.append(axs[1].imshow(src_img_data[:,:, i].T, cmap="gray", origin="lower", animated=True))
    blend_list.append(axs[1].imshow(etgt_img_data[:, :, i].T, cmap="Reds", origin="lower", animated=True, alpha=.4))
    blend_list.append(axs[1].set_title("test tc_gt"))
    ims.append(blend_list)

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=3000)

ani.save('dynamic_images.gif')

# plt.show()
import random
# import torch
# from model import vaereg
# def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords=None):
#     """
#     This function uses a generic approach to rescaling an affine to arbitrary
#     voxel dimensions. It allows for affines with off-diagonal elements by
#     decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
#     and applying the scaling to the scaling matrix (s).
#
#     Parameters
#     ----------
#     input_affine : np.array of shape 4,4
#         Result of nibabel.nifti1.Nifti1Image.affine
#     voxel_dims : list
#         Length in mm for x,y, and z dimensions of each voxel.
#     target_center_coords: list of float
#         3 numbers to specify the translation part of the affine if not using the same as the input_affine.
#
#     Returns
#     -------
#     target_affine : 4x4matrix
#         The resampled image.
#     """
#     # Initialize target_affine
#     target_affine = input_affine.copy()
#     # Decompose the image affine to allow scaling
#     u, s, v = np.linalg.svd(target_affine[:3, :3], full_matrices=False)
#
#     # Rescale the image to the appropriate voxel dimensions
#     s = voxel_dims
#
#     # Reconstruct the affine
#     target_affine[:3, :3] = u @ np.diag(s) @ v
#
#     # Set the translation component of the affine computed from the input
#     # image affine if coordinates are specified by the user.
#     if target_center_coords is not None:
#         target_affine[:3, 3] = target_center_coords
#     return target_affine
#
# newaffine = epi_img.affine.copy()
# # newaffine /= 2
# newaffine[:3, :3] *= 2
#
# # data_out = resample_to_output(epi_img, .5)
# data_out = resample_img(epi_img, target_affine=newaffine, target_shape=(26, 30, 16))
# print(type(data_out.get_fdata()))
# plt.subplot(1,2,1)
# plt.imshow(data_out.get_fdata()[:,:, 15].T, cmap="gray", origin="lower")
# plt.subplot(1,2,2)
# plt.imshow(epi_img_data[:,:, 30].T, cmap="gray", origin="lower")
# plt.show()
# random.seed(0)
# labeled_list = [random.randint(0,31) for _ in range(int(32*.5))]
# model = vaereg.UNet()
# model.load_state_dict(torch.load('./checkpoints/baseline/baseline', map_location=torch.device('cpu'))['model_state_dict'])
# model.eval()

# print(labeled_list)