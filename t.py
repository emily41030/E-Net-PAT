import torch
import PIL
from torchvision import transforms
import numpy as np


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    sum1 = torch.sum(torch.Tensor(split_sizes))
    if dim_size != sum1:
        raise KeyError("Sum of split sizes does not equal tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


img = PIL.Image.open("SR_result_1.png")
trans = transforms.Compose([transforms.ToTensor()])
a = trans(img)
a = torch.unsqueeze(a, 0)

split_sizes = []
for i in range(int(512/16)):
    split_sizes.append(16)

dim = 2
res = size_splits(tensor=a, split_sizes=split_sizes, dim=dim)

newa = torch.Tensor(0, 0, 0, 0)
for i in range(res.__len__()):
    newa = torch.cat((res[i], newa), 0)

res = size_splits(tensor=newa, split_sizes=split_sizes, dim=3)
img2 = res[0]
oi2 = img2.numpy()
oi2[np.where(oi2 < 0)] = 0.0
oi2[np.where(oi2 > 1)] = 1.0
save_img2 = torch.from_numpy(oi2)
save_img2 = transforms.ToPILImage()(save_img2[0])
save_img2.show()

newa = torch.Tensor(0, 0, 0, 0)
for i in range(res.__len__()):
    newa = torch.cat((res[i], newa), 0)


# img2 = res[0]
# oi2 = img2.numpy()
# oi2[np.where(oi2 < 0)] = 0.0
# oi2[np.where(oi2 > 1)] = 1.0
# save_img2 = torch.from_numpy(oi2)
# save_img2 = transforms.ToPILImage()(save_img2[0])
# save_img2.show()


for r, split_size in zip(res, split_sizes):
    assert r.size(dim) == split_size

try:
    res = size_splits(tensor=a, split_sizes=split_sizes, dim=3)
    # print(res)

    img2 = res[0]
    oi2 = img2.numpy()
    oi2[np.where(oi2 < 0)] = 0.0
    oi2[np.where(oi2 > 1)] = 1.0
    save_img2 = torch.from_numpy(oi2)
    save_img2 = transforms.ToPILImage()(save_img2[0])
    save_img2.show()


except KeyError as e:
    print(e)


# a = torch.randn(20, 10, 2, 2)

# # split_sizes = [2, 2, 6]
# # dim = 1
# # res = size_splits(tensor=a, split_sizes=split_sizes, dim=dim)

# # for r, split_size in zip(res, split_sizes):
# #     assert r.size(dim) == split_size

# # try:
# #     res = size_splits(tensor=a, split_sizes=split_sizes, dim=0)
# # except KeyError as e:
# #     print(e)

# split_sizes = [5, 5, 10]
# dim = 0
# res = size_splits(tensor=a, split_sizes=split_sizes, dim=dim)

# for r, split_size in zip(res, split_sizes):
#     assert r.size(dim) == split_size

# try:
#     res = size_splits(tensor=a, split_sizes=split_sizes, dim=0)
#     print(res)
# except KeyError as e:
#     print(e)
