import torch
import torch.nn.functional as F
import math
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import torch.nn as nn

from torchvision import transforms

smallest_float32 = torch.finfo(torch.float32).tiny
largest_float32 = torch.finfo(torch.float32).max

# Define your neural network
class JPEG_layer(nn.Module):

    def construct(self, opt):
        self.num_bit = opt.num_bit
        self.Q_inital = opt.Q_inital
        self.batch_size = opt.batch_size
        self.block_size = opt.block_size
        # self.num_channels = self.img_shape[-1]
        self.num_channels = 3
        self.num_block = int((self.img_shape[0]*self.img_shape[1])/(self.block_size**2))
        self.min_Q_Step = opt.min_Q_Step
        self.max_Q_Step = opt.max_Q_Step
        self.analysis = False
        self.low_level  = -2**(self.num_bit-1) + 1
        self.high_level =  2**(self.num_bit-1)

        self.num_non_zero_q = opt.num_non_zero_q
        self.num_non_zero_q_on_1_side = math.floor(self.num_non_zero_q/2)
        self.q_idx = torch.arange(0, self.num_non_zero_q)

        if torch.cuda.is_available():
            self.q_idx = self.q_idx.cuda()
        
    
    def __init__(self, opt, img_shape, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]):
        super(JPEG_layer, self).__init__()
        self.JPEG_alpha_trainable = opt.JPEG_alpha_trainable
        self.img_shape = img_shape
        
        self.construct(opt)

        self.lum_qtable = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32).to(device)
        self.chrom_qtable = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32).to(device)

        nn.init.constant_(self.lum_qtable , self.Q_inital)
        nn.init.constant_(self.chrom_qtable , self.Q_inital)

        # QT_Y = torch.tensor(quantizationTable(QF=50, Luminance=True))
        # QT_C = torch.tensor(quantizationTable(QF=50, Luminance=False))
        # self.initial_values = torch.stack([QT_Y, QT_C, QT_C], dim=0)

        self.block_idct = block_idct_callable(self.lum_qtable)
        self.block_dct = block_dct_callable(self.lum_qtable)
        
        # self.rgb_to_ycbcr = rgb_to_ycbcr_batch()
        # self.ycbcr_to_rgb = ycbcr_to_rgb_batch()

        self.alpha_lum = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
        self.alpha_chrom = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32)
        nn.init.constant_(self.alpha_lum, opt.JPEG_alpha)
        nn.init.constant_(self.alpha_chrom, opt.JPEG_alpha)

        self.lum_qtable = nn.Parameter(self.lum_qtable)
        self.chrom_qtable = nn.Parameter(self.chrom_qtable)
        
        if opt.JPEG_alpha_trainable:
            print("==> Alpha is trainable per quantization step")
            self.alpha_lum = nn.Parameter(self.alpha_lum)
            self.alpha_chrom = nn.Parameter(self.alpha_chrom)
        else:
            print("==> Alpha is not trainable")
            if torch.cuda.is_available():
                self.alpha_lum = self.alpha_lum.cuda()
                self.alpha_chrom = self.alpha_chrom.cuda()

        self.Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.register_forward_pre_hook(self.reinitialize_q_table_alpha)


    def forward(self, input_RGB):
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        input = input_RGB - 128
        input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size)).unsqueeze(-1)

        input_lum   = input_DCT_block_batch[:, 0:1, ...]
        input_chrom = input_DCT_block_batch[:, 1:3, ...]

        # print("input_lum.device", input_lum.device)
        # print("self.lum_qtable", self.lum_qtable)

        idx_lum   = torch.round(input_lum / self.lum_qtable)
        idx_chrom = torch.round(input_chrom / self.chrom_qtable.expand(1, 2, 1, self.block_size, self.block_size, 1))
        
        
        idx_lum   =  torch.clamp(idx_lum   - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q)
        idx_chrom =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q)
    
        
        idx_lum = idx_lum.expand(-1, -1, -1, -1,-1, self.num_non_zero_q) + self.q_idx
        idx_chrom = idx_chrom.expand(-1, 2, -1, -1, -1, self.num_non_zero_q) + self.q_idx
        
        iq_lum = idx_lum.detach() * self.lum_qtable
        iq_chrom = idx_chrom.detach() * self.chrom_qtable

        distortion_MSE_mask_lum = F.mse_loss(iq_lum, input_lum.expand(-1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')
        distortion_MSE_mask_chrom = F.mse_loss(iq_chrom, input_chrom.expand(-1, -1, -1 ,-1 ,-1 , self.num_non_zero_q), reduction='none')

        estimated_reconstructed_space_lum = torch.sum(F.softmax(-self.alpha_lum * distortion_MSE_mask_lum, dim=-1) * iq_lum , -1)
        estimated_reconstructed_space_chrom  = torch.sum(F.softmax(-self.alpha_chrom.expand(-1, 2, -1, -1, -1, 1)  * distortion_MSE_mask_chrom , dim=-1) * iq_chrom, -1)
        
        estimated_reconstructed_space = torch.cat((estimated_reconstructed_space_lum, estimated_reconstructed_space_chrom), 1)
        
        norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(estimated_reconstructed_space), (self.img_shape[0], self.img_shape[1])))
        norm_img += 128
        
        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        norm_img = self.normalize(self.Scale2One(norm_img))    

        if self.analysis:
            return norm_img, input_DCT_block_batch, estimated_reconstructed_space
        else:
            return norm_img


    def reinitialize_q_table_alpha(self, model, input):
        with torch.no_grad():
            if self.min_Q_Step == 0:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=smallest_float32, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=smallest_float32, max=self.max_Q_Step))
            else:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=self.min_Q_Step, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=self.min_Q_Step, max=self.max_Q_Step))

            if self.JPEG_alpha_trainable:
                self.alpha_lum.copy_(torch.clamp(self.alpha_lum, min=smallest_float32))
                self.alpha_chrom.copy_(torch.clamp(self.alpha_chrom, min=smallest_float32))

def zigzag(matrix):
    zigzag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])
    
    

    # Get the shape of the matrix
    matrix_size = np.shape(matrix)

    # Calculate the size of the vector
    vector_size = np.shape(zigzag)

    # Check if the matrix size matches the vector size
    if matrix_size != vector_size:
        raise ValueError("The matrix size does not match the vector size.")

    # Create an empty vector to store the values
    vector = np.zeros(matrix_size[0] * matrix_size[1])

    # Iterate over each element in the matrix and place it in the corresponding position in the vector
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            vector[index] = matrix[i, j]

    return vector


def inverse_zigzag(vector):
    zigzag = torch.tensor([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the vector
    vector_shape = vector.size()

    # Calculate the size of the 2D matrix
    matrix_size = zigzag.size()

    # Check if the vector size matches the matrix size
    if vector_shape != matrix_size:
        raise ValueError("The vector size does not match the matrix size.")

    # Create an empty matrix to store the values
    matrix = torch.zeros(matrix_size)

    # Iterate over each element in the vector and place it in the corresponding position in the matrix
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            matrix[i, j] = vector[index]

    return matrix


def get_average_model_magnitude(model):
    total_magnitude = 0.0
    num_parameters = 0
    for param in model.parameters():
        param_magnitude = torch.mean(torch.abs(param)).item()
        total_magnitude += param_magnitude
        num_parameters += 1
    if num_parameters == 0:
        return 0.0
    return total_magnitude / num_parameters


def get_max_model_magnitude(model):
    max_magnitude = 0.0
    for param in model.parameters():
        param_max = torch.max(torch.abs(param)).item()
        if param_max > max_magnitude:
            max_magnitude = param_max
    return max_magnitude


def quantizationTable(QF=50, Luminance=True):
    #  Luminance quantization table
    #  Standard
    # * 16 11 10 16 24  40  51  61
    # * 12 12 14 19 26  58  60  55
    # * 14 13 16 24 40  57  69  56
    # * 14 17 22 29 51  87  80  62
    # * 18 22 37 56 68  109 103 77
    # * 24 35 55 64 81  104 113 92
    # * 49 64 78 87 103 121 120 101
    # * 72 92 95 98 112 100 103 99

    quantizationTableData = np.ones((8, 8), dtype=np.float32)

    if QF == 100:
        # print(quantizationTableData)
        return quantizationTableData

    if Luminance == True:  # Y channel
        quantizationTableData[0][0] = 16
        quantizationTableData[0][1] = 11
        quantizationTableData[0][2] = 10
        quantizationTableData[0][3] = 16
        quantizationTableData[0][4] = 24
        quantizationTableData[0][5] = 40
        quantizationTableData[0][6] = 51
        quantizationTableData[0][7] = 61
        quantizationTableData[1][0] = 12
        quantizationTableData[1][1] = 12
        quantizationTableData[1][2] = 14
        quantizationTableData[1][3] = 19
        quantizationTableData[1][4] = 26
        quantizationTableData[1][5] = 58
        quantizationTableData[1][6] = 60
        quantizationTableData[1][7] = 55
        quantizationTableData[2][0] = 14
        quantizationTableData[2][1] = 13
        quantizationTableData[2][2] = 16
        quantizationTableData[2][3] = 24
        quantizationTableData[2][4] = 40
        quantizationTableData[2][5] = 57
        quantizationTableData[2][6] = 69
        quantizationTableData[2][7] = 56
        quantizationTableData[3][0] = 14
        quantizationTableData[3][1] = 17
        quantizationTableData[3][2] = 22
        quantizationTableData[3][3] = 29
        quantizationTableData[3][4] = 51
        quantizationTableData[3][5] = 87
        quantizationTableData[3][6] = 80
        quantizationTableData[3][7] = 62
        quantizationTableData[4][0] = 18
        quantizationTableData[4][1] = 22
        quantizationTableData[4][2] = 37
        quantizationTableData[4][3] = 56
        quantizationTableData[4][4] = 68
        quantizationTableData[4][5] = 109
        quantizationTableData[4][6] = 103
        quantizationTableData[4][7] = 77
        quantizationTableData[5][0] = 24
        quantizationTableData[5][1] = 35
        quantizationTableData[5][2] = 55
        quantizationTableData[5][3] = 64
        quantizationTableData[5][4] = 81
        quantizationTableData[5][5] = 104
        quantizationTableData[5][6] = 113
        quantizationTableData[5][7] = 92
        quantizationTableData[6][0] = 49
        quantizationTableData[6][1] = 64
        quantizationTableData[6][2] = 78
        quantizationTableData[6][3] = 87
        quantizationTableData[6][4] = 103
        quantizationTableData[6][5] = 121
        quantizationTableData[6][6] = 120
        quantizationTableData[6][7] = 101
        quantizationTableData[7][0] = 72
        quantizationTableData[7][1] = 92
        quantizationTableData[7][2] = 95
        quantizationTableData[7][3] = 98
        quantizationTableData[7][4] = 112
        quantizationTableData[7][5] = 100
        quantizationTableData[7][6] = 103
        quantizationTableData[7][7] = 99
    else:
        # Standard Cb Cr channel
        # 17 18  24  47  99  99  99  99
        # 18 21  26  66  99  99  99  99
        # 24 26  56  99  99  99  99  99
        # 47 66  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99

        quantizationTableData[0][0] = 17
        quantizationTableData[0][1] = 18
        quantizationTableData[0][2] = 24
        quantizationTableData[0][3] = 47
        quantizationTableData[0][4] = 99
        quantizationTableData[0][5] = 99
        quantizationTableData[0][6] = 99
        quantizationTableData[0][7] = 99
        quantizationTableData[1][0] = 18
        quantizationTableData[1][1] = 21
        quantizationTableData[1][2] = 26
        quantizationTableData[1][3] = 66
        quantizationTableData[1][4] = 99
        quantizationTableData[1][5] = 99
        quantizationTableData[1][6] = 99
        quantizationTableData[1][7] = 99
        quantizationTableData[2][0] = 24
        quantizationTableData[2][1] = 26
        quantizationTableData[2][2] = 56
        quantizationTableData[2][3] = 99
        quantizationTableData[2][4] = 99
        quantizationTableData[2][5] = 99
        quantizationTableData[2][6] = 99
        quantizationTableData[2][7] = 99
        quantizationTableData[3][0] = 47
        quantizationTableData[3][1] = 66
        quantizationTableData[3][2] = 99
        quantizationTableData[3][3] = 99
        quantizationTableData[3][4] = 99
        quantizationTableData[3][5] = 99
        quantizationTableData[3][6] = 99
        quantizationTableData[3][7] = 99
        quantizationTableData[4][0] = 99
        quantizationTableData[4][1] = 99
        quantizationTableData[4][2] = 99
        quantizationTableData[4][3] = 99
        quantizationTableData[4][4] = 99
        quantizationTableData[4][5] = 99
        quantizationTableData[4][6] = 99
        quantizationTableData[4][7] = 99
        quantizationTableData[5][0] = 99
        quantizationTableData[5][1] = 99
        quantizationTableData[5][2] = 99
        quantizationTableData[5][3] = 99
        quantizationTableData[5][4] = 99
        quantizationTableData[5][5] = 99
        quantizationTableData[5][6] = 99
        quantizationTableData[5][7] = 99
        quantizationTableData[6][0] = 99
        quantizationTableData[6][1] = 99
        quantizationTableData[6][2] = 99
        quantizationTableData[6][3] = 99
        quantizationTableData[6][4] = 99
        quantizationTableData[6][5] = 99
        quantizationTableData[6][6] = 99
        quantizationTableData[6][7] = 99
        quantizationTableData[7][0] = 99
        quantizationTableData[7][1] = 99
        quantizationTableData[7][2] = 99
        quantizationTableData[7][3] = 99
        quantizationTableData[7][4] = 99
        quantizationTableData[7][5] = 99
        quantizationTableData[7][6] = 99
        quantizationTableData[7][7] = 99

    if QF >= 1:
        if QF < 50:
            S = 5000 / QF
        else:
            S = 200 - 2 * QF

        for i in range(8):
            for j in range(8):
                q = (50 + S * quantizationTableData[i][j]) / 100
                q = np.clip(np.floor(q), 1, 255)
                quantizationTableData[i][j] = q
    return quantizationTableData



def get_zigzag():
    zigzag = torch.tensor(( [[0,   1,   5,  6,   14,  15,  27,  28],
                             [2,   4,   7,  13,  16,  26,  29,  42],
                             [3,   8,  12,  17,  25,  30,  41,  43],
                             [9,   11, 18,  24,  31,  40,  44,  53],
                             [10,  19, 23,  32,  39,  45,  52,  54],
                             [20,  22, 33,  38,  46,  51,  55,  60],
                             [21,  34, 37,  47,  50,  56,  59,  61],
                             [35,  36, 48,  49,  57,  58,  62,  63]]))
    return zigzag

def _normalize(N: int) -> torch.Tensor:
    n = torch.ones((N, 1)).to(device)
    n[0, 0] = 1 / math.sqrt(2)
    
    return n @ n.t()

def _harmonics(N: int) -> torch.Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)
    
def block_dct(blocks: torch.Tensor) -> torch.Tensor:
    N = blocks.shape[3]

    n = _normalize(N).float()
    h = _harmonics(N).float()

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()
    
    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff

def block_idct(coeff: torch.Tensor) -> torch.Tensor:
    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


class block_dct_callable(nn.Module):
    """Callable class."""
    
    def __init__(self,blocks):
        super(block_dct_callable, self).__init__()
        self.N = blocks.shape[3]
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()

        if torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, blocks):
        coeff = (1 / math.sqrt(2 * self.N)) * self.n * (self.h.t() @ blocks @ self.h)
        return coeff


class block_idct_callable(nn.Module):   
    def __init__(self, blocks: torch.Tensor):
        super(block_idct_callable, self).__init__()
        self.N = blocks.shape[3]
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()
    
        if torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, coeff):
        im = (1 / math.sqrt(2 * self.N)) * (self.h @ (self.n * coeff) @ self.h.t())
        return im

def rgb_to_ycbcr(image: torch.Tensor,
                 W_r = 0.299,
                 W_g = 0.587,
                 W_b = 0.114) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = W_r * r + W_g * g + W_b * b
    cb: torch.Tensor = (b - y) /(2*(1-W_b)) + delta
    cr: torch.Tensor = (r - y) /(2*(1-W_r)) + delta
    return torch.stack((y, cb, cr), -3)



class rgb_to_ycbcr_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r = 0.299
        # W_g = 0.587
        # W_b = 0.114
        # self.T = torch.tensor([ [W_r, W_g, W_b],
        #                         [-W_r/2, -W_g/2, (1-W_b)/2],
        #                         [(1-W_r)/2, -W_g/2, -W_b/2]], dtype=torch.float32)

        self.T = torch.tensor([ [0.299, 0.587, 0.114],
                                [-0.168736, -0.331264, 0.5],
                                [0.5, -0.418688, -0.081312]], dtype=torch.float32)

        self.B = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        if torch.cuda.is_available():
            self.T = self.T.cuda()
            self.B = self.B.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1,  C)
        
        # Perform the matrix multiplication and add the bias
        ycbcr_reshaped = torch.matmul(images_reshaped, self.T.T) + self.B
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        ycbcr_images = ycbcr_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        
        return ycbcr_images

def ycbcr_to_rgb(image: torch.Tensor,
                 W_r=0.299,
                 W_g=0.587,
                 W_b=0.114) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 2*(1-W_r) * cr_shifted
    g: torch.Tensor = y - 2*(1-W_r)*W_r/W_g * cr_shifted - 2*(1-W_b)*W_b/W_g * cb_shifted
    b: torch.Tensor = y + 2*(1-W_b) * cb_shifted
    return torch.stack([r, g, b], -3)

class ycbcr_to_rgb_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r=0.299
        # W_g=0.587
        # W_b=0.114
        # self.T_inv = torch.tensor([ [1.0, 0.0,  2*(1-W_r)], 
        #                             [1.0, - 2*(1-W_b)*W_b/W_g,  - 2*(1-W_r)*W_r/W_g], 
        #                             [1.0, 2*(1-W_b), 0.0]], dtype=torch.float32)
        
        self.T_inv = torch.tensor([[1.0, 0.0, 1.402],
                          [1.0, -0.344136, -0.714136],
                          [1.0, 1.772, 0.0]], dtype=torch.float32)
        self.B_inv = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        
        
        if torch.cuda.is_available():
            self.T_inv = self.T_inv.cuda()
            self.B_inv = self.B_inv.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1, C)
        
        # Subtract the bias from Cb and Cr channels
        images_reshaped -= self.B_inv
        
        # Perform the matrix multiplication
        rgb_reshaped = torch.matmul(images_reshaped, self.T_inv.T)
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        rgb_images = rgb_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        return rgb_images

def convert_NCWL_to_NWLC(img):
    return torch.transpose(torch.transpose(img,1,2),2,3)

def pad_shape(Num, size=8):
    res = Num%size
    pad = 1
    if(res == 0):
        pad = 0
    n = (Num//size+pad)*size
    return n

def blockify(im: torch.Tensor, size: int) -> torch.Tensor:
    shape = im.shape[-2:]
    padded_shape = [pad_shape(shape[0]),pad_shape(shape[1])]
    paded_im = F.pad(im, (0,padded_shape[1]-shape[1], 0,padded_shape[0]-shape[0]), 'constant',0)
    bs = paded_im.shape[0]
    ch = paded_im.shape[1]
    h = paded_im.shape[2]
    w = paded_im.shape[3]
    paded_im = paded_im.reshape(bs * ch, 1, h, w)
    paded_im = torch.nn.functional.unfold(paded_im, kernel_size=(size, size), stride=(size, size))
    paded_im = paded_im.transpose(1, 2)
    paded_im = paded_im.reshape(bs, ch, -1, size, size)
    return paded_im

def deblockify(blocks: torch.Tensor, size) -> torch.Tensor:
    padded_shape = pad_shape(size[0]),pad_shape(size[1])
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]
    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=padded_shape, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, padded_shape[0], padded_shape[1])
    blocks = blocks[:,:,:size[0],:size[1]]
    return blocks

def load_3x3_weight(model_name = "Alexnet"):
    rt_arr = np.zeros((3,3))
    seq_weight = np.genfromtxt("color_conv_W/"+model_name+"_W_OPT.txt")
    for i in range(3):
        for j in range(3):
            rt_arr[i, j] = seq_weight[i*3+j]
    return torch.Tensor(rt_arr)


# def plot_confidence_interval(x, top, bottom, mean, horizontal_line_width=0.25, color='#2187bb',label=None,alpha=1):
#     left = x - horizontal_line_width / 2
#     right = x + horizontal_line_width / 2
#     plt.plot([x, x], [top, bottom], color=color,alpha=0.7*alpha)
#     plt.plot([left, right], [top, top], color=color,alpha=0.7*alpha)
#     plt.plot([left, right], [bottom, bottom], color=color,alpha=0.7*alpha)
#     plt.plot(x, mean, 'o', color=color, label=label,alpha=alpha)
#     return mean