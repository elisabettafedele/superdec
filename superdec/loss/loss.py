import torch
import torch.nn as nn

from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.transforms import transform_to_primitive_frame



def sampling_from_parametric_space_to_equivalent_points(
    shape_params,
    epsilons,
    sq_sampler
):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points.

    Arguments:
    ----------
        shape_params: Tensor with size BxMx3, containing the scale along each
                      axis for the M primitives
        epsilons: Tensor with size BxMx2, containing the shape along the
                  latitude and the longitude for the M primitives

    Returns:
    ---------
        P: Tensor of size BxMxSx3 that contains S sampled points from the
           surface of each primitive
        N: Tensor of size BxMxSx3 that contains the normals of the S sampled
           points from the surface of each primitive
    """
    # Allocate memory to store the sampling steps
    def fexp(x, p):
        return torch.sign(x)*(torch.abs(x)**p)
    B = shape_params.shape[0]  # batch size
    M = shape_params.shape[1]  # number of primitives
    S = sq_sampler.n_samples

    etas, omegas = sq_sampler.sample_on_batch(
        shape_params.detach().cpu().numpy(),
        epsilons.detach().cpu().numpy()
    )
    # Make sure we don't get nan for gradients
    etas[etas == 0] += 1e-6
    omegas[omegas == 0] += 1e-6

    # Move to tensors
    etas = shape_params.new_tensor(etas)
    omegas = shape_params.new_tensor(omegas)

    # Make sure that all tensors have the right shape
    a1 = shape_params[:, :, 0].unsqueeze(-1)  # size BxMx1
    a2 = shape_params[:, :, 1].unsqueeze(-1)  # size BxMx1
    a3 = shape_params[:, :, 2].unsqueeze(-1)  # size BxMx1
    e1 = epsilons[:, :, 0].unsqueeze(-1)  # size BxMx1
    e2 = epsilons[:, :, 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    # Make sure we don't get INFs
    # x[torch.abs(x) <= 1e-9] = 1e-9
    # y[torch.abs(y) <= 1e-9] = 1e-9
    # z[torch.abs(z) <= 1e-9] = 1e-9
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

    # Compute the normals of the SQs
    nx = (torch.cos(etas)**2) * (torch.cos(omegas)**2) / x
    ny = (torch.cos(etas)**2) * (torch.sin(omegas)**2) / y
    nz = (torch.sin(etas)**2) / z

    return torch.stack([x, y, z], -1), torch.stack([nx, ny, nz], -1)



class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self._init_buffers()

        self.sampler = EqualDistanceSamplerSQ(n_samples=cfg.n_samples, D_eta=0.05, D_omega=0.05)

        self.w_sps = cfg.w_sps
        self.w_ext = cfg.w_ext
        self.w_cub = cfg.w_cub
        self.w_cd = cfg.w_cd

        self.cos_sim_cubes = nn.CosineSimilarity(dim=4, eps=1e-4)

    def _init_buffers(self):
        self.register_buffer('mask_project', torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('mask_plane', torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_normal', torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_planes', torch.FloatTensor([[-1,-1,-1],[1,1,1]]).unsqueeze(0).unsqueeze(0))

    def compute_cuboid_loss(self, pc_inver, normals_inver, out_dict):
        B, P = out_dict['scale'].shape[:2]
        N = pc_inver.shape[2]

        planes_scaled = self.cube_planes.repeat(B, P, 1, 1) * out_dict['scale'].unsqueeze(2).repeat(1,1,2,1)
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1,N,1,3,1).reshape(B,N,P*6,3)
        mask_project = self.mask_project.repeat(B,N,P,1)
        mask_plane = self.mask_plane.repeat(B,N,P,1)
        cube_normal = self.cube_normal.unsqueeze(2).repeat(B,N,P,1,1)
        scale_reshaped = out_dict['scale'].unsqueeze(1).repeat(1,N,1,6).reshape(B,N,P*6,3)

        normals_inver_reshaped = normals_inver.permute(0,2,1,3).unsqueeze(3).repeat(1,1,1,6,1)
        _, idx_normals_sim_max = torch.max(self.cos_sim_cubes(normals_inver_reshaped, cube_normal),dim=-1,keepdim=True)

        pc_project = pc_inver.permute(0,2,1,3).repeat(1,1,1,6).reshape(B,N,P*6,3) * mask_project + planes_scaled * mask_plane
        pc_project = torch.max(torch.min(pc_project, scale_reshaped), -scale_reshaped).view(B, N, P, 6, 3)
        pc_project = torch.gather(pc_project, dim=3, index = idx_normals_sim_max.unsqueeze(-1).repeat(1,1,1,1,3)).squeeze(3).permute(0,2,1,3)

        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0,2,1)
        diff = torch.mean(torch.mean(torch.sum(diff * out_dict['assign_matrix'], -1), 1))

        return diff

    def compute_cd_loss(self, pc_inver, out_dict):
        """
        Args:
            weights:       [B, P, N]
            scale:         [B, P, 3]
            shape:         [B, P, 2]
            exist:         [B, P, 1]
            transformed_points: [B, P, N, 3]
            normals_gt:    [B, P, N, 3]
        Returns:
            pcl_to_prim_loss, prim_to_pcl_loss, normal_loss, pcl_to_prim_distances
        """
        weights = out_dict['assign_matrix']  # [B, P, N]
        scale   = out_dict['scale']          # [B, P, 3]
        shape   = out_dict['shape']          # [B, P, 2]
        exist   = out_dict['exist']          # [B, P, 1]

        # Sample points and normals on superquadrics
        X_SQ, normals = sampling_from_parametric_space_to_equivalent_points(scale, shape, self.sampler)
        normals = normals.detach()        # [B, P, S, 3]
        normals = torch.nn.functional.normalize(normals, dim=-1)  # ensure unit normals

        # Compute squared distances: [B, P, S, N]
        diff = X_SQ.unsqueeze(3) - pc_inver.unsqueeze(2)  # [B, P, S, N, 3]
        D = (diff ** 2).sum(-1)                                     # [B, P, S, N]

        # Point-to-Primitive Chamfer
        pcl_to_prim_loss = D.min(dim=2)[0]
        pcl_to_prim_loss = (pcl_to_prim_loss.transpose(-1,-2) * weights).sum(-1).mean()            # [B, P, N]

        # Primitive-to-Point Chamfer
        distances_bis = D.min(dim=3)[0]                     # [B, P, S]
        prim_to_pcl = distances_bis.mean(dim=-1)            # [B, P]
        prim_to_pcl_loss = (prim_to_pcl * exist.squeeze(-1)).sum(dim=-1)
        prim_to_pcl_loss = (prim_to_pcl_loss / (exist.squeeze(-1).sum(dim=-1) + 1e-6)).mean()

        return pcl_to_prim_loss, prim_to_pcl_loss

    def compute_existence_loss(self, assign_matrix, exist):
        thred = 24
        loss = nn.BCELoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        entropy = loss(exist.squeeze(-1), gt)
        return entropy

    def get_sparsity_loss(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)
        return norm_05

    def forward(self, pc, normals, out_dict):
        pc_inver = transform_to_primitive_frame(pc, out_dict['trans'], out_dict['rotate'])
        normals_inver = transform_to_primitive_frame(normals, out_dict['trans'], out_dict['rotate'])

        loss = 0
        loss_dict = {}

        if self.w_cub > 0:
            cub_loss = self.compute_cuboid_loss(pc_inver, normals_inver, out_dict)
            loss += self.w_cub * cub_loss
            loss_dict['cub_loss'] = cub_loss.item()

        if self.w_cd > 0:
            pcl_to_prim_loss, prim_to_pcl_loss = self.compute_cd_loss(pc_inver, out_dict)
            cd_loss = pcl_to_prim_loss + prim_to_pcl_loss
            loss += self.w_cd * cd_loss
            loss_dict['cd_loss'] = cd_loss.item()

        if self.w_ext > 0:
            exist_loss = self.compute_existence_loss(out_dict['assign_matrix'], out_dict['exist'])
            loss += self.w_ext * exist_loss
            loss_dict['exist_loss'] = exist_loss.item()

        if self.w_sps > 0:
            sparsity_loss = self.get_sparsity_loss(out_dict['assign_matrix'])
            loss += self.w_sps * sparsity_loss
            loss_dict['sparsity_loss'] = sparsity_loss.item()

        loss_dict['expected_prim_num'] = out_dict['exist'].squeeze(-1).sum(-1).mean().data.detach().item()
        loss_dict['all'] = loss.item()
        return loss, loss_dict
