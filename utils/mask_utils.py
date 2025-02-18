import torch
import pdb
import math

class Scheduler:
    def __init__(self, step=5, mode="arccos", patch_size=16, embedding_dim=256):
        """
        :param
        step  -> int:  total number of prediction steps during inference
        mode  -> str:  the rate of value to unmask
        patch_size -> int: size of the patch (e.g., 16 for 16x16 tokens)
        embedding_dim -> int: dimension of the learnable embedding for masked values
        """
        self.step = step
        self.mode = mode
        self.patch_size = patch_size
        self.embedding = torch.nn.Parameter(torch.randn(embedding_dim))  # Learnable embedding
        self._create_scheduler()

    def _get_vals_to_mask(self,r):
        if self.mode == "root":              
            val_to_mask = 1 - (r ** .5)
        elif self.mode == "linear":          
            val_to_mask = 1 - r
        elif self.mode == "square":          
            val_to_mask = 1 - (r ** 2)
        elif self.mode == "cosine":          
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif self.mode == "arccos":          
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            raise ValueError("Invalid mode")
        return val_to_mask


    def _create_scheduler(self,patch_size=16):
        """ Create a sampling scheduler based on the mode """
        r = torch.linspace(1, 0, self.step)
        val_to_mask=self._get_vals_to_mask(r)

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (patch_size**2)
        sche = sche.round().int()
        sche[sche == 0] = 1  # at least 1 token per step
        sche[-1] += (patch_size**2) - sche.sum()  # ensure sum matches total tokens
        self.val_to_mask = sche


    def get_mask(self, step, code):
        """
        :param
        step  -> int: current step number
        code  -> torch.LongTensor: bsize * 16 * 16, the unmasked code
        :return
        mask  -> torch.Tensor: binary mask of size bsize * 16 * 16
        """
        if step < 0 or step >= self.step:
            raise ValueError("Step out of bounds")

        # Get the mask ratio for the current step
        mask_ratio = self.val_to_mask[step].float() / (self.patch_size * self.patch_size)
        
        # Create a mask based on the ratio
        mask = torch.rand(size=code.size()) < mask_ratio
        mask = mask.int()  # convert boolean mask to int (1 for masked, 0 for unmasked)

        return mask,self.val_to_mask[step]

    def add_mask_for_training(self, code, value=None):
        """
        Replace code token by *value* according to the *mode* scheduler.
        :param
        code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
        mode  -> str:  rate of value to mask
        value -> int:  mask the code by the value (can be replaced with learnable embedding)
        :return
        masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
        mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0),device=code.device)
        val_to_mask = self._get_vals_to_mask(r).to(code.device)
        # Create the mask based on the computed val_to_mask
        mask = (torch.rand(size=code.size(),device=code.device) < val_to_mask.view(code.size(0), 1)).int()

        return mask



if __name__ == '__main__':
    pass 


