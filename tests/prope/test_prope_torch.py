# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import tqdm

# Enable highest precision
torch.set_default_dtype(torch.float64)

from prope.torch import PropeDotProductAttention, prope_dot_product_attention


def test_prope_torch():
    torch.manual_seed(42)
    cameras = 3
    patches_x = 8
    patches_y = 8
    image_width = 128
    image_height = 128

    batch = 2
    seqlen = cameras * patches_x * patches_y
    num_heads = 4
    head_dim = 16

    q = torch.randn(batch, num_heads, seqlen, head_dim)
    k = torch.randn(batch, num_heads, seqlen, head_dim)
    v = torch.randn(batch, num_heads, seqlen, head_dim)

    viewmats = torch.eye(4).repeat(batch, cameras, 1, 1)
    Ks = torch.rand(batch, cameras, 3, 3)

    for _ in tqdm.tqdm(range(1), desc="prope as function"):
        out_torch_0 = prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
        )

    prope = PropeDotProductAttention(
        head_dim=head_dim,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    for _ in tqdm.tqdm(range(1), desc="prope as module"):
        out_torch_1 = prope(q, k, v, viewmats, Ks)

    torch.testing.assert_close(out_torch_0, out_torch_1)


if __name__ == "__main__":
    test_prope_torch()
