import unittest

import numpy as np
import torch

from mpcrl.envs import _contiguous_rgb_frame


class TestRenderFrameContiguity(unittest.TestCase):
    def test_negative_stride_render_frame_is_made_contiguous(self):
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)[::-1]
        self.assertLess(frame.strides[0], 0)

        fixed = _contiguous_rgb_frame(frame)

        self.assertTrue(fixed.flags.c_contiguous)
        self.assertEqual(fixed.shape, (4, 5, 3))
        self.assertEqual(fixed.dtype, np.uint8)
        tensor = torch.as_tensor(fixed)
        self.assertEqual(tuple(tensor.shape), (4, 5, 3))

    def test_contiguous_frame_is_preserved(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        fixed = _contiguous_rgb_frame(frame)
        self.assertIs(fixed, frame)


if __name__ == '__main__':
    unittest.main()
