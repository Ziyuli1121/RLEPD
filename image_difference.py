from pathlib import Path
import numpy as np
from PIL import Image

base = Path("samples/origin/000000/000009.png")      # 冷启动图像
rl   = Path("samples/rl/000000/000009.png")         # RL 导出的图像

img_base = np.array(Image.open(base).convert("RGB"), dtype=np.int16)
img_rl   = np.array(Image.open(rl).convert("RGB"), dtype=np.int16)

diff = img_rl - img_base
print("max abs diff:", np.abs(diff).max())
print("mean abs diff:", np.abs(diff).mean())
print("num pixels different:", np.count_nonzero(np.abs(diff)))

