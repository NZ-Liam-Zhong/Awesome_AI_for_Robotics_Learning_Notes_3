# Awesome_AI_for_Robotics_Learning_Notes_3
**SKip Back to** [Part 1](https://github.com/NZ-Liam-Zhong/Awesome_AI_for_Robotics_Learning_Notes)
![image](https://github.com/user-attachments/assets/069767fb-244a-46c2-8387-b6ad3b0ad572)<br>

## Notes🖊
1.[Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)<br>
only use positive pairs to train a representation<br>
![image](https://github.com/user-attachments/assets/1bcc7ca4-a3c6-4512-bff6-358ebbb95a96)<br>
[zhihu](https://proceedings.neurips.cc/paper_files/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)<br>

2.[SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)<br>
![image](https://github.com/user-attachments/assets/dd2c96d9-9bcc-4771-91bb-e61e569822be)<br>

3.[Slides for Meta-Learning](https://cs182sp21.github.io/static/slides/lec-21.pdf）<br>
UC Berkeley<br>

4.[ViVa: Video-Trained Value Functions for Guiding Online RL from Diverse Data](https://arxiv.org/pdf/2503.18210)<br>
Author: Sergey Levine<br>
![image](https://github.com/user-attachments/assets/9fa91e18-f6b5-4c3a-9a82-babcb1bbdaaf)<br>

5.[Curating Demonstrations using Online Experience](https://arxiv.org/pdf/2503.03707?)<br>
Author: Chelsea Finn<br>
![image](https://github.com/user-attachments/assets/560df2d3-50c2-4bfc-ba34-93c090b72e22)<br>

6.[AUTOEVAL: AUTONOMOUS EVALUATION OF GENERALIST ROBOT MANIPULATION POLICIES IN THE REAL WORLD](https://auto-eval.github.io/)<br>
Good way for evaluations<br>
![image](https://github.com/user-attachments/assets/a070b33d-749a-4f35-8138-09a13b5f547f)<br>
[twitter](https://x.com/svlevine/status/1906912333433298980)<br>

7.[Introduction to LLM](https://web.stanford.edu/~jurafsky/slp3/slides/LLM24aug.pdf)<br>
Stanford Slides<br>

8.CQL<br>
![image](https://github.com/user-attachments/assets/aec96229-bd39-4a76-86af-dce898555dfe)<br>

Offline-to-Online RL<br>

9.[Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble](https://proceedings.mlr.press/v164/lee22d/lee22d.pdf)<br>
![image](https://github.com/user-attachments/assets/c9db282b-398c-4c4b-b933-0da6f7d564a6)<br>

10.[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/c44a04289beaf0a7d968a94066a1d696-Paper-Conference.pdf)<br>
![image](https://github.com/user-attachments/assets/77ae3581-a061-4bdf-8b1e-10de9f8568ea)<br>
![image](https://github.com/user-attachments/assets/f369678f-af93-4805-905a-30aa6e1ba3b8)<br>

11.Don't want to open OpenGL?<br>

```
# generate_locomaze.py 开头，务必放在所有其他 import 之前

# 1) stub 掉 MjRenderContext，避免调用 mjr_makeContext 加载 OpenGL
import mujoco
mujoco.MjrContext = lambda model, fontscale: None

# 2) 准备一个最小化的 “摄像机” 接口
class DummyCam:
    def __init__(self):
        # 与 mujoco_rendering.OffScreenViewer._init_camera 中用到的属性保持一致 :contentReference[oaicite:0]{index=0}
        self.type = None
        self.fixedcamid = -1
        self.lookat = [0.0, 0.0, 0.0]   # 必须支持索引赋值，否则 MazeEnv.__init__ 会报错
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0

# 3) 定义一个 DummyViewer，用来替换 OffScreenViewer 和 RenderContextOffscreen
class DummyViewer:
    def __init__(self, *args, **kwargs):
        # 创建一个“假”摄像机
        self.cam = DummyCam()
    def render(self, *args, **kwargs):
        # 如果有代码调用 render()，直接 no-op
        return None

# 4) Monkey‑patch Gymnasium 的 MuJoCo 渲染模块
import gymnasium.envs.mujoco.mujoco_rendering as rendering
rendering.OffScreenViewer = DummyViewer
rendering.RenderContextOffscreen = DummyViewer
```

12.**Slides:**[How To Leverage Unlabeled Data in Offline Reinforcement Learning](https://icml.cc/media/icml-2022/Slides/17328_BtUukaj.pdf?utm_source=chatgpt.com)<br>
![image](https://github.com/user-attachments/assets/a6bc28b6-9702-4570-85f4-4cd3e6df81c9)<br>

13.[Survey on Mult-agent reinforcement learning](https://arxiv.org/pdf/1911.10635)<br>


14.[policy decorator](https://t.co/zmuPrSUXoa)<br>
![image](https://github.com/user-attachments/assets/a94e360e-ae6a-4728-867b-53be57b2cae8)<br>

15.[PI 0.5](https://arxiv.org/pdf/2504.16054)<br>
![image](https://github.com/user-attachments/assets/fab1e5d5-9224-42a8-b678-74f9e39877a7)<br>
![image](https://github.com/user-attachments/assets/bf665c82-8051-4121-907a-ab32a2845ebd)<br>

16.[FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/pdf/2501.09747)<br>
About to read<br>



