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

17.[A good article for PPO&IPPO&CTDE in MARL](https://zhuanlan.zhihu.com/p/544046358)<br>

18.Different Lamda setting for reinforcement learning<br>
![image](https://github.com/user-attachments/assets/cb378062-bd83-4c2b-bd9a-ace96f97817c)<br>

19.Good paper about srlfplay in LLM<br>
[Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://www.arxiv.org/pdf/2505.03335)<br>
![image](https://github.com/user-attachments/assets/0723485a-f3e5-4aea-be6b-104a3ae15cc0)<br>

20.about to read<br>
[Click Here](https://arxiv.org/pdf/2503.01067?)<br>
![image](https://github.com/user-attachments/assets/9f106780-b790-418a-9116-658027fd3705)<br>

21.[Visual Imitation Enables Contextual Humanoid Control](https://arxiv.org/pdf/2505.03729)<br>
![image](https://github.com/user-attachments/assets/e5e551a7-174c-41b3-a26f-b3399758c5a2)<br>
[website](https://www.videomimic.net/)<br>

22.[Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/pdf/2505.00949)<br>
Certainly! Here's a detailed explanation of the **five-stage training pipeline** from the Llama-Nemotron paper, rewritten clearly in English:

---

## 🔄 Five-Stage Training Pipeline (from Llama-Nemotron)

### **Stage 1: Neural Architecture Search (NAS) + FFN Fusion**

**Goal:** Build a highly efficient model architecture optimized for inference speed and resource usage.

* **Puzzle Framework:** Performs block-wise local distillation by replacing individual Transformer blocks (e.g., removing attention, shrinking FFN width) and evaluates them in terms of latency, memory, and accuracy.
* **Mixed-Integer Programming (MIP):** Used to select the optimal combination of blocks across layers to meet deployment constraints (e.g., latency budget on H100 GPUs).
* **FFN Fusion:** After pruning attention layers, multiple sequential FFN blocks are fused into a wider parallel FFN to reduce model depth and increase GPU utilization.

> 📍 **Reference:** Section 2, Figure 3

---

### **Stage 2: Knowledge Distillation + Continued Pretraining (CPT)**

**Goal:** Restore and even surpass the performance of original Llama-3 models despite architecture compression.

* **Knowledge Distillation:** The models are trained to imitate outputs from strong teacher models like Llama-3.3-70B or DeepSeek-R1 using the **Distillation Mix** dataset.
* **Continued Pretraining (CPT):** LN-Ultra continues unsupervised pretraining on the **Nemotron-H** dataset (88B tokens), which improves generalization on math, code, and science tasks.

> 📍 **Reference:** Section 2.2, Table 1

---

### **Stage 3: Supervised Fine-Tuning (SFT)**

**Goal:** Teach the model to perform both multi-step reasoning and regular chat, with the ability to switch styles dynamically.

* **“Detailed Thinking” Mode:** Each training sample is tagged with a system prompt indicating whether to use chain-of-thought (CoT) reasoning (`detailed thinking on`) or to give concise answers (`detailed thinking off`).
* **Paired Data Construction:** For every reasoning-style response, a non-reasoning version is generated using an LLM, enabling the model to learn to toggle between reasoning and short-form answers.

> 📍 **Reference:** Section 3 and Section 4

---

### **Stage 4: Reinforcement Learning (RL)**

**Goal:** Further enhance reasoning capabilities and outperform teacher models on complex scientific benchmarks.

* **GRPO Algorithm:** Uses **Group Relative Policy Optimization**, a custom RL method designed to improve reasoning quality while maintaining response formatting.
* **Reward Functions:** Combine correctness, proper reasoning format (e.g., "Let's think step by step"), and difficulty-based curriculum learning to improve both accuracy and stability.
* **FP8 Sampling:** Inference during RL training is run in FP8 precision to reduce memory footprint and increase throughput on limited hardware (e.g., 8×H100).

> 📍 **Reference:** Section 5.1 and 5.2

---

### **Stage 5: Alignment**

**Goal:** Make the model safer, more helpful, and more user-aligned, suitable for real-world deployment.

* **Human Preference Fine-Tuning:** Aligns the model with human values using safety-tuned datasets and reward models.
* **Instruction Following:** Trains the model to better follow user prompts and behave predictably, especially in enterprise use cases.

> 📍 **Reference:** Section 5 (final phase)

---


23.RL in LLM<br>
![image](https://github.com/user-attachments/assets/ea358ddc-9d37-4d96-9450-4dce84e7b501)<br>

24.IGAL Conference<br>
(1)Use PnP to make SFM Reconstruction<br>
Iterative SfM add the new photos  iteratively (cannot be global optimal)
Global SfM can be global optimal<br>
Learning based SfM<br>
(2)Robotics open problems<br>
1.Evaluation/Benchmark <br>
2.Data Collection (Data Engineering) <br>
3.Algorithm (robot encoder) <br>
4.How to make a demo (livestreaming) <br>
(3) Go-1<br>
web-scale date > simulation data > real world data <br>
data ratio, how to cellect the data and engineer the data <br>
