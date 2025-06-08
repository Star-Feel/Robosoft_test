# SoftRoboticaSimulator

## install
```shell
conda create -n ssim python=3.11
conda activate ssim

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
cp ./pyelastica.patch ./third_patry/pyelastica
cd ./third_patry/pyelastica && git apply pyelastica.patch
pip install .
pip install -r requirements.txt
pip install -e.
```
适配mesh surface
注释掉elastica/modules/contact.py中第67-75行contact合法性检验

RodMeshSurfaceContact jit加速数据类型不对齐bug
elastica/mesh/mesh_initializer.py 第92行，修改为
```python
self.face_normals = self.face_normal_calculation(self.mesh.face_normals.astype(np.float64))
```
为了渲染，我们需要进行如下安装

```shell
sudo apt-get update
sudo apt-get install -y autoconf automake build-essential cmake libboost-all-dev libpng-dev libjpeg-dev libtiff-dev libopenexr-dev libsdl1.2-dev libxrender-dev libgl-dev libxi6 libxxf86vm-dev libxfixes3 libxkbcommon-x11-0 libsm6 libxext6
conda install ffmpeg -y
```
运行 `tests` 路径下的文件进行测试。

## grab ball

<div style="text-align: center;">
  <img src="videos/2d.gif" alt="Demo GIF" width="300"/>
</div>

<div style="text-align: center;">
  <img src="videos/3d.gif" alt="Demo GIF" width="300"/>
</div>

## VLN Data Generate

Run the following scripts steply

```bash
bash scirpts/navigation/date_gen.sh
```

## VLM Data Generate
To run RL model, we need to
```bash
pip install tensorflow==2.15.0
pip install stable_baselines==2.10.2
```
stablebaselines需要低版本tensorflow，如果你安装了高版本tensorflow，请在每一个报错处修改 import tensorflow.compat.v1 as tf
