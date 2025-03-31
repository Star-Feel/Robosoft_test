# SoftRoboticaSimulator

## install
```shell
conda create -n ssim python=3.11
conda activate ssim

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyelastica
pip install -r requirements.txt

```
适配mesh surface
注释掉elastica/modules/contact.py中第67-75行contact合法性检验

适配 mesh rigid
注释掉 elastica/memory_block/memory_block_rigid_body.py 第48-49行

RodMeshSurfaceContact jit加速数据类型不对齐bug
elastica/mesh/mesh_initializer.py 第92行，修改为
```python
self.face_normals = self.face_normal_calculation(self.mesh.face_normals.astype(np.float64))
```

## grab ball

<div style="text-align: center;">
  <img src="videos/2d.gif" alt="Demo GIF" width="300"/>
</div>

<div style="text-align: center;">
  <img src="videos/3d.gif" alt="Demo GIF" width="300"/>
</div>
