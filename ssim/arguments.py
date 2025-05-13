import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, NewType, Optional, Sequence, Type, Union, List

import numpy as np
from transformers import HfArgumentParser
import yaml

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


class SuperArgumentParser(HfArgumentParser):

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType,
                                        Iterable[DataClassType]]] = None,
        prefix: Optional[Sequence[str]] = None,
    ):
        if prefix is None:
            super().__init__(dataclass_types)
        else:
            if dataclass_types is None:
                dataclass_types = []
            elif not isinstance(dataclass_types, Iterable):
                dataclass_types = [dataclass_types]

            if dataclasses.is_dataclass(dataclass_types):
                dataclass_types = [dataclass_types]
            self.dataclass_types = list(dataclass_types)

        self.prefix = prefix

    def parse_dict(
        self,
        args: dict[str, Any],
        allow_extra_keys: bool = False,
    ) -> tuple[DataClass, ...]:
        if self.prefix is None:
            unused_keys = set(args.keys())
            outputs = []
            for dtype in self.dataclass_types:
                keys = {f.name for f in dataclasses.fields(dtype) if f.init}
                types = {f.name: f.type for f in fields(dtype) if f.init}
                inputs = {
                    k: self._convert_type(v, types[k])
                    for k, v in args.items() if k in keys
                }
                unused_keys.difference_update(inputs.keys())
                obj = dtype(**inputs)
                outputs.append(obj)
            if not allow_extra_keys and unused_keys:
                raise ValueError(f"Some keys are not used by the"
                                 f"HfArgumentParser: {sorted(unused_keys)}")
            return tuple(outputs)
        outputs = []
        for idx, dtype in enumerate(self.dataclass_types):

            local_args = args.get(self.prefix[idx], {})

            keys = {f.name for f in fields(dtype) if f.init}
            types = {f.name: f.type for f in fields(dtype) if f.init}
            inputs = {
                k: self._convert_type(v, types[k])
                for k, v in local_args.items() if k in keys
            }
            obj = dtype(**inputs)
            outputs.append(obj)
        return tuple(outputs)

    def _convert_type(self, value, target_type: Type):
        # 处理numpy数组
        if target_type is np.ndarray:
            return np.asarray(value, dtype=float)

        # 处理基本类型转换
        try:
            return target_type(value)
        except TypeError:
            # 处理需要特殊构造的类型（如自定义类）
            if hasattr(target_type, '__annotations__'):
                return target_type(**value)
            raise


@dataclass
class SuperArguments:

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) == self.__annotations__[field_name]:

                continue
            value = getattr(self, field_name)
            if isinstance(value, dict):
                dataclass_type = value.get("type", field_name)
                setattr(
                    self, field_name,
                    SuperArgumentParser(args_dict[dataclass_type]).parse_dict(
                        value, True)[0])
            elif isinstance(value, list):
                configs = []
                for config in value:
                    configs.append(
                        SuperArgumentParser(
                            args_dict[config["type"]]).parse_dict(
                                config, True)[0])
                setattr(self, field_name, configs)

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        return cls(**data)

@dataclass
class RodControllerArgumets:
    trainable: bool = False
    number_of_control_points: int = 6
    obs_state_points: int = 10

    boundary: Optional[tuple] = None


@dataclass
class SimulatorArguments:

    final_time: float = field(default_factory=float)
    time_step: float = field(default_factory=float)
    collect_data: bool = field(default_factory=bool)
    rendering_fps: float = field(default_factory=float)
    update_interval: int = field(default_factory=float)


@dataclass
class RodControllerArgumets:
    trainable: bool = False
    number_of_control_points: int = 6
    obs_state_points: int = 10

    boundary: Optional[tuple] = None


@dataclass
class SphereArguments:
    center: np.ndarray = field(default_factory=list)
    radius: float = field(default_factory=float)
    density: float = field(default_factory=float)
    direction: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0.]))


@dataclass
class RodArguments:

    n_elem: int = field(default_factory=int)
    start: np.ndarray = field(default_factory=list)
    direction: np.ndarray = field(default_factory=list)
    normal: np.ndarray = field(default_factory=list)
    base_length: float = field(default=0.35)
    radius_tip: float = field(default=0.05)
    base_radius: float = field(default=0.0035)
    density: float = field(default=1000.0)
    nu: float = field(default=30)
    youngs_modulus: float = field(default_factory=float)
    poisson_ratio: float = field(default=0.5)


@dataclass
class MeshSurfaceArguments:

    mesh_path: str
    center: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    scale: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))
    rotate: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))


args_dict = {
    "rod": RodArguments,
    "sphere": SphereArguments,
    "mesh_surface": MeshSurfaceArguments,
    "simulator": SimulatorArguments,
    "controller": RodControllerArgumets
}
