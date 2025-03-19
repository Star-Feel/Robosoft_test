import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, NewType, Optional, Sequence, Type, Union

import numpy as np
from transformers import HfArgumentParser

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
            return super().parse_dict(args, allow_extra_keys)

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
class SphereArguments:
    center: np.ndarray = field(default_factory=list)
    radius: float = field(default_factory=float)
    density: float = field(default_factory=float)


@dataclass
class SimulatorArguments:

    final_time: float = field(default_factory=float)
    time_step: float = field(default_factory=float)
    collect_data: bool = field(default_factory=bool)
    rendering_fps: float = field(default_factory=float)
    update_interval: int = field(default_factory=float)


@dataclass
class RodArguments:

    n_elem: int = field(default_factory=int)
    start: np.ndarray = field(default_factory=list)
    direction: np.ndarray = field(default_factory=list)
    normal: np.ndarray = field(default_factory=list)
    base_length: float = field(default=0.35)
    base_radius: float = field(default=0.0035)
    density: float = field(default=1000.0)
    youngs_modulus: float = field(default_factory=float)
    poisson_ratio: float = field(default=0.5)
