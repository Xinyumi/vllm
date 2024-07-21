from typing import Any, Dict, List, Optional


class QServeQuantConfig:
    def __init__(
        self,
        weight_bits: int = 8,
    ) -> None:
        self.weight_bits = weight_bits

        if self.weight_bits not in [4, 8]:
            raise ValueError(
                "Currently, QServe only supports 4 or 8 bits for weight quantization, "
                f" but got {self.weight_bits}-bit weights here."
            )

    def __repr__(self) -> str:
        return f"QServeQuantConfig(weight_bits={self.weight_bits})"

    @classmethod
    def get_name(cls) -> str:
        return "qserve"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QServeQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        return cls(weight_bits)

    @classmethod
    def get_packed_dim(cls, tensor_name: str) -> Optional[int]:
        return None

    @classmethod
    def is_transposed(cls, tensor_name: str) -> bool:
        return False

    @classmethod
    def get_col_parallel_tensor_names(cls) -> List[str]:
        return ["weight"]

    @classmethod
    def get_row_parallel_tensor_names(cls) -> List[str]:
        return ["weight"]
