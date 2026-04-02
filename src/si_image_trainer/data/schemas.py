from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ReferenceRecord:
    city_code: str
    invader_id: str
    image_path: str
    role: str
    status: str
    source_type: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class QueryRecord:
    query_id: str
    image_path: str
    city_code: str | None
    city_name: str | None
    flash_id: str | None
    player: str | None
    observed_at: str | None
    label_invader_id: str | None
    split: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)
