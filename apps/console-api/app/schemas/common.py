from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConsoleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageResponse(ConsoleModel):
    message: str

