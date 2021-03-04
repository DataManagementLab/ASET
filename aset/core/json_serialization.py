"""JSON serialization."""
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Component(ABC):
    """
    JSON-serializable component.

    JSON-serializable data objects can extend this class and implement the two abstract methods. They provide the means
    to transform an object into a values so that it can be serialized by the dump and load methods of the Python
    json module.
    """

    @property
    @abstractmethod
    def json_dict(self):
        """JSON-serializable values representation of the component."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, values: dict):
        """Create a component from the values in the values."""
        raise NotImplementedError

    @property
    def json_str(self):
        """JSON string representation of the component."""
        return json.dumps(self.json_dict)

    @classmethod
    def from_json_str(cls, json_str: str):
        """Create a component from the given JSON string."""
        return cls.from_json_dict(json.loads(json_str))
