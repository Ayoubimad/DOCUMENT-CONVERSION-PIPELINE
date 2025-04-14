from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import colorlog
import logging


@dataclass
class RAGGenerationConfig:
    """Configuration for the language model generation part of RAG"""

    model: str
    api_base: str
    temperature: float = 0.7
    max_tokens: int = 1500
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }


@dataclass
class SearchSettings:
    """Configuration for the search part of RAG"""

    limit: int = 10
    search_strategy: Optional[str] = None
    use_hybrid_search: bool = False
    use_semantic_search: bool = True
    filters: Optional[Dict[str, Any]] = None
    graph_settings: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        settings = {
            "limit": self.limit,
            "use_hybrid_search": self.use_hybrid_search,
            "use_semantic_search": self.use_semantic_search,
        }

        if self.search_strategy:
            settings["search_strategy"] = self.search_strategy
        if self.filters:
            settings["filters"] = self.filters
        if self.graph_settings:
            settings["graph_settings"] = self.graph_settings

        return settings


@dataclass
class RAGConfig:
    """Configuration for RAG system testing"""

    generation_config: RAGGenerationConfig
    search_settings: SearchSettings
    search_mode: Literal["basic", "advanced", "custom"] = "custom"
    task_prompt: Optional[str] = None
    include_title_if_available: bool = False
    include_web_search: bool = False

    @property
    def name(self) -> str:
        """Generate a short name for this configuration"""
        model_name = self.generation_config.model.split("/")[-1]
        search_type = "hybrid" if self.search_settings.use_hybrid_search else "semantic"
        return f"{model_name}_{search_type}_{self.search_settings.limit}"

    def to_rag_params(self) -> Dict[str, Any]:
        """Convert the configuration to R2R RAG API parameters"""
        params = {
            "rag_generation_config": self.generation_config.to_dict(),
            "search_settings": self.search_settings.to_dict(),
            "search_mode": self.search_mode,
            "include_title_if_available": self.include_title_if_available,
            "include_web_search": self.include_web_search,
        }

        if self.task_prompt:
            params["task_prompt"] = self.task_prompt

        return params


def setup_logger() -> logging.Logger:
    """Configure and return the logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }

        formatter = colorlog.ColoredFormatter(
            "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
            log_colors=colors,
            reset=True,
            style="%",
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
