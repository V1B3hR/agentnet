"""Multi-modal support for AgentNet (text-first with extensibility hooks)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("agentnet.multimodal")


class ModalityType(str, Enum):
    """Supported modality types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code"
    STRUCTURED = "structured"  # JSON, XML, etc.


@dataclass
class ModalityContent:
    """Content with modality information."""

    content: Any
    modality: ModalityType
    metadata: Optional[Dict[str, Any]] = None
    encoding: Optional[str] = None  # base64, url, raw, etc.
    mime_type: Optional[str] = None

    def is_text(self) -> bool:
        """Check if content is text-based."""
        return self.modality in [
            ModalityType.TEXT,
            ModalityType.CODE,
            ModalityType.STRUCTURED,
        ]

    def get_text_representation(self) -> Optional[str]:
        """Get text representation of content."""
        if self.modality == ModalityType.TEXT:
            return str(self.content)
        elif self.modality == ModalityType.CODE:
            return str(self.content)
        elif self.modality == ModalityType.STRUCTURED:
            if isinstance(self.content, (dict, list)):
                import json

                return json.dumps(self.content, indent=2)
            return str(self.content)
        elif self.modality == ModalityType.DOCUMENT:
            # Could extract text from document
            return f"[DOCUMENT: {self.mime_type or 'unknown'}]"
        elif self.modality == ModalityType.IMAGE:
            # Could use OCR or image description
            return f"[IMAGE: {self.metadata.get('description', 'no description')}]"
        elif self.modality == ModalityType.AUDIO:
            # Could use speech-to-text
            return f"[AUDIO: {self.metadata.get('transcript', 'no transcript')}]"
        elif self.modality == ModalityType.VIDEO:
            # Could use video description
            return f"[VIDEO: {self.metadata.get('description', 'no description')}]"
        return None


class ModalityProcessor(ABC):
    """Base class for modality processors."""

    @abstractmethod
    def supported_modalities(self) -> List[ModalityType]:
        """Return list of supported modalities."""
        pass

    @abstractmethod
    def process(self, content: ModalityContent) -> ModalityContent:
        """Process content and return processed version."""
        pass

    @abstractmethod
    def validate(self, content: ModalityContent) -> bool:
        """Validate that content can be processed."""
        pass


class TextProcessor(ModalityProcessor):
    """Processor for text content."""

    def supported_modalities(self) -> List[ModalityType]:
        return [ModalityType.TEXT, ModalityType.CODE, ModalityType.STRUCTURED]

    def process(self, content: ModalityContent) -> ModalityContent:
        """Process text content (currently pass-through)."""
        if not self.validate(content):
            raise ValueError(f"Invalid content for TextProcessor: {content.modality}")

        # Text processing could include:
        # - Normalization
        # - Encoding conversion
        # - Language detection
        # - Preprocessing for embeddings

        return content

    def validate(self, content: ModalityContent) -> bool:
        """Validate text content."""
        return content.modality in self.supported_modalities() and isinstance(
            content.content, str
        )


class ImageProcessor(ModalityProcessor):
    """Processor for image content (placeholder)."""

    def supported_modalities(self) -> List[ModalityType]:
        return [ModalityType.IMAGE]

    def process(self, content: ModalityContent) -> ModalityContent:
        """Process image content."""
        if not self.validate(content):
            raise ValueError(f"Invalid content for ImageProcessor: {content.modality}")

        # Image processing could include:
        # - Format conversion
        # - Resizing/compression
        # - OCR text extraction
        # - Object detection
        # - Image description generation

        # For now, just add placeholder metadata
        processed_metadata = content.metadata or {}
        processed_metadata["processed"] = True
        processed_metadata["processor"] = "ImageProcessor"

        return ModalityContent(
            content=content.content,
            modality=content.modality,
            metadata=processed_metadata,
            encoding=content.encoding,
            mime_type=content.mime_type,
        )

    def validate(self, content: ModalityContent) -> bool:
        """Validate image content."""
        return content.modality == ModalityType.IMAGE and content.content is not None


class AudioProcessor(ModalityProcessor):
    """Processor for audio content (placeholder)."""

    def supported_modalities(self) -> List[ModalityType]:
        return [ModalityType.AUDIO]

    def process(self, content: ModalityContent) -> ModalityContent:
        """Process audio content."""
        if not self.validate(content):
            raise ValueError(f"Invalid content for AudioProcessor: {content.modality}")

        # Audio processing could include:
        # - Format conversion
        # - Speech-to-text transcription
        # - Audio feature extraction
        # - Noise reduction

        processed_metadata = content.metadata or {}
        processed_metadata["processed"] = True
        processed_metadata["processor"] = "AudioProcessor"

        return ModalityContent(
            content=content.content,
            modality=content.modality,
            metadata=processed_metadata,
            encoding=content.encoding,
            mime_type=content.mime_type,
        )

    def validate(self, content: ModalityContent) -> bool:
        """Validate audio content."""
        return content.modality == ModalityType.AUDIO and content.content is not None


class MultiModalMessage:
    """A message that can contain multiple modalities."""

    def __init__(self, primary_text: str = ""):
        self.primary_text = primary_text
        self.contents: List[ModalityContent] = []
        self.metadata: Dict[str, Any] = {}

    def add_content(self, content: ModalityContent) -> None:
        """Add content with modality information."""
        self.contents.append(content)

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add text content."""
        content = ModalityContent(
            content=text, modality=ModalityType.TEXT, metadata=metadata
        )
        self.add_content(content)

    def add_image(
        self,
        image_data: Any,
        encoding: str = "base64",
        mime_type: str = "image/png",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add image content."""
        content = ModalityContent(
            content=image_data,
            modality=ModalityType.IMAGE,
            encoding=encoding,
            mime_type=mime_type,
            metadata=metadata,
        )
        self.add_content(content)

    def get_text_only(self) -> str:
        """Get text-only representation of the message."""
        text_parts = [self.primary_text] if self.primary_text else []

        for content in self.contents:
            text_repr = content.get_text_representation()
            if text_repr:
                text_parts.append(text_repr)

        return "\n".join(text_parts)

    def get_modalities(self) -> List[ModalityType]:
        """Get list of modalities present in the message."""
        modalities = []
        if self.primary_text:
            modalities.append(ModalityType.TEXT)

        for content in self.contents:
            if content.modality not in modalities:
                modalities.append(content.modality)

        return modalities

    def has_modality(self, modality: ModalityType) -> bool:
        """Check if message contains specific modality."""
        return modality in self.get_modalities()


class MultiModalManager:
    """Manager for multi-modal content processing."""

    def __init__(self):
        self.processors: Dict[ModalityType, ModalityProcessor] = {}
        self.register_default_processors()

    def register_default_processors(self) -> None:
        """Register default processors for built-in modalities."""
        text_processor = TextProcessor()
        for modality in text_processor.supported_modalities():
            self.processors[modality] = text_processor

        # Register placeholders for other modalities
        # These can be replaced with actual implementations
        self.processors[ModalityType.IMAGE] = ImageProcessor()
        self.processors[ModalityType.AUDIO] = AudioProcessor()

    def register_processor(
        self, modality: ModalityType, processor: ModalityProcessor
    ) -> None:
        """Register a processor for a specific modality."""
        if modality not in processor.supported_modalities():
            raise ValueError(f"Processor does not support modality {modality}")

        self.processors[modality] = processor
        logger.info(
            f"Registered processor for {modality}: {processor.__class__.__name__}"
        )

    def process_content(self, content: ModalityContent) -> ModalityContent:
        """Process content using appropriate processor."""
        processor = self.processors.get(content.modality)
        if not processor:
            logger.warning(f"No processor registered for modality {content.modality}")
            return content

        try:
            return processor.process(content)
        except Exception as e:
            logger.error(f"Error processing {content.modality} content: {e}")
            return content

    def process_message(self, message: MultiModalMessage) -> MultiModalMessage:
        """Process all content in a multi-modal message."""
        processed_message = MultiModalMessage(message.primary_text)
        processed_message.metadata = message.metadata.copy()

        for content in message.contents:
            processed_content = self.process_content(content)
            processed_message.add_content(processed_content)

        return processed_message

    def validate_message(self, message: MultiModalMessage) -> bool:
        """Validate that all content in a message can be processed."""
        for content in message.contents:
            processor = self.processors.get(content.modality)
            if not processor or not processor.validate(content):
                return False
        return True

    def get_supported_modalities(self) -> List[ModalityType]:
        """Get list of supported modalities."""
        return list(self.processors.keys())

    def convert_to_text_only(self, message: MultiModalMessage) -> str:
        """Convert multi-modal message to text-only representation."""
        # Process message first to ensure all content is properly handled
        processed_message = self.process_message(message)
        return processed_message.get_text_only()


# Global multi-modal manager instance
_multimodal_manager: Optional[MultiModalManager] = None


def get_multimodal_manager() -> MultiModalManager:
    """Get global multi-modal manager."""
    global _multimodal_manager
    if _multimodal_manager is None:
        _multimodal_manager = MultiModalManager()
    return _multimodal_manager


def register_processor(modality: ModalityType, processor: ModalityProcessor) -> None:
    """Register a processor with the global manager."""
    manager = get_multimodal_manager()
    manager.register_processor(modality, processor)


def process_multimodal_content(content: ModalityContent) -> ModalityContent:
    """Process content using global manager."""
    manager = get_multimodal_manager()
    return manager.process_content(content)


def create_text_message(text: str) -> MultiModalMessage:
    """Create a simple text message."""
    return MultiModalMessage(primary_text=text)


def create_multimodal_message(
    primary_text: str = "", contents: Optional[List[ModalityContent]] = None
) -> MultiModalMessage:
    """Create a multi-modal message."""
    message = MultiModalMessage(primary_text)
    if contents:
        for content in contents:
            message.add_content(content)
    return message
