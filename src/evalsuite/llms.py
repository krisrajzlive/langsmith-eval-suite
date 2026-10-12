"""Provider factories for application models, judge models and embeddings."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import get_settings


STRUCTURED_OUTPUT_ATTEMPTS = 6


def _tool_calling_ollama():
    """ChatOllama that structures output through tool calls, retrying when the model answers in prose.

    langchain-ollama defaults to JSON-schema constrained decoding, which some hosted models (for example
    gpt-oss) ignore. Tool calling is honored, but intermittently the model replies without a tool call, which
    parses to None. Retrying that call at this layer keeps every judge consistent.
    """
    from langchain_core.runnables import RunnableLambda
    from langchain_ollama import ChatOllama

    class ToolCallingChatOllama(ChatOllama):
        def with_structured_output(self, schema, *, method="function_calling", include_raw=False, **kwargs):
            inner = super().with_structured_output(schema, method=method, include_raw=True, **kwargs)

            def invoke(prompt, config=None):
                for _ in range(STRUCTURED_OUTPUT_ATTEMPTS):
                    result = inner.invoke(prompt, config)
                    if result.get("parsed") is not None:
                        return result if include_raw else result["parsed"]
                raise ValueError(f"model returned no structured output after {STRUCTURED_OUTPUT_ATTEMPTS} attempts")

            return RunnableLambda(invoke)

    return ToolCallingChatOllama


def build_chat_model(provider: str, model: str, *, temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, max_retries=6)
    if provider == "ollama":
        headers = {}
        if key := os.environ.get("OLLAMA_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        return _tool_calling_ollama()(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            client_kwargs={"headers": headers},
        )
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Credentials come from the standard AWS environment variables or profile; nothing is stored here.
        return ChatBedrockConverse(
            model=model, temperature=temperature, region_name=settings.aws_region, max_tokens=2048
        )
    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            temperature=max(temperature, 0.01),
            huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        )
        return ChatHuggingFace(llm=endpoint)
    raise ValueError(f"Unsupported provider: {provider!r}")


def get_app_model() -> BaseChatModel:
    s = get_settings()
    return build_chat_model(s.app_provider, s.app_model)


def get_judge_model() -> BaseChatModel:
    s = get_settings()
    return build_chat_model(s.judge_provider, s.judge_model)


def get_embeddings() -> Embeddings | None:
    """Return a dense embedding model, or None when the offline lexical retriever is selected."""
    s = get_settings()
    if s.embeddings_provider == "lexical":
        return None
    if s.embeddings_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=s.embeddings_model, huggingfacehub_api_token=os.environ.get("HF_TOKEN")
        )
    if s.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embeddings_model)
    raise ValueError(f"Unsupported embeddings provider: {s.embeddings_provider!r}")
