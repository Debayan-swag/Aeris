import os
from io import BytesIO
from typing import Any, Dict, List, Literal, TypedDict

import cv2
import faiss
import numpy as np
import open_clip
import pandas as pd
import torch

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langgraph.graph import StateGraph, START, END


load_dotenv()


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError("NVIDIA_API_KEY was not found in the environment.")

MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"

DISTILGPT2_DIR = os.getenv(
    "DISTILGPT2_DIR",
    r"C:\Users\Debayan\OneDrive\Desktop\Aeris\models\DistilGPT2"
)

REMOTECLIP_DIR = os.getenv(
    "REMOTECLIP_DIR",
    r"C:\Users\Debayan\OneDrive\Desktop\Aeris\models\RemoteCLIP"
)

EMBEDDING_DIR = os.getenv(
    "EMBEDDING_DIR",
    r"C:\Users\Debayan\OneDrive\Desktop\Aeris\remoteclip_embeddings"
)

REMOTECLIP_MODEL_NAME = "ViT-B-32"

REMOTECLIP_EMBEDDINGS_PATH = os.path.join(
    EMBEDDING_DIR,
    "remoteclip_embeddings.npy"
)

REMOTECLIP_INDEX_PATH = os.path.join(
    EMBEDDING_DIR,
    "remoteclip_index.parquet"
)

REMOTECLIP_FAISS_PATH = os.path.join(
    EMBEDDING_DIR,
    "remoteclip.faiss"
)

NORMAL_TOP_K = 20
NORMAL_FINAL_K = 5
CHANGE_TOP_K = 3
DISTILGPT2_MAX_NEW_TOKENS = 8


STATUS = {
    "intent": "Understanding your request...",
    "image_analysis": "Interpreting the uploaded satellite image...",
    "semantic_analysis": "Extracting relevant visual concepts...",
    "image_embedding": "Creating the image representation...",
    "description": "Preparing a detailed satellite description...",
    "archive_search": "Searching the satellite image archive...",
    "ranking": "Ranking the strongest matches...",
    "comparison": "Comparing the source image with retrieved candidates...",
    "no_data": "No matching picture was found in the database.",
    "similar_question": "Would you like to see similar pictures instead?"
}


class IntentResult(BaseModel):
    intent: Literal["normal_retriever", "change_detection"]

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"normal_retriever", "change_detection"}:
            raise ValueError("Invalid intent.")
        return value


class SemanticEvidence(BaseModel):
    concepts: List[str] = Field(default_factory=list)
    embedding_dimension: int
    evidence_text: str = ""

    @field_validator("concepts")
    @classmethod
    def clean_concepts(cls, value: List[str]) -> List[str]:
        cleaned = []
        for concept in value:
            concept = concept.strip()
            if concept and concept not in cleaned:
                cleaned.append(concept)
        return cleaned


class ImageTextState(TypedDict, total=False):
    image: Any
    image_name: str
    user_text: str
    intent: str
    semantic_evidence: Dict[str, Any]
    description: str
    embedding: Any
    retrieved_results: List[Dict[str, Any]]
    top_results: List[Dict[str, Any]]
    change_results: List[Dict[str, Any]]
    no_data: bool


nvidia_llm = ChatNVIDIA(
    model=MODEL_NAME,
    api_key=NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)


DISTILGPT2_REPO_ID = "distilbert/distilgpt2"

distilgpt2_tokenizer = AutoTokenizer.from_pretrained(
    DISTILGPT2_REPO_ID,
    cache_dir=DISTILGPT2_DIR
)

distilgpt2_model = AutoModelForCausalLM.from_pretrained(
    DISTILGPT2_REPO_ID,
    cache_dir=DISTILGPT2_DIR
)

distilgpt2_device = "cuda" if torch.cuda.is_available() else "cpu"
distilgpt2_model = distilgpt2_model.to(distilgpt2_device)
distilgpt2_model.eval()

if distilgpt2_tokenizer.pad_token is None:
    distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token


remoteclip_checkpoint = next(
    os.path.join(REMOTECLIP_DIR, file_name)
    for file_name in os.listdir(REMOTECLIP_DIR)
    if file_name.endswith(".pt")
)

remoteclip_model, _, remoteclip_preprocess = (
    open_clip.create_model_and_transforms(
        REMOTECLIP_MODEL_NAME
    )
)

remoteclip_tokenizer = open_clip.get_tokenizer(
    REMOTECLIP_MODEL_NAME
)

remoteclip_checkpoint_data = torch.load(
    remoteclip_checkpoint,
    map_location="cpu"
)

if "state_dict" in remoteclip_checkpoint_data:
    remoteclip_checkpoint_data = remoteclip_checkpoint_data["state_dict"]

remoteclip_model.load_state_dict(
    remoteclip_checkpoint_data,
    strict=True
)

remoteclip_device = "cuda" if torch.cuda.is_available() else "cpu"
remoteclip_model = remoteclip_model.to(remoteclip_device)
remoteclip_model.eval()


intent_prompt = """
You are the intent router for a remote-sensing satellite imagery system.

The user provides an image and text.

There are exactly two possible intents.

normal_retriever:
The user wants to search the satellite-image database using the uploaded image and accompanying text and retrieve semantically related images.

change_detection:
The user wants to determine whether the uploaded source image is different from images in the database. The system retrieves the three strongest candidates and compares the source image against them.

Understand the complete user context before deciding.

Return exactly one label:
normal_retriever
or
change_detection

Do not answer the user.
Do not invent context.
"""


description_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the response-generation layer of a satellite imagery analysis system. Generate a sophisticated, natural and engaging description based only on the supplied user request and grounded semantic evidence. Do not invent objects, locations, coordinates, dates, sensors, events, weather, relationships or visual details that are not supported. Do not claim certainty when the evidence is insufficient. Do not mention models, embeddings, prompts, APIs or internal implementation. The result should feel natural and useful to a user exploring satellite imagery."
        ),
        (
            "human",
            "User request: {user_text}\nGrounded semantic evidence: {semantic_evidence}"
        )
    ]
)

description_chain = description_prompt | nvidia_llm


def detect_intent(user_text: str) -> str:
    prompt = (
        intent_prompt
        + "\nUser request:\n"
        + user_text
        + "\n\nIntent:"
    )

    inputs = distilgpt2_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(distilgpt2_device)

    with torch.no_grad():
        output = distilgpt2_model.generate(
            **inputs,
            max_new_tokens=DISTILGPT2_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=distilgpt2_tokenizer.eos_token_id
        )

    generated = distilgpt2_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    if "change_detection" in generated:
        return "change_detection"

    if "normal_retriever" in generated:
        return "normal_retriever"

    change_terms = [
        "change detection",
        "changed",
        "difference",
        "different",
        "before and after",
        "compare",
        "comparison",
        "what changed",
        "detect changes"
    ]

    if any(term in user_text.lower() for term in change_terms):
        return "change_detection"

    return "normal_retriever"


def generate_semantic_evidence(
    image: Image.Image,
    user_text: str = ""
) -> Dict[str, Any]:

    image_input = remoteclip_preprocess(
        image.convert("RGB")
    ).unsqueeze(0).to(remoteclip_device)

    concepts = [
        "airport",
        "airplane",
        "runway",
        "road",
        "highway",
        "bridge",
        "building",
        "dense urban area",
        "residential area",
        "industrial area",
        "farmland",
        "agriculture",
        "forest",
        "vegetation",
        "river",
        "lake",
        "water body",
        "coastline",
        "harbor",
        "stadium",
        "parking area",
        "construction site",
        "solar panels",
        "railway",
        "intersection"
    ]

    text_input = remoteclip_tokenizer(
        concepts
    ).to(remoteclip_device)

    with torch.no_grad():
        image_features = remoteclip_model.encode_image(
            image_input
        )

        text_features = remoteclip_model.encode_text(
            text_input
        )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        similarities = (
            image_features @ text_features.T
        ).squeeze(0)

    top_indices = torch.topk(
        similarities,
        k=min(8, len(concepts))
    ).indices.cpu().tolist()

    selected_concepts = [
        concepts[index]
        for index in top_indices
    ]

    evidence = SemanticEvidence(
        concepts=selected_concepts,
        embedding_dimension=int(
            image_features.shape[-1]
        ),
        evidence_text=(
            "The uploaded satellite image has strongest "
            "semantic associations with: "
            + ", ".join(selected_concepts)
            + "."
        )
    )

    return evidence.model_dump()


def generate_image_embedding(
    image: Image.Image
) -> np.ndarray:

    image_input = remoteclip_preprocess(
        image.convert("RGB")
    ).unsqueeze(0).to(remoteclip_device)

    with torch.no_grad():
        embedding = remoteclip_model.encode_image(
            image_input
        )

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return embedding.cpu().numpy().astype(
        np.float32
    )


def generate_nim_description(
    user_text: str,
    semantic_evidence: Dict[str, Any]
) -> str:

    response = description_chain.invoke(
        {
            "user_text": user_text,
            "semantic_evidence": semantic_evidence
        }
    )

    return response.content.strip()


def load_vector_resources():
    vector_store = faiss.read_index(
        REMOTECLIP_FAISS_PATH
    )

    index_mapping = pd.read_parquet(
        REMOTECLIP_INDEX_PATH
    )

    embeddings = np.load(
        REMOTECLIP_EMBEDDINGS_PATH
    )

    return vector_store, index_mapping, embeddings


def similarity_search(
    embedding: np.ndarray,
    vector_store: Any,
    top_k: int
) -> List[Dict[str, Any]]:

    scores, indices = vector_store.search(
        embedding,
        top_k
    )

    results = []

    for index, score in zip(
        indices[0],
        scores[0]
    ):
        if int(index) < 0:
            continue

        results.append(
            {
                "index": int(index),
                "score": float(score)
            }
        )

    return results


def resolve_image_names(
    results: List[Dict[str, Any]],
    index_mapping: pd.DataFrame
) -> List[Dict[str, Any]]:

    resolved = []

    for result in results:
        index = int(result["index"])

        row = index_mapping.iloc[index]

        resolved.append(
            {
                **result,
                "image_name": row["image_name"]
            }
        )

    return resolved


def retrieve_images(
    embedding: np.ndarray,
    vector_store: Any,
    index_mapping: pd.DataFrame,
    top_k: int
) -> List[Dict[str, Any]]:

    results = similarity_search(
        embedding,
        vector_store,
        top_k
    )

    return resolve_image_names(
        results,
        index_mapping
    )


def select_top_five(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    ranked = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )

    unique_results = []
    seen = set()

    for result in ranked:
        image_name = result["image_name"]

        if image_name in seen:
            continue

        seen.add(image_name)
        unique_results.append(result)

        if len(unique_results) == NORMAL_FINAL_K:
            break

    return unique_results


def get_image_by_name(
    image_name: str,
    dataframe: pd.DataFrame
) -> Image.Image:

    rows = dataframe.loc[
        dataframe["image_name"] == image_name
    ]

    if rows.empty:
        raise FileNotFoundError(
            f"Image not found: {image_name}"
        )

    image_bytes = rows.iloc[0]["image"]

    return Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")


def compare_images(
    source_image: Image.Image,
    candidate_image: Image.Image
) -> Dict[str, Any]:

    source = np.asarray(
        source_image.convert("RGB")
    )

    candidate = np.asarray(
        candidate_image.convert("RGB")
    )

    candidate = cv2.resize(
        candidate,
        (source.shape[1], source.shape[0])
    )

    source_gray = cv2.cvtColor(
        source,
        cv2.COLOR_RGB2GRAY
    )

    candidate_gray = cv2.cvtColor(
        candidate,
        cv2.COLOR_RGB2GRAY
    )

    source_gray = cv2.GaussianBlur(
        source_gray,
        (5, 5),
        0
    )

    candidate_gray = cv2.GaussianBlur(
        candidate_gray,
        (5, 5),
        0
    )

    difference = cv2.absdiff(
        source_gray,
        candidate_gray
    )

    _, threshold = cv2.threshold(
        difference,
        30,
        255,
        cv2.THRESH_BINARY
    )

    changed_pixels = np.count_nonzero(
        threshold
    )

    total_pixels = threshold.size

    change_ratio = (
        changed_pixels / total_pixels
        if total_pixels
        else 0.0
    )

    return {
        "change_ratio": float(change_ratio),
        "different": change_ratio > 0.10
    }


def compare_source_with_top_three(
    source_image: Image.Image,
    results: List[Dict[str, Any]],
    dataframe: pd.DataFrame
) -> List[Dict[str, Any]]:

    comparisons = []

    for result in results[:CHANGE_TOP_K]:
        image_name = result["image_name"]

        candidate_image = get_image_by_name(
            image_name,
            dataframe
        )

        comparison = compare_images(
            source_image,
            candidate_image
        )

        comparisons.append(
            {
                "image_name": image_name,
                "retrieval_score": result["score"],
                **comparison
            }
        )

    return comparisons


def intent_node(
    state: ImageTextState
) -> Dict[str, Any]:
    return {
        "intent": detect_intent(
            state["user_text"]
        )
    }


def normal_parallel_node(
    state: ImageTextState
) -> Dict[str, Any]:

    parallel = RunnableParallel(
        semantic_evidence=RunnableLambda(
            lambda _: generate_semantic_evidence(
                state["image"],
                state["user_text"]
            )
        ),
        embedding=RunnableLambda(
            lambda _: generate_image_embedding(
                state["image"]
            )
        )
    )

    return parallel.invoke({})


def normal_description_node(
    state: ImageTextState
) -> Dict[str, Any]:

    description = generate_nim_description(
        state["user_text"],
        state["semantic_evidence"]
    )

    return {
        "description": description
    }


def normal_retrieval_node(
    state: ImageTextState,
    vector_store: Any,
    index_mapping: pd.DataFrame
) -> Dict[str, Any]:

    results = retrieve_images(
        state["embedding"],
        vector_store,
        index_mapping,
        NORMAL_TOP_K
    )

    return {
        "retrieved_results": results,
        "top_results": select_top_five(results)
    }


def change_parallel_node(
    state: ImageTextState
) -> Dict[str, Any]:

    parallel = RunnableParallel(
        semantic_evidence=RunnableLambda(
            lambda _: generate_semantic_evidence(
                state["image"],
                state["user_text"]
            )
        ),
        embedding=RunnableLambda(
            lambda _: generate_image_embedding(
                state["image"]
            )
        )
    )

    return parallel.invoke({})


def change_description_node(
    state: ImageTextState
) -> Dict[str, Any]:

    description = generate_nim_description(
        state["user_text"],
        state["semantic_evidence"]
    )

    return {
        "description": description
    }


def change_retrieval_node(
    state: ImageTextState,
    vector_store: Any,
    index_mapping: pd.DataFrame
) -> Dict[str, Any]:

    results = retrieve_images(
        state["embedding"],
        vector_store,
        index_mapping,
        CHANGE_TOP_K
    )

    return {
        "retrieved_results": results
    }


def opencv_comparison_node(
    state: ImageTextState,
    dataframe: pd.DataFrame
) -> Dict[str, Any]:

    comparisons = compare_source_with_top_three(
        state["image"],
        state["retrieved_results"],
        dataframe
    )

    similar_found = any(
        not result["different"]
        for result in comparisons
    )

    return {
        "change_results": comparisons,
        "no_data": not similar_found
    }


def similar_branch(
    state: ImageTextState
) -> Dict[str, Any]:
    pass


def build_image_text_graph(
    vector_store: Any,
    index_mapping: pd.DataFrame,
    dataframe: pd.DataFrame
):

    graph = StateGraph(ImageTextState)

    graph.add_node(
        "intent_detection",
        RunnableLambda(intent_node)
    )

    graph.add_node(
        "normal_parallel",
        RunnableLambda(normal_parallel_node)
    )

    graph.add_node(
        "normal_description",
        RunnableLambda(normal_description_node)
    )

    graph.add_node(
        "normal_retrieval",
        RunnableLambda(
            lambda state: normal_retrieval_node(
                state,
                vector_store,
                index_mapping
            )
        )
    )

    graph.add_node(
        "change_parallel",
        RunnableLambda(change_parallel_node)
    )

    graph.add_node(
        "change_description",
        RunnableLambda(change_description_node)
    )

    graph.add_node(
        "change_retrieval",
        RunnableLambda(
            lambda state: change_retrieval_node(
                state,
                vector_store,
                index_mapping
            )
        )
    )

    graph.add_node(
        "opencv_comparison",
        RunnableLambda(
            lambda state: opencv_comparison_node(
                state,
                dataframe
            )
        )
    )

    graph.add_node(
        "similar_branch",
        RunnableLambda(similar_branch)
    )

    graph.add_edge(
        START,
        "intent_detection"
    )

    graph.add_conditional_edges(
        "intent_detection",
        lambda state: state["intent"],
        {
            "normal_retriever": "normal_parallel",
            "change_detection": "change_parallel"
        }
    )

    graph.add_edge(
        "normal_parallel",
        "normal_description"
    )

    graph.add_edge(
        "normal_parallel",
        "normal_retrieval"
    )

    graph.add_edge(
        "normal_description",
        END
    )

    graph.add_edge(
        "normal_retrieval",
        END
    )

    graph.add_edge(
        "change_parallel",
        "change_description"
    )

    graph.add_edge(
        "change_parallel",
        "change_retrieval"
    )

    graph.add_edge(
        "change_description",
        END
    )

    graph.add_edge(
        "change_retrieval",
        "opencv_comparison"
    )

    graph.add_conditional_edges(
        "opencv_comparison",
        lambda state: (
            "similar"
            if not state.get("no_data", False)
            else "different"
        ),
        {
            "similar": "similar_branch",
            "different": END
        }
    )

    graph.add_edge(
        "similar_branch",
        END
    )

    return graph.compile()


def run_image_text_pipeline(
    image: Image.Image,
    user_text: str,
    vector_store: Any,
    index_mapping: pd.DataFrame,
    dataframe: pd.DataFrame,
    image_name: str = ""
) -> Dict[str, Any]:

    graph = build_image_text_graph(
        vector_store=vector_store,
        index_mapping=index_mapping,
        dataframe=dataframe
    )

    return graph.invoke(
        {
            "image": image,
            "image_name": image_name,
            "user_text": user_text
        }
    )


def stream_nim_description(
    user_text: str,
    semantic_evidence: Dict[str, Any]
):

    prompt = (
        f"User request: {user_text}\n"
        f"Grounded semantic evidence: {semantic_evidence}"
    )

    for chunk in nvidia_llm.stream(
        [
            {
                "role": "user",
                "content": prompt
            }
        ]
    ):

        if (
            chunk.additional_kwargs
            and "reasoning_content"
            in chunk.additional_kwargs
        ):
            yield {
                "type": "reasoning",
                "data": chunk.additional_kwargs[
                    "reasoning_content"
                ]
            }

        if chunk.content:
            yield {
                "type": "content",
                "data": chunk.content
            }


def stream_image_text_pipeline(
    image: Image.Image,
    user_text: str,
    vector_store: Any,
    index_mapping: pd.DataFrame,
    dataframe: pd.DataFrame,
    image_name: str = ""
):

    yield {
        "type": "status",
        "data": STATUS["intent"]
    }

    intent = detect_intent(user_text)

    yield {
        "type": "intent",
        "data": intent
    }

    yield {
        "type": "status",
        "data": STATUS["image_analysis"]
    }

    yield {
        "type": "status",
        "data": STATUS["semantic_analysis"]
    }

    if intent == "normal_retriever":

        parallel = RunnableParallel(
            semantic_evidence=RunnableLambda(
                lambda _: generate_semantic_evidence(
                    image,
                    user_text
                )
            ),
            embedding=RunnableLambda(
                lambda _: generate_image_embedding(
                    image
                )
            )
        )

        parallel_result = parallel.invoke({})

        yield {
            "type": "status",
            "data": STATUS["description"]
        }

        yield {
            "type": "semantic_evidence",
            "data": parallel_result[
                "semantic_evidence"
            ]
        }

        yield from stream_nim_description(
            user_text,
            parallel_result["semantic_evidence"]
        )

        yield {
            "type": "status",
            "data": STATUS["archive_search"]
        }

        results = retrieve_images(
            parallel_result["embedding"],
            vector_store,
            index_mapping,
            NORMAL_TOP_K
        )

        yield {
            "type": "status",
            "data": STATUS["ranking"]
        }

        yield {
            "type": "retrieval_results",
            "data": select_top_five(results)
        }

        return

    parallel = RunnableParallel(
        semantic_evidence=RunnableLambda(
            lambda _: generate_semantic_evidence(
                image,
                user_text
            )
        ),
        embedding=RunnableLambda(
            lambda _: generate_image_embedding(
                image
            )
        )
    )

    parallel_result = parallel.invoke({})

    yield {
        "type": "status",
        "data": STATUS["description"]
    }

    yield {
        "type": "semantic_evidence",
        "data": parallel_result[
            "semantic_evidence"
        ]
    }

    yield from stream_nim_description(
        user_text,
        parallel_result["semantic_evidence"]
    )

    yield {
        "type": "status",
        "data": STATUS["archive_search"]
    }

    results = retrieve_images(
        parallel_result["embedding"],
        vector_store,
        index_mapping,
        CHANGE_TOP_K
    )

    yield {
        "type": "status",
        "data": STATUS["comparison"]
    }

    comparisons = compare_source_with_top_three(
        image,
        results,
        dataframe
    )

    yield {
        "type": "change_results",
        "data": comparisons
    }

    similar_found = any(
        not result["different"]
        for result in comparisons
    )

    if not similar_found:

        yield {
            "type": "status",
            "data": STATUS["no_data"]
        }

        yield {
            "type": "no_data",
            "data": STATUS["no_data"]
        }

        yield {
            "type": "user_confirmation_required",
            "data": STATUS["similar_question"]
        }

    else:
        pass