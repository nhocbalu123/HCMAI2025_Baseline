
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
)
from schema.response import KeyframeServiceReponse, SingleKeyframeDisplay, KeyframeDisplay
from controller.query_controller import QueryController
from core.dependencies import get_query_controller, get_translator_service
from core.logger import SimpleLogger
from core.settings import ImageSettings
from service.translator_service import TranslatorService

from pathlib import Path


image_settings = ImageSettings()
BASE_IMAGE_URL = image_settings.BASE_URL
logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/keyframe",
    tags=["keyframe"],
    responses={404: {"description": "Not found"}},
)


def convert_string_as_list_to_list(string_as_list: str):
    if len(string_as_list) > 0:
        return [x.strip() for x in string_as_list.split(',') if x.strip()]

    return []


def compose_input_query(translator, request) -> str:
    en_query = request.query  # same as input query, no translation

    if request.using_translator:
        en_query = translator.perform(request.query)
        print(f"-------search query in vi: {request.query}")
        print(f"-------search query in en: {en_query}")
    
    return en_query


@router.post(
    "/search",
    response_model=KeyframeDisplay,
    summary="Simple text search for keyframes",
    description="""
    Perform a simple text-based search for keyframes using semantic similarity.
    
    This endpoint converts the input text query to an embedding and searches for 
    the most similar keyframes in the database.
    
    **Parameters:**
    - **query**: The search text (1-1000 characters)
    - **top_k**: Maximum number of results to return (1-100, default: 10)
    - **score_threshold**: Minimum confidence score (0.0-1.0, default: 0.0)
    
    **Returns:**
    List of keyframes with their metadata and confidence scores, ordered by similarity.
    
    **Example:**
    ```json
    {
        "query": "person walking in the park",
        "top_k": 5,
        "score_threshold": 0.7,
        "using_translator": False
    }
    ```
    """,
    response_description="List of matching keyframes with confidence scores"
)
async def search_keyframes(
    request: TextSearchRequest,
    controller: QueryController = Depends(get_query_controller),
    translator: TranslatorService | None = Depends(get_translator_service)
):
    """
    Search for keyframes using text query with semantic similarity.
    """

    logger.info(
        (
            "Text search request: "
            f"query='{request.query}', "
            f"top_k={request.top_k}, "
            f"threshold={request.score_threshold}, "
            f"using_translator={request.using_translator}"
        )
    )

    query = compose_input_query(translator=translator, request=request)

    results = await controller.search_text(
        query=query,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )

    logger.info(f"Found {len(results)} results for query: '{request.query}'")
    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results)
        )
    )
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/exclude-groups",
    response_model=KeyframeDisplay,
    summary="Text search with group exclusion",
    description="""
    Perform text-based search for keyframes while excluding specific groups.
    
    This endpoint allows you to search for keyframes while filtering out 
    results from specified groups (e.g., to avoid certain video categories).
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **exclude_groups**: List of group IDs to exclude from results
    
    **Use Cases:**
    - Exclude specific video categories or datasets
    - Filter out content from certain time periods
    - Remove specific collections from search results
    
    **Example:**
    ```json
    {
        "query": "sunset landscape",
        "top_k": 15,
        "score_threshold": 0.6,
        "exclude_groups": [1, 3, 7]
    }
    ```
    """,
    response_description="List of matching keyframes excluding specified groups"
)
async def search_keyframes_exclude_groups(
    request: TextSearchWithExcludeGroupsRequest,
    controller: QueryController = Depends(get_query_controller),
    translator: TranslatorService | None = Depends(get_translator_service)
):
    """
    Search for keyframes with group exclusion filtering.
    """

    logger.info(f"Text search with group exclusion: query='{request.query}', exclude_groups={request.exclude_groups}")

    query = compose_input_query(translator=translator, request=request)

    results: list[KeyframeServiceReponse] = await controller.search_text_with_exclude_group(
        query=query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_group_exlude=convert_string_as_list_to_list(
            request.exclude_groups
        )
    )
    
    logger.info(f"Found {len(results)} results excluding groups {request.exclude_groups}")\

    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results)
        )
    )
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/selected-groups-videos",
    response_model=KeyframeDisplay,
    summary="Text search within selected groups and videos",
    description="""
    Perform text-based search for keyframes within specific groups and videos only.
    
    This endpoint allows you to limit your search to specific groups and videos,
    effectively creating a filtered search scope.
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **include_groups**: List of group IDs to search within
    - **include_videos**: List of video IDs to search within
    
    **Behavior:**
    - Only keyframes from the specified groups AND videos will be searched
    - If a keyframe belongs to an included group OR an included video, it will be considered
    - Empty lists mean no filtering for that category
    
    **Use Cases:**
    - Search within specific video collections
    - Focus on particular time periods or datasets
    - Limit search to curated content sets
    
    **Example:**
    ```json
    {
        "query": "car driving on highway",
        "top_k": 20,
        "score_threshold": 0.5,
        "include_groups": [2, 4, 6],
        "include_videos": [101, 102, 203, 204]
    }
    ```
    """,
    response_description="List of matching keyframes from selected groups and videos"
)
async def search_keyframes_selected_groups_videos(
    request: TextSearchWithSelectedGroupsAndVideosRequest,
    controller: QueryController = Depends(get_query_controller),
    translator: TranslatorService | None = Depends(get_translator_service)
):
    """
    Search for keyframes within selected groups and videos.
    """

    logger.info(f"Text search with selection: query='{request.query}', include_groups={request.include_groups}, include_videos={request.include_videos}")
    
    query = compose_input_query(translator=translator, request=request)

    results = await controller.search_with_selected_video_group(
        query=query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_of_include_groups=convert_string_as_list_to_list(
            request.include_groups
        ),
        list_of_include_videos=convert_string_as_list_to_list(
            request.include_videos
        )
    )

    logger.info(f"Found {len(results)} results within selected groups/videos")

    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results)
        )
    )
    return KeyframeDisplay(results=display_results)
