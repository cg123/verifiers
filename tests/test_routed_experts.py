"""Tests for MoE routed experts extraction in response_utils."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from verifiers.utils.response_utils import (
    _extract_routed_experts,
    parse_response_tokens,
)


class TestExtractRoutedExperts:
    """Tests for the _extract_routed_experts helper function."""

    def test_returns_none_for_none_input(self):
        result = _extract_routed_experts(None)
        assert result is None

    def test_converts_numpy_array_to_list(self):
        # Shape: [seq_len=3, num_layers=2, topk=2]
        routed_experts = np.array([
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
            [[8, 9], [10, 11]],
        ])
        result = _extract_routed_experts(routed_experts)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == [[0, 1], [2, 3]]
        assert result[1] == [[4, 5], [6, 7]]
        assert result[2] == [[8, 9], [10, 11]]

    def test_passes_through_list(self):
        routed_experts = [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ]
        result = _extract_routed_experts(routed_experts)

        assert result is not None
        assert result == routed_experts


class TestParseResponseTokensWithRoutedExperts:
    """Tests for parse_response_tokens with routed experts."""

    @pytest.fixture
    def mock_chat_response_with_routed_experts(self):
        """Create a mock ChatCompletion with routed_experts."""
        from openai.types.chat.chat_completion import ChatCompletion

        response = MagicMock(spec=ChatCompletion)
        choice = MagicMock()

        # Setup basic token data
        response.prompt_token_ids = [1, 2, 3]  # 3 prompt tokens
        choice.token_ids = [4, 5, 6, 7]  # 4 completion tokens
        choice.finish_reason = "stop"

        # Setup logprobs
        logprobs_content = [
            MagicMock(logprob=-0.1),
            MagicMock(logprob=-0.2),
            MagicMock(logprob=-0.3),
            MagicMock(logprob=-0.4),
        ]
        choice.logprobs = MagicMock()
        choice.logprobs.content = logprobs_content

        # Setup routed_experts - full sequence [7 tokens, 2 layers, 2 topk]
        choice.routed_experts = np.array([
            [[0, 1], [2, 3]],  # prompt token 0
            [[4, 5], [6, 7]],  # prompt token 1
            [[8, 9], [10, 11]],  # prompt token 2
            [[12, 13], [14, 15]],  # completion token 0
            [[16, 17], [18, 19]],  # completion token 1
            [[20, 21], [22, 23]],  # completion token 2
            [[24, 25], [26, 27]],  # completion token 3
        ])

        response.choices = [choice]
        return response

    @pytest.fixture
    def mock_chat_response_without_routed_experts(self):
        """Create a mock ChatCompletion without routed_experts."""
        from openai.types.chat.chat_completion import ChatCompletion

        response = MagicMock(spec=ChatCompletion)
        choice = MagicMock()

        # Setup basic token data
        response.prompt_token_ids = [1, 2, 3]
        choice.token_ids = [4, 5, 6, 7]
        choice.finish_reason = "stop"

        # Setup logprobs
        logprobs_content = [
            MagicMock(logprob=-0.1),
            MagicMock(logprob=-0.2),
            MagicMock(logprob=-0.3),
            MagicMock(logprob=-0.4),
        ]
        choice.logprobs = MagicMock()
        choice.logprobs.content = logprobs_content

        # No routed_experts attribute
        del choice.routed_experts

        response.choices = [choice]
        return response

    @pytest.mark.asyncio
    async def test_extracts_full_sequence_routed_experts(
        self, mock_chat_response_with_routed_experts
    ):
        result = await parse_response_tokens(
            mock_chat_response_with_routed_experts, "chat"
        )

        assert result is not None
        assert result["routed_experts"] is not None
        # Full sequence: 3 prompt + 4 completion = 7 tokens
        assert len(result["routed_experts"]) == 7
        # Check prompt token routing is included
        assert result["routed_experts"][0] == [[0, 1], [2, 3]]
        # Check completion token routing
        assert result["routed_experts"][3] == [[12, 13], [14, 15]]

    @pytest.mark.asyncio
    async def test_returns_none_when_routed_experts_not_present(
        self, mock_chat_response_without_routed_experts
    ):
        result = await parse_response_tokens(
            mock_chat_response_without_routed_experts, "chat"
        )

        assert result is not None
        assert result["routed_experts"] is None

    @pytest.mark.asyncio
    async def test_truncates_routed_experts_with_max_seq_len(
        self, mock_chat_response_with_routed_experts
    ):
        # max_seq_len = 5, full sequence is 7, so truncate to 5
        result = await parse_response_tokens(
            mock_chat_response_with_routed_experts, "chat", max_seq_len=5
        )

        assert result is not None
        assert result["is_truncated"] is True
        assert result["routed_experts"] is not None
        # Truncated to max_seq_len
        assert len(result["routed_experts"]) == 5
        # Still starts with prompt tokens
        assert result["routed_experts"][0] == [[0, 1], [2, 3]]

    @pytest.mark.asyncio
    async def test_overlong_prompt_truncates_routed_experts(self):
        """When prompt is longer than max_seq_len, routed_experts is truncated."""
        from openai.types.chat.chat_completion import ChatCompletion

        response = MagicMock(spec=ChatCompletion)
        choice = MagicMock()

        # Long prompt
        response.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
        choice.token_ids = [11, 12]  # 2 completion tokens
        choice.finish_reason = "stop"

        logprobs_content = [MagicMock(logprob=-0.1), MagicMock(logprob=-0.2)]
        choice.logprobs = MagicMock()
        choice.logprobs.content = logprobs_content

        # routed_experts for full sequence (12 tokens)
        choice.routed_experts = np.arange(12 * 2 * 2).reshape(12, 2, 2)

        response.choices = [choice]

        # max_seq_len smaller than prompt
        result = await parse_response_tokens(response, "chat", max_seq_len=5)

        assert result is not None
        assert result["overlong_prompt"] is True
        assert result["routed_experts"] is not None
        # Truncated to max_seq_len
        assert len(result["routed_experts"]) == 5


class TestParseResponseTokensCompletionWithRoutedExperts:
    """Tests for parse_response_tokens with completion message type and routed experts."""

    @pytest.fixture
    def mock_completion_response_with_routed_experts(self):
        """Create a mock Completion with routed_experts."""
        from openai.types.completion import Completion

        response = MagicMock(spec=Completion)
        choice = MagicMock()

        # Setup basic token data
        choice.prompt_token_ids = [1, 2]  # 2 prompt tokens
        choice.token_ids = [3, 4, 5]  # 3 completion tokens
        choice.finish_reason = "stop"

        # Setup logprobs
        choice.logprobs = MagicMock()
        choice.logprobs.token_logprobs = [-0.1, -0.2, -0.3]

        # Setup routed_experts - full sequence [5 tokens, 2 layers, 2 topk]
        choice.routed_experts = np.array([
            [[0, 1], [2, 3]],  # prompt token 0
            [[4, 5], [6, 7]],  # prompt token 1
            [[8, 9], [10, 11]],  # completion token 0
            [[12, 13], [14, 15]],  # completion token 1
            [[16, 17], [18, 19]],  # completion token 2
        ])

        response.choices = [choice]
        return response

    @pytest.mark.asyncio
    async def test_extracts_full_sequence_routed_experts_from_completion(
        self, mock_completion_response_with_routed_experts
    ):
        result = await parse_response_tokens(
            mock_completion_response_with_routed_experts, "completion"
        )

        assert result is not None
        assert result["routed_experts"] is not None
        # Full sequence: 2 prompt + 3 completion = 5 tokens
        assert len(result["routed_experts"]) == 5
        # Check prompt token routing is included
        assert result["routed_experts"][0] == [[0, 1], [2, 3]]
        # Check completion token routing
        assert result["routed_experts"][2] == [[8, 9], [10, 11]]
